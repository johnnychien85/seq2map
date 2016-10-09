#include <boost\thread.hpp>
#include <seq2map\solve.hpp>

using namespace seq2map;

size_t LeastSquaresProblem::GetHardwareConcurrency()
{
    return boost::thread::hardware_concurrency();
}

cv::Mat LeastSquaresProblem::ComputeJacobian(const VectorisableD::Vec& x, VectorisableD::Vec& y) const
{
    cv::Mat J = cv::Mat::zeros(static_cast<int>(m_conds), static_cast<int>(m_varIdx.size()), CV_64F);
    bool masking = !m_jacobianPattern.empty();
    std::vector<boost::thread> threads;

    y = y.empty() ? Evaluate(x) : y;

    // make evaluation slices for multi-threaded numerical differentiation
    std::vector<JacobianSlices> slices(m_diffThreads);
    size_t i = 0;

    for (Indices::const_iterator var = m_varIdx.begin(); var != m_varIdx.end(); var++)
    {
        JacobianSlice slice;
        size_t threadIdx = i % m_diffThreads;

        slice.x    = x;
        slice.y    = y;
        slice.var  = *var;
        slice.col  = J.col(static_cast<int>(i));
        slice.mask = masking ? m_jacobianPattern.col(static_cast<int>(*var)) : cv::Mat();

        slices[threadIdx].push_back(slice);

        i++;
    }

    // lunch the threads
    for (size_t k = 0; k < slices.size(); k++)
    {
        threads.push_back(boost::thread(LeastSquaresProblem::DiffThread, this, slices[k]));
    }

    // assign Jacobian column(s) to each differentiation thread
    /*
    Indices::const_iterator& var0 = m_varIdx.begin();
    for (size_t threadIdx = 0; threadIdx < m_diffThreads; threadIdx++)
    {
        JacobianSlices slices;

        //for (size_t var = threadIdx; var < m_vars; var += m_diffThreads)
        for (Indices::const_iterator& var = var0; var != m_varIdx.end(); var = var + m_diffThreads)
        {
            JacobianSlice slice;

            slice.x    = x;
            slice.y    = y;
            slice.var  = var;
            
            slices.push_back(slice);
        }

        // lunch the thread
        threads.push_back(boost::thread(LeastSquaresProblem::DiffThread, this, slices));
    }
    */

    // wait for completions
    BOOST_FOREACH(boost::thread& thread, threads)
    {
        thread.join();
    }

    return J;
}

void LeastSquaresProblem::DiffThread(const LeastSquaresProblem* lsq, JacobianSlices& slices)
{
    BOOST_FOREACH (JacobianSlice& slice, slices)
    {
        cv::Mat x = cv::Mat(slice.x).clone();
        cv::Mat y = cv::Mat(slice.y).clone();
        double dx = lsq->m_diffStep;

        x.at<double>(static_cast<int>(slice.var)) += dx;
        cv::Mat dy = cv::Mat(lsq->Evaluate(x)) - y;

        // Jk = (f(x+dx) - f(x)) / dx
        cv::divide(dy, dx, slice.col);
    }
}

bool LeastSquaresProblem::SetActiveVars(const Indices& varIdx)
{
    BOOST_FOREACH(size_t var, varIdx)
    {
        if (var >= m_vars) return false;
    }

    m_varIdx = varIdx;

    return true;
}

cv::Mat LeastSquaresProblem::ApplyUpdate(const VectorisableD::Vec& x0, const VectorisableD::Vec& delta)
{
    assert(delta.size() <= x0.size());
    VectorisableD::Vec x = x0;
    size_t i = 0;
    
    BOOST_FOREACH(size_t var, m_varIdx)
    {
        //E_INFO << var << " -> " << delta[i];
        x[var] += delta[i++];
    }

    //E_INFO << x0.size();
    //E_INFO << delta.size();

    return cv::Mat(x).clone();
}

bool LevenbergMarquardtAlgorithm::Solve(LeastSquaresProblem& problem, const VectorisableD::Vec& x0)
{
    assert(m_eta > 1.0f);

    double lambda = m_lambda;
    bool converged = false;
    std::vector<double> derr;

    VectorisableD::Vec x_best = x0;
    VectorisableD::Vec y_best = problem.Evaluate(x0);
    double e_best = rms(cv::Mat(y_best));

    size_t updates = 0; // iteration number
    
    if (m_verbose)
    {
        E_INFO << std::setw(6) << std::right << "Update" << std::setw(12) << std::right << "RMSE" << std::setw(16) << std::right << "lambda" << std::setw(16) << std::right << "Rel. Step Size" << std::setw(16) << std::right << "Rel. Error Drop";
        E_INFO << std::setw(6) << std::right << updates << std::setw(12) << std::right << e_best << std::setw(16) << std::right << lambda;
    }

    try
    {
        while (!converged)
        {
            cv::Mat J = problem.ComputeJacobian(x_best, y_best);

            cv::Mat H = J.t() * J; // Hessian matrix
            cv::Mat D = J.t() * cv::Mat(y_best); // error gradient

            lambda = lambda == 0 ? 1e-3 * cv::mean(H.diag())[0] : lambda;

            bool better = false;
            double derrRatio, stepRatio;

            while (!better && !converged)
            {
                // augmented normal equations
                cv::Mat A = H + lambda * cv::Mat::diag(H.diag()); // or A = N + lambda*eye(d);
                cv::Mat x_delta = A.inv() * -D; // x_delta =  A \ -D;

                VectorisableD::Vec x_try = problem.ApplyUpdate(x_best, x_delta); // = cv::Mat(cv::Mat(x_best) + x_delta);
                VectorisableD::Vec y_try = problem.Evaluate(x_try);

                double e_try = rms(cv::Mat(y_try));
                double de = e_best - e_try;
                better = de > 0;

                if (better) // accept the update
                {
                    lambda /= m_eta;

                    x_best = x_try;
                    y_best = y_try;
                    e_best = e_try;

                    derr.push_back(de);
                    updates++;
                }
                else // reject the update
                {
                    lambda *= m_eta;
                }

                // convergence control
                derrRatio = derr.size() > 1 ? (derr[derr.size() - 1] / derr[derr.size() - 2]) : 1.0f;
                stepRatio = norm(x_delta) / norm(cv::Mat(x_best));

                converged |= (updates >= m_term.maxCount); // # iterations check
                converged |= (updates > 1) && (derrRatio < m_term.epsilon); // error differential check
                converged |= (updates > 1) && (stepRatio < m_term.epsilon); // step ratio check
            }

            if (m_verbose)
            {
                E_INFO << std::setw(6) << std::right << updates << std::setw(12) << std::right << e_best << std::setw(16) << std::right << lambda << std::setw(16) << std::right << stepRatio << std::setw(16) << std::right << derrRatio;
            }

            //if (updates == 1 && mat2raw(J, "jacob.raw"))
            //{
            //    E_INFO << "Jacobian (" << size2string(J.size()) << ") written";
            //}
        }
    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught in optimisation loop";
        E_ERROR << ex.what();
        return false;
    }

    if (!problem.SetSolution(x_best))
    {
        E_ERROR << "error setting solution";
        E_ERROR << mat2string(cv::Mat(x_best), "x_best");

        return false;
    }

    return true;
}

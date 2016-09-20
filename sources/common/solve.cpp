#include <boost\thread.hpp>
#include <seq2map\solve.hpp>

using namespace seq2map;

size_t LeastSquaresProblem::GetHardwareConcurrency()
{
    return boost::thread::hardware_concurrency();
}

cv::Mat LeastSquaresProblem::ComputeJacobian(const cv::Mat& x, cv::Mat& y) const
{
    cv::Mat J = cv::Mat::zeros(static_cast<int>(m_conds), static_cast<int>(m_vars), CV_64F);
    std::vector<boost::thread> threads;

    y = y.empty() ? Evaluate(x) : y;

    // assign Jacobian column(s) to each differentiation thread
    for (size_t threadIdx = 0; threadIdx < m_diffThreads; threadIdx++)
    {
        JacobianSlices slices;
        for (size_t var = threadIdx; var < m_vars; var += m_diffThreads)
        {
            JacobianSlice slice;

            slice.x   = x;
            slice.y   = y;
            slice.var = var;
            slice.col = J.col(static_cast<int>(var));
            slices.push_back(slice);
        }

        // lunch the thread
        threads.push_back(boost::thread(LeastSquaresProblem::DiffThread, this, slices));
    }

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
        cv::Mat x = slice.x.clone();
        cv::Mat y = slice.y;
        double dx = lsq->m_diffStep;

        x.at<double>(static_cast<int>(slice.var)) += dx;
        cv::Mat dy = lsq->Evaluate(x) - y;

        // Jk = (f(x+dx) - f(x)) / dx
        cv::divide(dy, dx, slice.col);
    }
}

bool LevenbergMarquardtAlgorithm::Solve(LeastSquaresProblem& problem, const cv::Mat& x0)
{
    assert(m_eta > 1.0f);

    double lambda = m_lambda;
    bool converged = false;
    std::vector<double> derr;

    cv::Mat x_best = x0.clone();
    cv::Mat y_best = problem.Evaluate(x0);
    double  e_best = rms(y_best);

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
            cv::Mat H = J.t() * J;      // Hessian matrix
            cv::Mat D = J.t() * y_best; // error gradient

            lambda = lambda == 0 ? 1e-3 * cv::mean(H.diag())[0] : lambda;

            bool better = false;
            double derrRatio, stepRatio;

            while (!better && !converged)
            {
                // augmented normal equations
                cv::Mat A = H + lambda * cv::Mat::diag(H.diag()); // or A = N + lambda*eye(d);
                cv::Mat x_delta = A.inv() * -D; // x_delta =  A \ -D;

                cv::Mat x_try = x_best + x_delta;
                cv::Mat y_try = problem.Evaluate(x_try);
                double  e_try = rms(y_try);

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
                stepRatio = norm(x_delta) / norm(x_best);

                converged |= (updates >= m_term.maxCount); // # iterations check
                converged |= (updates > 1) && (derrRatio < m_term.epsilon); // error differential check
                converged |= (updates > 1) && (stepRatio < m_term.epsilon); // step ratio check
            }

            if (m_verbose)
            {
                E_INFO << std::setw(6) << std::right << updates << std::setw(12) << std::right << e_best << std::setw(16) << std::right << lambda << std::setw(16) << std::right << stepRatio << std::setw(16) << std::right << derrRatio;
            }

            // progress reporting
            //if (m_verbose)
            //{
            //    E_INFO << updates << ": " << e_best << ", lambda=" << lambda << ", relStepSize=" << stepRatio << ", relErrorDrop=" << derrRatio;
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
        E_ERROR << mat2string(x_best, "x_best");

        return false;
    }

    return true;
}

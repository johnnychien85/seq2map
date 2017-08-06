#include <boost/thread.hpp>
#include <seq2map/solve.hpp>

using namespace seq2map;

//==[ LeastSquaresProblem ]===================================================//

size_t LeastSquaresProblem::GetHardwareConcurrency()
{
    return boost::thread::hardware_concurrency();
}

VectorisableD::Vec LeastSquaresProblem::operator() (const VectorisableD& vec) const
{
    VectorisableD::Vec v;

    if (!vec.Store(v))
    {
        E_ERROR << "vectorisation failed";
        return VectorisableD::Vec();
    }

    return (*this)(v);
}

cv::Mat LeastSquaresProblem::ComputeJacobian(const VectorisableD::Vec& x, VectorisableD::Vec& y) const
{
    cv::Mat J = cv::Mat::zeros(static_cast<int>(m_conds), static_cast<int>(m_varIdx.size()), CV_64F);
    bool masking = !m_jacobianPattern.empty();

    boost::thread_group threads;

    y = y.empty() ? (*this)(x) : y;

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
        //threads.push_back(boost::thread(
        threads.add_thread(new boost::thread(
            LeastSquaresProblem::DiffThread,
            (const LeastSquaresProblem*) this,
            slices[k])
        );
    }

    // wait for completions
    threads.join_all();

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
        cv::Mat dy = cv::Mat((*lsq)(x)) - y;

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
        x[var] += delta[i++];
    }

    return cv::Mat(x, true);
}

//==[ LeastSquaresSolver ]====================================================//

bool LeastSquaresSolver::Logger::operator() (const LeastSquaresSolver::State& state)
{
    if (state.updates == 0)
    {
        E_TRACE << std::setw(80) << std::setfill('=') << "";
        E_TRACE << std::setw( 6) << std::right << "Update" 
                << std::setw(14) << std::right << "RMSE"
                << std::setw(20) << std::right << "Lambda"
                << std::setw(20) << std::right << "Rel. Step Size"
                << std::setw(20) << std::right << "Rel. Error Drop";
        E_TRACE << std::setw(80) << std::setfill('=') << "";
        E_TRACE << std::setw( 6) << std::right << state.updates
                << std::setw(14) << std::right << state.error
                << std::setw(20) << std::right << state.lambda
                << std::setw(20) << std::right << "-"
                << std::setw(20) << std::right << "-";
    }
    else
    {
        E_TRACE << std::setw(6)  << std::right << state.updates
                << std::setw(14) << std::right << state.error
                << std::setw(20) << std::right << state.lambda
                << std::setw(20) << std::right << state.relStep
                << std::setw(20) << std::right << state.relError;
    }

    return true;
}

//==[ LevenbergMarquardtAlgorithm ]===========================================//

bool LevenbergMarquardtAlgorithm::Solve(LeastSquaresProblem& f)
{
    assert(m_eta > 1.0f);
    State state;

    if (!f.Initialise(state.x))
    {
        return false;
    }

    state.y = f(state.x);
    state.error = rms(cv::Mat(state.y, false));
    state.lambda = m_lambda;
    state.converged = false;
    state.updates = 0;

    if (m_verbose && m_updater)
    {
        (*m_updater)(state);
    }

    try
    {
        //
        // outer loop : goes until a possible minimum is approached, or termination criterion met
        //
        while (!state.converged)
        {
            cv::Mat J = f.ComputeJacobian(state.x, state.y); // Jacobian matrix
            cv::Mat H = J.t() * J;                           // Hessian matrix
            cv::Mat D = J.t() * cv::Mat(state.y);            // error gradient

            state.lambda = state.lambda < 0 ? cv::mean(H.diag())[0] : state.lambda;

            bool better = false;
            size_t trials = 0;

            //
            // inner loop : keep trying
            //
            while (!better && !state.converged)
            {
                // augmented normal equations
                cv::Mat A = H + state.lambda * cv::Mat::diag(H.diag()); // or A = N + lambda*eye(d);
                cv::Mat x_delta = A.inv() * -D; // x_delta =  A \ -D;

                VectorisableD::Vec x = f.ApplyUpdate(state.x, x_delta); // = cv::Mat(cv::Mat(x_best) + x_delta);
                VectorisableD::Vec y = f(x);

                const double e_try = rms(cv::Mat(y));
                const double de = state.error - e_try;

                better = de > 0;
                trials++;

                if (better) // accept the update
                {
                    state.lambda /= m_eta;

                    state.de.push_back(de);
                    state.x        = x;
                    state.y        = y;
                    state.error    = e_try;
                    state.relError = state.de.size() > 1 ? (state.de.rbegin()[0] / state.de.rbegin()[1]) : 1.0f;
                    state.relStep  = cv::norm(x_delta) / cv::norm(cv::Mat(x, false));
                    state.jacobian = J;
                    state.updates++;
                }
                else // reject the update
                {
                    state.lambda *= m_eta;
                }

                // convergence control
                state.converged |= (state.updates >= m_term.maxCount); // # iterations check
                state.converged |= (state.updates > 1) && (state.relError < m_term.epsilon); // error differential check
                state.converged |= (state.updates > 1) && (state.relStep  < m_term.epsilon); // step ratio check
                state.converged |= (!better && trials >= m_term.maxCount);

                cv::Mat icv = H.diag();

                bool ill = false;

                for (int d = 0; d < icv.total(); d++)
                {
                    if (icv.at<double>(d) == 0)
                    {
                        E_WARNING << "change of parameter " << d << " not responsive";
                        ill = true;
                    }
                }

                ill |= !isfinite(norm(x_delta));

                if (ill)
                {
                    PersistentMat(J).Store(Path("J.bin"));
                    PersistentMat(x_delta).Store(Path("x_delta.bin"));
                    PersistentMat(cv::Mat(x)).Store(Path("x.bin"));
                    PersistentMat(cv::Mat(y)).Store(Path("y.bin"));

                    E_ERROR << "problem ill-posed";

                    return false;
                }
            }

            if (m_verbose && m_updater)
            {
                if (!(*m_updater)(state))
                {
                    return true;
                }
            }
        }
    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught in optimisation loop";
        E_ERROR << ex.what();

        return false;
    }

    if (!f.Finalise(state.x))
    {
        E_ERROR << "error setting solution";
        E_ERROR << mat2string(cv::Mat(state.x, false), "x_final");

        return false;
    }

    return true;
}

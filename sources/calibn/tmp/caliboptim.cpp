#include <calibn/caliboptim.hpp>
#include <calibn/helpers.hpp>
#include <iomanip>
#include <boost/thread.hpp>

void WorkerThread(CalibOptim*, const Indices&);

size_t CalibOptim::Optimise(const cv::TermCriteria& criterion, size_t threads)
{
	m_termCriteria = criterion;
	m_threads = threads;
	m_state.errors = std::vector<double>(m_termCriteria.maxCount + 1);

	m_state.x = MakeIntialGuess(); // get the initial estimate
	m_state.y = Evaluate(m_state.x); // and use it to evaluate the objective function

	assert(m_state.x.rows > 0 && m_state.x.cols == 1 && m_state.y.rows > 0 && m_state.y.cols == 1);

	m_state.errors[0] = rmse(m_state.y);

	// data dimensions
	m_state.d = m_state.x.rows; // number of parameters
	m_state.n = m_state.y.rows; // number of constraints

	// mask to accelerate function evaluation
	m_state.evalMask = std::vector<Mask>(m_state.d);

	// convergence control
	m_state.lambda = 0; // doesn't matter here, the value will be estimated in the first iteration
	m_state.converged = false;
	m_state.i = 0;
	m_state.evals = 0;

	// initialise thread indices
	m_state.threadIdx = std::vector<Indices>(m_threads);
	for (size_t idx = 0; idx < m_state.d; idx++)
	{
		m_state.threadIdx[idx % m_threads].push_back(idx);
	}

	// start iterative optimisation
	using namespace std;
	
	cout << "Starting optimising " << m_state.d << " parameter(s) with " << m_state.n << " condition(s):" << endl;
	cout <<	left << setw(16) << "epsilon =" << "[" << right << setw(16) << setprecision(5) << scientific << m_termCriteria.epsilon << "]" << endl;
	cout << left << setw(16) << "#iterations =" << "[" << right << setw(16) << m_termCriteria.maxCount << "]" << endl;
	cout << left << setw(16) << "#threads =" << "[" << right << setw(16) << m_threads << "]" << endl;
	cout << left << setw(16) << "initial err =" << "["  << right << setw(16) << setprecision(5) << fixed << m_state.errors[0] << "]" << endl;
	cout << left;
	cout << setfill('-') << setw(80 - 1) << "" <<endl;
	cout << setfill(' ') << setw(8) << "# Itr" << setw(8) << "Evals" << setw(16) << "Error" << setw(16) << "Improvement (%)" << setw(16) << "Step Size (%)" << setw(16 - 1) << "Lambda" << endl;
	cout << setfill('-') << setw(80 - 1) << "" << endl;
	clock_t t0 = clock();
	m_state.tSolve	= m_state.tDiff = 0;
	while (Loop());
	m_state.tSum = clock() - t0;
	cout << setfill('-') << setw(80 - 1) << "" << endl;
	cout << "Optimisation finished, " << fixed << setprecision(2) << (1000.0f * m_state.tSum) / CLOCKS_PER_SEC << " milliseccond(s) elapsed." << endl;

	// finished
	Finalise(m_state);

	return m_state.i;
}

bool CalibOptim::Loop()
{
	if (m_state.converged)
	{
		return false;
	}

	try
	{
		// approximate Jocobian matrix by numerical differentiation
		m_state.J = cv::Mat::zeros((int)m_state.n, (int)m_state.d, CV_64F);

		clock_t t0 = clock();
		boost::thread* threads = new boost::thread[m_threads];
		/** launch differentiation thrads ******/ for (size_t j = 0; j < m_threads; j++) threads[j] = boost::thread(WorkerThread, this, m_state.threadIdx[j]);
		/** .. and wait for their completions **/ for (size_t j = 0; j < m_threads; j++) threads[j].join();
		delete[] threads;
		m_state.tDiff += (clock() - t0);

		cv::Mat D = m_state.J.t() * m_state.y; // error gradiant
		cv::Mat H = m_state.J.t() * m_state.J; // Hessian matrix

		if (m_state.lambda == 0) // set initial value of lambda
		{
			m_state.lambda = 1e-3 * cv::mean(H.diag())[0]; //lambda = 1e-3 * mean(diag(N));
		}

        bool better = false;
        cv::Mat x2, y2, x_delta;
        
        while (!better)
        {
		    // compute delta by solving the augmented normal equations
		    t0 = clock();
		    cv::Mat A = H + m_state.lambda * cv::Mat::diag(H.diag()); // or A = N + lambda*eye(d);
		    //Mat A = H + m_state.lambda * Mat::eye(H.rows, H.cols, H.type()); // or A = N + lambda*eye(d);
		    x_delta	=  A.inv() * -D; // x_delta =  A \ -D;
		    m_state.tSolve += (clock() - t0);

		    // estimate standard error
		    cv::Mat C = (0.5 * H).inv(); // C = (H/2)^-1
		    cv::sqrt(C.diag(), m_state.sigma); // sigma(i) = sqrt(C(i,i))

		    // test new parameters
		    x2 = m_state.x + x_delta;
		    y2 = Evaluate(x2); // evaluate the improvement
		    m_state.de	= rmse(y2) - m_state.errors[m_state.i];
		    m_state.H	= H;
		    better	    = m_state.de < 0;

		    t0 = clock();
		    m_state.dt = m_state.t - t0;
		    m_state.t = t0;

		    if (!better) // a worse result?
		    {
			    m_state.lambda *= eta;	// try a new lambda,
			    //return true; //  reject the update
		    }
		}

		m_state.lambda /= eta;
		m_state.x = x2; // accept the update
		m_state.y = y2;

		m_state.errors[++m_state.i] = m_state.errors[m_state.i-1] + m_state.de;

		// check convengence conditions
		m_state.derrRatio = std::abs(m_state.de / m_state.errors[m_state.i]);
		m_state.stepRatio = cv::norm(x_delta) / cv::norm(m_state.x);

		m_state.converged = false;
		m_state.converged |= (m_state.i >= m_termCriteria.maxCount); // # iterations check
		m_state.converged |= (m_state.i > 1) && (m_state.derrRatio < m_termCriteria.epsilon); // error differential check
		m_state.converged |= (m_state.i > 1) && (m_state.stepRatio < m_termCriteria.epsilon); // step ratio check

        using namespace std;

		cout << setfill(' ') << setw(8) << m_state.i;
		cout << setfill(' ') << setw(8) << m_state.evals;
		cout << setfill(' ') << setw(16) << setprecision(5) << scientific << m_state.errors[m_state.i];
		cout << setfill(' ') << setw(16) << setprecision(5) << scientific << m_state.derrRatio;
		cout << setfill(' ') << setw(16) << setprecision(5) << scientific << m_state.stepRatio;
		cout << setfill(' ') << setw(15) << setprecision(5) << scientific << m_state.lambda << endl;

		return !m_state.converged;

	}
	catch (std::exception& ex)
	{
		E_ERROR << "error iterating";
        E_ERROR << ex.what();
		m_state.i = (size_t) -1;

		return false;
	}

	return true;
}

cv::Mat CalibOptim::Differentiate(const cv::Mat& x, const cv::Mat& y, size_t j, const std::vector<bool>& mask) const
{
	clock_t t0 = clock();

	cv::Mat x2 = x.clone(); x2.at<double>((int)j,0) += dx;
	cv::Mat y2 = Evaluate(x2, m_state.evalMask[j]);
	cv::Mat dy = y2 - y;

	return dy;
}

void WorkerThread(CalibOptim* optim, const Indices& threadIdx)
{
	for (Indices::const_iterator idx = threadIdx.begin(); idx != threadIdx.end(); idx++)
	{
		// dereferencing
		size_t d = *idx;
		Mask& mask = optim->m_state.evalMask[d];

		// should we build the evaluation mask for
		//  skipping unrelated parameter-condition pairs?
		bool buildMask = mask.empty();

		if (buildMask) // initialise the mask
		{
			mask = Mask(optim->m_state.n, true);
		}

		// differentiation
		cv::Mat dy = optim->Differentiate(optim->m_state.x, optim->m_state.y, d, mask);
		//cout << threadIdx[0] << " DIFF " << d << endl;

		if (buildMask)
		{
			for (size_t k = 0; k < mask.size(); k++)
			{
				mask[k] = (dy.at<double>((int)k,0) != 0);
			}
		}
		else
		{
			for (size_t k = 0; k < mask.size(); k++)
			{
				if (!mask[k]) dy.at<double>((int)k,0) = 0;
			}
		}

		divide(dy, optim->dx, optim->m_state.J.col(d)); //J = dy / dx;	
		optim->m_state.evals++; // TODO: protect this with a mutex lock
	}
}

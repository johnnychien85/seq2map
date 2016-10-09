#ifndef CALIBOPTIM_HPP
#define CALIBOPTIM_HPP

#include <ctime>
#include <opencv2/opencv.hpp>
#include <calibn/helpers.hpp>

typedef std::vector<bool> Mask;

struct CalibOptimState
{
	clock_t		   t;			// timestamp of the last state update
	clock_t		   dt;			// time interval between the last two updates
	clock_t		   tDiff;		// total time spent on differentiation
	clock_t		   tSolve;		// total time spent on solving linear system
	clock_t		   tSum;		// total optimisation time
	size_t		   n;			// number of equations
	size_t		   d;			// number of parameters
	size_t		   i;			// iteration index
	size_t		   evals;		// number of function evaluations
	double		   lambda;		// the damping parameter of LM algorithm
	cv::Mat        x0;			// initial solution
	cv::Mat        y0;			// initial y0 = f(x0)
	cv::Mat        x;			// last solution
	cv::Mat        y;			// last achivement
	cv::Mat        J;			// Jacobian matrix
	cv::Mat        H;			// Hessian matrix
	cv::Mat		   sigma;		// standard errors of the estimated x
	std::vector<double> errors;	// error history
	double         de;			// error drop
	double         stepRatio;	// ration of delta x
	double         derrRatio;	// ration of delta y
	bool           converged;	// convergence indicator
	std::vector<Mask> evalMask;	//
	std::vector<Indices> threadIdx;
};

class CalibOptim
{
public:
	/* ctor */			CalibOptim() : dx(1e-2), eta(10.0f), m_threads(1) {};
	/* dtor */ virtual	~CalibOptim() {};
	size_t				Optimise(const cv::TermCriteria& termCriteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-3), size_t threads = 1);
	friend void			WorkerThread(CalibOptim* optim, const Indices& threadIdx);

protected:
	virtual cv::Mat		MakeIntialGuess() = 0;
	virtual cv::Mat		Differentiate(const cv::Mat& x, const cv::Mat& y, size_t j, const Mask& mask = Mask(0)) const; // numerical differentiation
	virtual cv::Mat		Evaluate(const cv::Mat& x, const Mask& mask = Mask(0)) const = 0;
	virtual void		Finalise(const CalibOptimState& state) = 0;

	cv::TermCriteria    m_termCriteria;
	size_t				m_threads;

private:
	bool				Loop();

	//
	// internal-use only variables
	//
	CalibOptimState		m_state;
	double 				dx;			// for numerical differentiation
	double				eta;		// multiplier for the LM algorithm
};

#endif

#ifndef SOLVE_HPP
#define SOLVE_HPP

#include <seq2map\common.hpp>

namespace seq2map
{
    /**
     * Non-linear Least Squares Problem
     */
    class LeastSquaresProblem
    {
    public:
        LeastSquaresProblem(size_t m, size_t n, double dx = 1e-3, size_t diffThreads = 1)
            : m_conds(m), m_vars(n), m_diffStep(dx), m_diffThreads(diffThreads > 0 ? diffThreads : GetHardwareConcurrency()) {}
        virtual cv::Mat Initialise() = 0;
        virtual cv::Mat Evaluate(const cv::Mat& x) const = 0;
        virtual cv::Mat ComputeJacobian(const cv::Mat& x, cv::Mat& y = cv::Mat()) const;
        virtual bool SetSolution(const cv::Mat& x) = 0;
    protected:
        size_t m_conds;
        size_t m_vars;
        double m_diffStep;
        size_t m_diffThreads;
    private:
        struct JacobianSlice
        {
            size_t var;
            cv::Mat x;
            cv::Mat y;
            cv::Mat col;
        };

        typedef std::vector<JacobianSlice> JacobianSlices;
        static size_t GetHardwareConcurrency();
        static void DiffThread(const LeastSquaresProblem* lsq, JacobianSlices& slices);
    };

    /**
     * Levenberg-Marquardt algorithm
     */
    class LevenbergMarquardtAlgorithm
    {
    public:
        LevenbergMarquardtAlgorithm(double eta = 10.0f, double lambda = 0.0f, bool verbose = false, 
            const cv::TermCriteria& term = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6))
            : m_eta(eta), m_lambda(lambda), m_verbose(verbose), m_term(term) {}
        inline void SetVervbose(bool verbose) { m_verbose = verbose; }
        bool Solve(LeastSquaresProblem& problem, const cv::Mat& x0);
    protected:
        double m_eta;
        double m_lambda;
        bool m_verbose;
        cv::TermCriteria m_term;
    };
}
#endif // SOLVE_HPP

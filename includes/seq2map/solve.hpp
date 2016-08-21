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
        static size_t GetHardwareConcurrency();
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
        static void DiffThread(const LeastSquaresProblem* lsq, JacobianSlices& slices);
    };

    /**
     * Levenberg-Marquardt algorithm
     */
    class LevenbergMarquardtAlgorithm
    {
    public:
        LevenbergMarquardtAlgorithm(
            const cv::TermCriteria& term = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-3),
            double eta = 10.0f,
            bool verbose = false)
            : m_term(term), m_eta(eta), m_verbose(verbose) {}
        void SetVervbose(bool verbose) { m_verbose = verbose; }
        bool Solve(LeastSquaresProblem& problem, const cv::Mat& x0);
    protected:
        cv::TermCriteria m_term;
        double m_eta;
        bool m_verbose;
    };
}
#endif // SOLVE_HPP

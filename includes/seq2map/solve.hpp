#ifndef SOLVE_HPP
#define SOLVE_HPP

#include <seq2map/common.hpp>

namespace seq2map
{
    /**
     * Non-linear least squares problem.
     */
    class LeastSquaresProblem
    {
    public:
        //
        //
        //
        LeastSquaresProblem(size_t m, size_t n, const Indices& vars = Indices(), double dx = 1e-3, size_t diffThreads = 0)
        : m_conds(m), m_vars(n), m_diffStep(dx), m_diffThreads(diffThreads > 0 ? diffThreads : GetHardwareConcurrency())
        {
            if (vars.empty() || !SetActiveVars(vars)) m_varIdx = makeIndices(0, n-1); 
        }

        virtual VectorisableD::Vec operator() (const VectorisableD::Vec& x) const = 0;
        VectorisableD::Vec operator() (const VectorisableD& vec) const;

        virtual cv::Mat ComputeJacobian(const VectorisableD::Vec& x, VectorisableD::Vec& y) const;
        
        void SetDifferentiationStep(double step) { m_diffStep = step; }
        
        //
        //
        //
        bool SetActiveVars(const Indices& varIdx);

        /**
         * Initialise the state of problem and set initial guess.
         * The method is called by a solver at the beginning of solving process.
         *
         * \param x Initial guess
         */
        virtual bool Initialise(VectorisableD::Vec& x) { x = VectorisableD::Vec(m_conds, 0); return true; }

        /**
         * Apply an update to an estimate.
         * The method is called by a solver when an estimate needs to be updated.
         * By default x is set to x + delta.
         *
         * \param x Current estimate.
         * \param delta Difference to be applied to x.
         *
         * \return A row vector containing updated estimate.
         */
        virtual cv::Mat ApplyUpdate(const VectorisableD::Vec& x, const VectorisableD::Vec& delta);

        /**
         * Finalise the sate of problem and set final solution.
         * The method is called by a solver in the end of solving process.
         *
         * \param x Final solution.
         *
         * \return True if the solution is accepted, otherwise false.
         */
        virtual bool Finalise(const VectorisableD::Vec& x) = 0;

    protected:
        const size_t m_vars;  ///< number of variables to be estimated
        size_t       m_conds; ///< size of output vector

    private:
        struct JacobianSlice
        {
            size_t var;
            VectorisableD::Vec x;
            VectorisableD::Vec y;
            cv::Mat col;
            cv::Mat mask;
        };

        typedef std::vector<JacobianSlice> JacobianSlices;

        static size_t GetHardwareConcurrency();
        static void DiffThread(const LeastSquaresProblem* lsq, JacobianSlices& slices);

        Indices m_varIdx; ///< list of active variables
        double  m_diffStep;
        size_t  m_diffThreads;
        cv::Mat m_jacobianPattern;
    };

    class LeastSquaresSolver
    {
    public:
        /**
         * An instantaneous state of the solver.
         */
        struct State
        {
            State() : error(-1.0f), relStep(-1.0f), relError(-1.0f), lambda(-1.0f), updates(0), converged(false) {}
            void Report();

            VectorisableD::Vec x;
            VectorisableD::Vec y;
            VectorisableD::Vec de;
            double error;
            double relStep;
            double relError;
            cv::Mat jacobian;
            cv::Mat hessian;
            double lambda;
            size_t updates;
            bool converged;
        };

        /**
         * A handler to process state update event.
         */
        class UpdateHandler : public Referenced<UpdateHandler>
        {
        public:
            virtual bool operator() (const State& state) = 0;
        };

        /**
         *
         */
        class Logger : public UpdateHandler
        {
        public:
            virtual bool operator() (const State& state);
        };

        //
        // Constructor
        //
        LeastSquaresSolver()
        : m_verbose(false),
          m_term(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6)),
          m_updater(UpdateHandler::Own(new Logger)) {}

        //
        // Accessor
        //

        /**
         *
         */
        inline void SetVervbose(bool verbose) { m_verbose = verbose; }

        /**
         *
         */
        inline void SetTermCriteria(const cv::TermCriteria term) { m_term = term; }

        /**
         *
         */
        inline void SetUpdateHandler(UpdateHandler::Own& handler) { m_updater = handler; }

        //
        //
        //
        virtual bool Solve(LeastSquaresProblem& problem, State& state) = 0;

    protected:
        bool               m_verbose;
        cv::TermCriteria   m_term;
        UpdateHandler::Own m_updater;
    };

    /**
     * Levenberg-Marquardt algorithm
     */
    class LevenbergMarquardtAlgorithm : public LeastSquaresSolver
    {
    public:
        /**
         *
         */
        LevenbergMarquardtAlgorithm(double eta = 10.0f, double lambda = -1.0f)
        : m_eta(eta), m_lambda(lambda) {}
        
        /**
         *
         */
        inline void SetInitialDamp(double lambda) { m_lambda = lambda; }
        
        /**
         *
         */
        bool Solve(LeastSquaresProblem& problem, State& state = State());

    protected:
        double m_eta;
        double m_lambda;
    };


}
#endif // SOLVE_HPP

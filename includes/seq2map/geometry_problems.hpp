#ifndef GEOMETRY_PROBLEMS_HPP
#define GEOMETRY_PROBLEMS_HPP

#include <seq2map/geometry.hpp>
#include <seq2map/solve.hpp>

namespace seq2map
{
    /**
     * An alignment objective evaluates an Euclidean transform by applying it
     * to transform a source geometry data and comparing the resultant geometry with
     * the target.
     */
    class AlignmentObjective
    : public EuclideanTransform::Objective,
      public Referenced<AlignmentObjective>
    {
    public:
        /**
         * A helper to locate inliers in the evaluation data.
         */
        struct InlierSelector
        {
            struct Stats
            {
                Stats() : population(0), inliers(0), secs(0) {}

                size_t population;
                size_t inliers;

                double secs;
            };

            InlierSelector() : threshold(-1) {}

            InlierSelector(AlignmentObjective::ConstOwn& objective, double threshold)
            : objective(objective), threshold(threshold) {}

            bool operator() (const EuclideanTransform& tform, Indices& inliers) const;
            bool operator() (const EuclideanTransform& tform, Indices& inliers, Indices& outliers) const;
            inline bool IsEnabled() const { return threshold > 0; }

            AlignmentObjective::ConstOwn objective;
            double threshold;
            mutable Speedometre metre;
        };

        /**
         * Set evaluation data.
         * \param data evaluation data.
         * \return true if the data is accepted, otherwise false.
         */
        virtual bool SetData(const GeometricMapping& data) = 0;

        /**
         * Read-only data getter.
         */
        const GeometricMapping& GetData() const { return m_data; }

        /**
         * Make an InlierSelector using this objective.
         */
        InlierSelector GetSelector(double threshold) const { return InlierSelector(shared_from_this(), threshold); }

        /**
         * Make a new objective from a subset of data.
         */
        virtual AlignmentObjective::Own GetSubObjective(const Indices& indices) const = 0;

    protected:
        GeometricMapping m_data;
    };

    /**
     * Epipolar geometry derived objective functions.
     */
    class EpipolarObjective : public AlignmentObjective
    {
    public:
        enum DistanceType
        {
            ALGEBRAIC, // algebraic distance
            GEOMETRIC, // geometrical meaningfully normalised algebraic distance
            SAMPSON    // Sampson first-order approximation of the MLE estimation
        };

        //
        // Constructor and destructor
        //
        EpipolarObjective(DistanceType distType = SAMPSON) : m_src(new PinholeModel()), m_dst(m_src), m_distType(distType) {}
        EpipolarObjective(const ProjectionModel::ConstOwn& src, const ProjectionModel::ConstOwn& dst, DistanceType distance = SAMPSON)
        : m_src(src), m_dst(dst), m_distType(distance) {}

        //
        // Accessor
        //
        void SetDistance(DistanceType distType) { m_distType = distType; }
        virtual bool SetData(const GeometricMapping& data) { return SetData(data, m_src, m_dst); }
        bool SetData(const GeometricMapping& data, ProjectionModel::ConstOwn& src, ProjectionModel::ConstOwn& dst);

        virtual AlignmentObjective::Own GetSubObjective(const Indices& indices) const;

        //
        // Evaluation
        //
        virtual cv::Mat operator() (const EuclideanTransform& tform) const;

    private:
        ProjectionModel::ConstOwn m_src;
        ProjectionModel::ConstOwn m_dst;
        DistanceType m_distType;
    };

    /**
     * Perspective projection derived objective function.
     */
    class ProjectionObjective : public AlignmentObjective
    {
    public:
        //
        // Constructor and destructor
        //
        ProjectionObjective(const ProjectionModel::ConstOwn& proj, bool forward = true) : m_proj(proj), m_forward(forward) {}

        //
        // Accessor
        //
        virtual void SetProjectionModel(ProjectionModel::Own& proj) { m_proj = proj; }
        virtual bool SetData(const GeometricMapping& data);

        virtual AlignmentObjective::Own GetSubObjective(const Indices& indices) const;

        //
        // Evaluation
        //
        virtual cv::Mat operator() (const EuclideanTransform& tform) const;

    protected:
        ProjectionModel::ConstOwn m_proj;
        bool m_forward;

        struct Metres
        {
            Metres()
            : proj("Projection"           ),
              jaco("Jacobian computation" ),
              tfrm("Metric transformation"),
              eval("Metric evaluation"    ) {}

            Speedometre proj;
            Speedometre jaco;
            Speedometre tfrm;
            Speedometre eval;
        };

        mutable Metres m_metres;
    };

    /**
     * Image intensity derived photometric objective function.
     */
    class PhotometricObjective : public ProjectionObjective
    {
    public:
        //
        // Constructor and destructor
        //
        PhotometricObjective(const ProjectionModel::ConstOwn& proj, const cv::Mat& dst, int type = CV_32F, int interp = cv::INTER_LINEAR)
        : m_type(type), m_interp(interp), ProjectionObjective(proj)
        { dst.convertTo(m_dst, m_type); }

        //
        // Accessor
        //
        virtual bool SetData(const GeometricMapping& data);

        /**
         * Build objective data from 3D points and a base image
         */
        virtual bool SetData(const Geometry& g, const cv::Mat& src, const Metric::Own metric = Metric::Own(), const std::vector<size_t>& indices = std::vector<size_t>());

        virtual AlignmentObjective::Own GetSubObjective(const Indices& indices) const;

        //
        // Evaluation
        //
        virtual cv::Mat operator() (const EuclideanTransform& tform) const;

    private:
        int m_type;
        int m_interp;
        cv::Mat m_dst;
    };

    /**
     * Rigid alignment objective function.
     */
    class RigidObjective : public AlignmentObjective
    {
    public:
        //
        // Accessor
        //
        virtual bool SetData(const GeometricMapping& data);

        virtual AlignmentObjective::Own GetSubObjective(const Indices& indices) const;

        //
        // Evaluation
        //
        virtual cv::Mat operator() (const EuclideanTransform& tform) const;
    };

    /**
     * Abstract pose estimation problem.
     */
    class PoseEstimator
    : public Referenced<PoseEstimator> // by ConsensusPoseEstimator
    {
    public:
        struct Estimate
        {
            Estimate() : valid(false) {}

            EuclideanTransform pose;
            Metric::Own metric;
            bool valid;
        };

        /**
         * Solve relative pose from the source geometry to the target specified by mapping.
         *
         * \param mapping Correspondences used to find the pose.
         * \param estimate An estimate of pose.
         *
         * \return True if the estimate is valid, otherwise false.
         */
        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const = 0;
        
        /**
         * Get the minimum number of correspondences required to yeild a valid pose estimate.
         *
         * \return Number of correspondences minimally required.
         */
        virtual size_t GetMinPoints() const = 0;
    };

    /**
     * Pose estimation by essential matrix decomposition.
     */
    class EssentialMatrixDecomposer : public PoseEstimator
    {
    public:
        EssentialMatrixDecomposer(const ProjectionModel::ConstOwn& srcProj, const ProjectionModel::ConstOwn& dstProj)
        : m_srcProj(srcProj), m_dstProj(dstProj) {}

        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
        virtual size_t GetMinPoints() const { return 8; }

    private:
        const ProjectionModel::ConstOwn m_srcProj;
        const ProjectionModel::ConstOwn m_dstProj;
    };

    /**
     * Pose estimation by a perspective-n-points solver.
     */
    class PerspevtivePoseEstimator : public PoseEstimator
    {
    public:
        PerspevtivePoseEstimator(const ProjectionModel::ConstOwn& proj) : m_proj(proj) {}

        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
        virtual size_t GetMinPoints() const { return 6; }

    private:
        const ProjectionModel::ConstOwn m_proj;
    };

    /**
     * Pose estimation by Horn's quaternion-based absolute orientation solving method.
     */
    class QuatAbsOrientationSolver : public PoseEstimator
    {
    public:
        QuatAbsOrientationSolver() {}

        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
        virtual size_t GetMinPoints() const { return 3; }
    };

    /**
     * Dummy pose estimator always returns same pose.
     */
    class DummyPoseEstimator : public PoseEstimator
    {
    public:
        DummyPoseEstimator(const EuclideanTransform& pose) : m_pose(pose) {}

        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
        virtual size_t GetMinPoints() const { return 0; }

    private:
        const EuclideanTransform m_pose;
    };

    /**
     * Inverse pose estimator makes pose estimation by means of a desginated PoseEstimator and
     * returns the inverse of the estimated pose.
     */
    class InversePoseEstimator : public PoseEstimator
    {
    public:
        InversePoseEstimator(PoseEstimator::ConstOwn estimator) : m_estimator(estimator) {}

        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
        virtual size_t GetMinPoints() const { return m_estimator ? m_estimator->GetMinPoints() : 0; }

    private:
        const PoseEstimator::ConstOwn m_estimator;
    };

    /**
     * Robust pose estimation from data with presense of outliers.
     */
    class ConsensusPoseEstimator : public PoseEstimator
    {
    public:
        enum Strategy
        {
            RANSAC,
            LMEDS,
            MSAC
        };

        typedef std::vector<AlignmentObjective::InlierSelector> Selectors;
        typedef std::vector<Indices> IndexLists;

        //
        // Constructor
        //
        ConsensusPoseEstimator()
        : m_maxIter(100), m_minInlierRatio(0.5f), m_confidence(0.95f), m_optimisation(false), m_verbose(false) {}

        //
        // Pose estimation
        //
        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate, IndexLists& inliers, IndexLists& outliers) const;

        /**
         * Return the minimum number of correspondences the delegated inner pose solver requires.
         */
        virtual size_t GetMinPoints() const { return m_solver ? m_solver->GetMinPoints() : 0; }

        //
        // Accessors
        //
        inline void SetStrategy(Strategy strategy) { m_strategy = strategy; }
        inline Strategy GetStrategy() const { return m_strategy; }

        inline void SetMaxIterations(size_t maxIter) { m_maxIter = maxIter; }
        inline void SetMinInlierRatio(double ratio)  { m_minInlierRatio = ratio; }
        inline void SetConfidence(double p) { m_confidence = p > 1.0f ? 1.0f : (p < 0.0f ? 0.0f : p); }
        inline void EnableOptimisation()  { m_optimisation = true;  }
        inline void DisableOptimisation() { m_optimisation = false; }
        inline void SetVerbose(bool verbose) { m_verbose = verbose; }

        inline void AddSelector(const AlignmentObjective::InlierSelector& selector) { m_selectors.push_back(selector); }
        inline void SetSolver(PoseEstimator::ConstOwn& solver) { m_solver = solver; }
        inline Selectors& GetSelectors() { return m_selectors; }
        size_t GetPopulation() const;

    private:
        static Indices DrawSamples(size_t population, size_t samples);

        Strategy m_strategy;
        PoseEstimator::ConstOwn m_solver;
        Selectors m_selectors;

        size_t m_maxIter;
        double m_minInlierRatio;
        double m_confidence;
        bool m_optimisation;
        bool m_verbose;
    };

    /**
     * A generic multi-objective pose estimator.
     */
    class MultiObjectivePoseEstimation : public LeastSquaresProblem
    {
    public:
        //
        // Constructor
        //
        MultiObjectivePoseEstimation() : LeastSquaresProblem(0, 6), m_tform(Rotation::EULER_ANGLES) {}

        //
        // Accessors
        //
        inline void AddObjective(AlignmentObjective::ConstOwn& objective) { m_objectives.push_back(objective); }
        EuclideanTransform  GetPose() const         { return m_tform; }
        EuclideanTransform& GetPose()               { return m_tform; }
        void SetPose(const EuclideanTransform pose) { m_tform = pose; }

        //
        // Least-squares problem
        //
        virtual bool Initialise(VectorisableD::Vec& x);
        virtual VectorisableD::Vec operator()(const VectorisableD::Vec& x) const;
        virtual bool Finalise(const VectorisableD::Vec& x) { return m_tform.Restore(x); }

    private:
        size_t GetConds() const;

        std::vector<AlignmentObjective::ConstOwn> m_objectives;
        EuclideanTransform m_tform;
    };

    /**
     * Abstract structure estimation problem.
     */
    class StructureEstimation
    {
    public:
        /**
         * An estimate of structure containing geometry of points and an
         * optionally associated metric modelling the uncertainty of the
         * estimation.
         */
        struct Estimate
        {
            /**
             * Construct an empty estimate.
             *
             * \param shape The storage shape of the geometry.
             */
            Estimate(Geometry::Shape shape) : structure(shape) {}

            /**
             * Construct an estimate with given geometry.
             */
            Estimate(Geometry g) : structure(g) {}

            /**
             * Construct an estimate with given geometry and metric.
             */
            Estimate(Geometry g, Metric::Own& m) : structure(g), metric(m) {}

            /**
             * Retrieve subset of the estimated structure.
             *
             * \param indices List of indices of elements to be extracted from structure and metric.
             * \return extracted sub-elements.
             */
            Estimate operator[] (const Indices& indices) const;

            /**
             * Update the current estimate using a recursive Bayesian filter.
             * In current implementation both estimates have to be associted
             * with Mahanlanobis metrics, from which the error covariances
             * can be retrieved to derive the sum of two Gaussian distribution.
             */
            Estimate& operator+= (const Estimate& estimate);

            /**
             * Merge two estimates using a recursive Bayesian filter.
             *
             * \param estimate Another estimate transformed to the frame of the estimate being merged with.
             * \return fusion of two estimates.
             */
            Estimate operator+ (const Estimate& estimate) const;

            /**
             * Apply an Euclidean transform to the estimate.
             *
             * \param tform The transform to be applied to the underlying geometry and any associated metric.
             * \return Transformed estimate.
             */
            Estimate Transform(const EuclideanTransform& tform) const
            { return Estimate(tform(Geometry(structure)), metric ? metric->Transform(tform) : Metric::Own()); }

            Geometry structure;
            Metric::Own metric;
        };

        //
        // Constructor and destructor
        //
        StructureEstimation() {}
        virtual ~StructureEstimation() {}
    };

    /**
     * Structure estimation takes a set of projected 2D geometry to compute one 3D geometry by back-projection.
     */
    class TwoViewTriangulation : public StructureEstimation
    {
    public:
        TwoViewTriangulation(const PosedProjection& P0, const PosedProjection& P1, const EuclideanTransform& tform = EuclideanTransform::Identity)
        : M01(tform >> P1.pose), P0(P0), P1((PosedProjection(M01, P1.proj))) {}

        Estimate operator() (const GeometricMapping& m) const;

        const EuclideanTransform M01;
        const PosedProjection P0;
        const PosedProjection P1;

    protected:
        virtual Geometry operator() (const Geometry& x0, const Geometry& x1) const = 0;
        virtual Geometry GetJacobian(const Geometry& x0, const Geometry& x1, const Geometry& g) const;
    };

    /**
     * Linear triangulation is based on a Direct Linear Transformation (DLT) to find the algebraic
     * solution to the homogeneous 3D-to-2D projection equations.
     */
    // class LinearTriangulation : public TwoViewTriangulation
    // {
    //
    // };

    /**
     * Mid-point triangulation finds the middle point of the shortest line between two
     * rays back-projected from image points.
     */
    class MidPointTriangulation : public TwoViewTriangulation
    {
    public:
        MidPointTriangulation(const PosedProjection& P0, const PosedProjection& P1, const EuclideanTransform tform)
            : TwoViewTriangulation(P0, P1, tform) {}

        using TwoViewTriangulation::operator();

    protected:
        virtual Geometry operator() (const Geometry& x0, const Geometry& x1) const;
        static void DecomposeProjMatrix(const cv::Mat& P, cv::Mat& KRinv, cv::Mat& c);
    };

    /**
     * Optimal triangulation finds the 3D point that onced projected minimises the geodesic distances to the
     * image point following Hartley's work.
     */
    class OptimalTriangulation : public TwoViewTriangulation
    {
    public:
        OptimalTriangulation(const PosedProjection& P0, const PosedProjection& P1, const EuclideanTransform tform)
        : TwoViewTriangulation(P0, P1, tform) {}
        
        using TwoViewTriangulation::operator();

    protected:
        virtual Geometry operator() (const Geometry& x0, const Geometry& x1) const;
    };

    class MultipleViewBlockMatcher
    {
    public:
        struct RefView
        {
            RefView() : p(EuclideanTransform::Identity, NullProjection) {}
            RefView(cv::Mat im, PosedProjection& p) : im(im), p(p) {}

            static PinholeModel NullProjection;

            const cv::Mat im;
            const PosedProjection p;
        };

        typedef std::vector<RefView> RefViews;

        MultipleViewBlockMatcher(const cv::Mat baseImage, const ProjectionModel::ConstOwn& baseProj, const RefViews refs, int blockSize, bool perspectiveWarping = false)
        : m_baseImage(baseImage), m_baseProj(baseProj), m_blockSize(blockSize), m_perspectiveWarping(perspectiveWarping) {}

        StructureEstimation::Estimate operator() (const StructureEstimation::Estimate& g0, const std::vector<Points2F>& proj) const;

    private:
        const cv::Mat m_baseImage;
        const ProjectionModel::ConstOwn m_baseProj;

        int   m_blockSize;
        bool  m_perspectiveWarping;
    };
}
#endif // GEOMETRY_HPP

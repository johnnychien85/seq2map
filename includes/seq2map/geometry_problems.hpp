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
            InlierSelector(AlignmentObjective::Own& objective, double threshold)
            : objective(objective), threshold(threshold) {}

            Indices operator() (const EuclideanTransform& tform) const;
            inline bool IsEnabled() const { return threshold > 0; }

            AlignmentObjective::Own objective;
            double threshold;
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
        EpipolarObjective(ProjectionModel::ConstOwn& src, ProjectionModel::ConstOwn& dst, DistanceType distance = SAMPSON) : m_src(src), m_dst(dst), m_distType(distance) {}

        //
        // Accessor
        //
        void SetDistance(DistanceType distType) { m_distType = distType; }
        virtual bool SetData(const GeometricMapping& data) { return SetData(data, m_src, m_dst); }
        bool SetData(const GeometricMapping& data, ProjectionModel::ConstOwn& src, ProjectionModel::ConstOwn& dst);

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
        ProjectionObjective(ProjectionModel::ConstOwn& proj) : m_proj(proj) {}

        //
        // Accessor
        //
        virtual void SetProjectionModel(ProjectionModel::Own& proj) { m_proj = proj; }
        virtual bool SetData(const GeometricMapping& data);

        //
        // Evaluation
        //
        virtual cv::Mat operator() (const EuclideanTransform& tform) const;

    protected:
        ProjectionModel::ConstOwn m_proj;
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
        PhotometricObjective(ProjectionModel::ConstOwn& proj, const cv::Mat& dst, int type = CV_32F, int interp = cv::INTER_LINEAR)
        : m_type(type), m_interp(interp), ProjectionObjective(proj)
        { dst.convertTo(m_dst, m_type); }

        //
        // Accessor
        //
        virtual bool SetData(const GeometricMapping& data);

        /**
         * Build objective data from 3D points and a base image
         */
        virtual bool SetData(const Geometry& g, const cv::Mat& src);

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
            EuclideanTransform pose;
            Metric::Own metric;
        };

        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const = 0;
    };

    /**
     * Pose estimation by essential matrix decomposition.
     */
    class EssentialMatrixDecomposer : public PoseEstimator
    {
    public:
        EssentialMatrixDecomposer(ProjectionModel::ConstOwn& srcProj, ProjectionModel::ConstOwn& dstProj)
        : m_srcProj(srcProj), m_dstProj(dstProj) {}

        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;

    private:
        const ProjectionModel::ConstOwn m_srcProj;
        const ProjectionModel::ConstOwn m_dstProj;
    };

    /**
     * Pose estimation by a perspective-n-points solver
     */
    class PerspevtivePoseEstimator : public PoseEstimator
    {
    public:
        PerspevtivePoseEstimator(ProjectionModel::ConstOwn& proj) : m_proj(proj) {}
        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;

    private:
        const ProjectionModel::ConstOwn m_proj;
    };

    /**
     * Pose estimation by Horn's quaternion-based absolute orientation solving method
     */
    class QuatAbsOrientationSolver : public PoseEstimator
    {
    public:
        QuatAbsOrientationSolver(const RigidObjective& objective) {}
        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
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
            MASC
        };

        typedef std::vector<AlignmentObjective::InlierSelector> Selectors;

        //
        // Constructor
        //
        ConsensusPoseEstimator() {}

        //
        // Pose estimation
        //
        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate) const;
        virtual bool operator() (const GeometricMapping& mapping, Estimate& estimate, std::vector<Indices> inliers) const;

        //
        // Accessors
        //
        inline void SetStrategy(Strategy strategy) { m_strategy = strategy; }
        inline Strategy GetStrategy() const { return m_strategy; }

        inline void AddSelector(const AlignmentObjective::InlierSelector& selector) { m_selectors.push_back(selector); }
        inline void SetEstimator(PoseEstimator::ConstOwn& estimator) { m_estimator = estimator; }
        inline Selectors& GetSelectors() { return m_selectors; }

    private:
        Strategy m_strategy;
        PoseEstimator::ConstOwn m_estimator;
        Selectors m_selectors;
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
        inline void AddObjective(AlignmentObjective::Own& objective) { m_objectives.push_back(objective); }
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

        std::vector<AlignmentObjective::Own> m_objectives;
        EuclideanTransform m_tform;
    };

    /**
     * Abstract structure estimation problem.
     */
    class StructureEstimation
    {
    public:
        struct Estimate
        {
            Estimate(Geometry::Shape shape) : structure(shape) {}
            Estimate(Geometry g) : structure(g) {}

            Geometry structure;
            Metric::Own metric;
        };

        //
        // Constructor and destructor
        //
        StructureEstimation() {}
        virtual ~StructureEstimation() {}

        virtual Estimate operator() (const GeometricMapping& m) const = 0;
    };

    /**
     * Structure estimation takes a set of projected 2D geometry to compute one 3D geometry by back-projection.
     */
    class TwoViewTriangulation : public StructureEstimation
    {
    public:
        TwoViewTriangulation(const PosedProjection& P0, const PosedProjection& P1) : P0(P0), P1(P1) {}

        const PosedProjection P0;
        const PosedProjection P1;
    };

    class OptimalTriangulation : public TwoViewTriangulation
    {
        OptimalTriangulation(const PosedProjection& P0, const PosedProjection& P1) : TwoViewTriangulation(P0, P1) {}
        virtual Estimate operator() (const GeometricMapping& m) const;
    };

    class MidPointTriangulation : public TwoViewTriangulation
    {
    public:
        MidPointTriangulation(const PosedProjection& P0, const PosedProjection& P1) : TwoViewTriangulation(P0, P1) {}
        virtual Estimate operator() (const GeometricMapping& m) const;
    protected:
        static void DecomposeProjMatrix(const cv::Mat& P, cv::Mat& KRinv, cv::Mat& c);
    };

    /**
     * ........
     */
    /*
    template<class T0, class T1> class Mapping
    {
    public:
        Mapping() : m_size(0) {}
        size_t Add(const T0& pt0, const T1& pt1)
        {
            m_pts0.push_back(pt0);
            m_pts1.push_back(pt1);

            return m_size = m_pts0.size();
        }
        inline size_t GetSize() const { return m_size; }
        inline const std::vector<T0>& From() const { return m_pts0; }
        inline const std::vector<T1>& To()   const { return m_pts1; }
    protected:
        size_t m_size;
        std::vector<T0> m_pts0;
        std::vector<T1> m_pts1;
    };

    typedef Mapping<Point2D, Point2D> PointMap2Dto2D;
    typedef Mapping<Point3D, Point2D> PointMap3Dto2D;
    */
}
#endif // GEOMETRY_HPP

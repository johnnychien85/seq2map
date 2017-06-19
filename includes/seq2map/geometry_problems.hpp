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
            InlierSelector() : threshold(0.0f) {}
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
        enum Distance
        {
            ALGEBRAIC, // algebraic distance
            GEOMETRIC, // geometrical meaningfully normalised algebraic distance
            SAMPSON    // Sampson first-order approximation of the MLE estimation
        };

        //
        // Constructor and destructor
        //
        EpipolarObjective(Distance dist = SAMPSON) : m_src(new PinholeModel()), m_dst(m_src), m_dist(dist) {}
        EpipolarObjective(ProjectionModel::ConstOwn& src, ProjectionModel::ConstOwn& dst, Distance distance = SAMPSON) : m_src(src), m_dst(dst), m_dist(distance) {}

        //
        // Accessor
        //
        void SetDistance(Distance dist) { m_dist = dist; }
        virtual bool SetData(const GeometricMapping& data) { return SetData(data, m_src, m_dst); }
        bool SetData(const GeometricMapping& data, ProjectionModel::ConstOwn& src, ProjectionModel::ConstOwn& dst);

        //
        // Evaluation
        //
        virtual cv::Mat operator() (const EuclideanTransform& tform) const;

    private:
        ProjectionModel::ConstOwn m_src;
        ProjectionModel::ConstOwn m_dst;
        Distance m_dist;
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
    class PoseEstimation
    {
    public:
        virtual bool operator() (EuclideanTransform& pose) const = 0;
    };

    /**
     * Pose estimation by essential matrix decomposition.
     */
    class EssentialMatDecomposition : public PoseEstimation
    {
    public:
        EssentialMatDecomposition(const EpipolarObjective& objective) : m_objective(objective) {}
        virtual bool operator() (EuclideanTransform& pose) const;

    private:
        const EpipolarObjective& m_objective;
    };

    /**
     * Pose estimation by a perspective-n-points solver
     */
    class PerspevtivePoseEstimation : public PoseEstimation
    {
    public:
        PerspevtivePoseEstimation(const ProjectionObjective& objective) : m_objective(objective) {}
        virtual bool operator() (EuclideanTransform& pose) const;

    private:
        const ProjectionObjective& m_objective;
    };

    /**
     * Pose estimation by Horn's quaternion-based absolute orientation solving method
     */
    class QuatAbsOrientationSolver : public PoseEstimation
    {
    public:
        QuatAbsOrientationSolver(const RigidObjective& objective) : m_objective(objective) {}
        virtual bool operator() (EuclideanTransform& pose) const;

    private:
        const RigidObjective& m_objective;
    };

    /**
     * A generic multi-objective pose estimator.
     */
    class MultiObjectivePoseEstimation
    : public PoseEstimation,
      public LeastSquaresProblem
    {
    public:
        //
        // Constructor
        //
        MultiObjectivePoseEstimation() : LeastSquaresProblem(0, 6), m_tform(Rotation::EULER_ANGLES) {}

        //
        // Pose estimation
        //
        virtual bool operator() (EuclideanTransform& pose) const;

        //
        // Accessors
        //
        inline void AddObjective(AlignmentObjective::Own& objective) { m_objectives.push_back(objective); }
        //inline EuclideanTransform GetTransform() const { return m_transform; }
        size_t GetConds() const;
        EuclideanTransform GetSolution() const { return m_tform; }

        //
        // Least-squares problem
        //
        virtual VectorisableD::Vec Initialise();
        virtual VectorisableD::Vec operator()(const VectorisableD::Vec& x) const;
        virtual bool SetSolution(const VectorisableD::Vec& x) { return m_tform.Restore(x); }

    protected:
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

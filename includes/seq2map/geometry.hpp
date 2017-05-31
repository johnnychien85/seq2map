#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <seq2map/common.hpp>
#include <seq2map/solve.hpp>

namespace seq2map
{
    cv::Mat eucl2homo(const cv::Mat& x, bool rowMajor = true);
    cv::Mat homo2eucl(const cv::Mat& x, bool rowMajor = true);
    cv::Mat skewsymat(const cv::Mat& x);

    class EuclideanTransform : public Persistent<cv::FileStorage, cv::FileNode>, public VectorisableD
    {
    public:
        EuclideanTransform() :
            m_matrix(cv::Mat::eye(3, 4, CV_64F)),
            m_rvec(cv::Mat::zeros(3, 1, CV_64F)),
            m_rmat(m_matrix.rowRange(0, 3).colRange(0, 3)),
            m_tvec(m_matrix.rowRange(0, 3).colRange(3, 4)) {}
        EuclideanTransform(const cv::Mat& rotation, const cv::Mat& tvec);
        inline EuclideanTransform operator<<(const EuclideanTransform& tform) const { return EuclideanTransform(tform.m_rmat * m_rmat, tform.GetRotationMatrix() * m_tvec + tform.m_tvec); }
        inline EuclideanTransform operator>>(const EuclideanTransform& tform) const { return tform << *this;                }
        inline EuclideanTransform operator- (const EuclideanTransform& tform) const { return tform.GetInverse() >> (*this); }
        //void Apply(Point3F& pt) const;
        //void Apply(Points3F& pts) const;
        void Apply(Point3D& pt) const;
        void Apply(Points3D& pts) const;
        bool SetRotationMatrix(const cv::Mat& rmat);
        bool SetRotationVector(const cv::Mat& rvec);
        bool SetTranslation(const cv::Mat& tvec);
        bool SetTransformMatrix(const cv::Mat& matrix);
        inline cv::Mat GetRotationMatrix() const { return m_rmat; }
        inline cv::Mat GetRotationVector() const { return m_rvec; }
        inline cv::Mat GetTranslation() const { return m_tvec; }
        cv::Mat GetTransformMatrix(bool sqrMat = false, bool preMult = true) const;
        cv::Mat GetEssentialMatrix() const;
        EuclideanTransform GetInverse() const;
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);
        virtual Vec ToVector() const;
        virtual bool FromVector(const Vec& v);
        virtual size_t GetDimension() const { return 6; }

        static const EuclideanTransform Identity;
    protected:
        cv::Mat m_matrix;
        cv::Mat m_rmat;
        cv::Mat m_rvec;
        cv::Mat m_tvec;
    };

    typedef std::vector<EuclideanTransform> EuclideanTransforms;

    class Motion : public Persistent<Path>
    {
    public:
        Motion() {}
        virtual ~Motion() {}
        size_t Update(const EuclideanTransform& tform);
        inline EuclideanTransform GetGlobalTransform(size_t to) const { return m_tforms[to]; }
        inline EuclideanTransform GetLocalTransform(size_t from, size_t to) const { return m_tforms[from].GetInverse() >> m_tforms[to]; }
        inline bool IsEmpty() const { return m_tforms.empty(); }
        inline size_t GetSize() const { return m_tforms.size(); }
        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path);
    protected:
        EuclideanTransforms m_tforms;
    };

    class ProjectionModel : public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
        typedef boost::shared_ptr<ProjectionModel> Ptr;
        typedef boost::shared_ptr<const ProjectionModel> ConstPtr;

        virtual void Project(const Points3D& pts3d, Points2D& pts2d) const = 0;
        virtual bool Store(cv::FileStorage& fs) const = 0;
        virtual bool Restore(const cv::FileNode& fn) = 0;
        virtual String GetModelName() const = 0;
    };

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

    struct AlignmentCriteria
    {
        enum Metric
        {
            EPIPOLAR_DIRECT,
            EPIPOLAR_NORMALISED,
            EPIPOLAR_SAMPSON,
            PROJECTIVE,
            RIGID,
            PHOTOMETRIC
        };

        inline bool IsEnabled() const { return threshold > 0; }

        Metric metric;
        double threshold;
    };

    template<typename T, size_t dim> struct Weighted
    {
        Weighted(const T& pt = T(), const cv::Mat& cov = cv::Mat::eye(dim, dim, CV_64F))
        : pt(pt), cov(cov) {}

        T pt;
        cv::Mat cov;
    };

    typedef Weighted<Point2D, 2> PointW2D;
    typedef Weighted<Point3D, 3> PointW3D;

    typedef Mapping<Point2D, Point2D> PointMap2Dto2D;
    typedef Mapping<Point3D, Point2D> PointMap3Dto2D;

    typedef Mapping<PointW2D, PointW2D> WeightedPointMap2Dto2D;
    typedef Mapping<PointW3D, PointW2D> WeightedPointMap3Dto2D;

    class MotionEstimation
    : public LeastSquaresProblem,
      public Persistent<Path>
    {
    public:
        MotionEstimation(const cv::Mat& K, double alpha = 0.0f, bool sepRpe = false, bool sampsonError = true) :
            m_alpha(alpha), m_separatedRpe(sepRpe), m_sampsonError(sampsonError),
            LeastSquaresProblem(0, 6) { K.convertTo(m_cameraMatrix, CV_64F); }
        inline size_t AddObservation(int uid, const Point2D& pts2di, const Point2D& pts2dj) { m_uepi.push_back(uid); return m_epi.Add(pts2di, pts2dj); }
        inline size_t AddObservation(int uid, const Point3D& pts3di, const Point2D& pts2dj, double w) { m_upnp.push_back(uid); m_wpnp.push_back(w); return m_pnp.Add(pts3di, pts2dj); }
        inline PointMap2Dto2D GetEpipolarConds() const { return m_epi; }
        inline PointMap3Dto2D GetReprojectionConds() const { return m_pnp; }
        virtual VectorisableD::Vec Initialise();
        virtual VectorisableD::Vec Evaluate(const VectorisableD::Vec& x0) const;
        virtual bool SetSolution(const VectorisableD::Vec& x) { return m_transform.FromVector(x); }
        size_t GetEvaluationSize() const;
        inline EuclideanTransform GetTransform() const { return m_transform; }
        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path) { return false; }
    protected:
        cv::Mat EvalEpipolarConds(const EuclideanTransform& transform) const;
        cv::Mat EvalReprojectionConds(const EuclideanTransform& transform) const;
        double m_alpha;
        bool m_separatedRpe;
        bool m_sampsonError;
        cv::Mat m_cameraMatrix;
        cv::Mat m_invCameraMatrix;
        PointMap2Dto2D m_epi;
        PointMap3Dto2D m_pnp;
        std::vector<double> m_wpnp;
        std::vector<int> m_uepi;
        std::vector<int> m_upnp;
        EuclideanTransform m_transform;
    };

    class PointTriangulator
    {
    public:
        PointTriangulator(const cv::Mat& P0, const cv::Mat& P1) : m_projMatrix0(P0), m_projMatrix1(P1) {}
        virtual ~PointTriangulator() {}
        virtual void Triangulate(const PointMap2Dto2D& map, Points3D& x3d, std::vector<double>& err) = 0;
    protected:
        cv::Mat m_projMatrix0;
        cv::Mat m_projMatrix1;
    };

    class OptimalTriangulator : public PointTriangulator
    {
    public:
        OptimalTriangulator(const cv::Mat& P0, const cv::Mat& P1) : PointTriangulator(P0, P1) {}
        virtual void Triangulate(const PointMap2Dto2D& map, Points3D& x3d, std::vector<double>& err);
    };

    class MidPointTriangulator : public PointTriangulator
    {
    public:
        MidPointTriangulator(const cv::Mat& P0, const cv::Mat& P1) : PointTriangulator(P0, P1) {}
        virtual void Triangulate(const PointMap2Dto2D& map, Points3D& x3d, std::vector<double>& err);
    protected:
        static void DecomposeProjMatrix(const cv::Mat& P, cv::Mat& KRinv, cv::Mat& c);
    };

    class BouguetModel
    : public ProjectionModel,
      public VectorisableD
    {
    public:
        typedef boost::shared_ptr<BouguetModel> Ptr;

        BouguetModel(const cv::Mat& cameraMatrix = s_canonical.m_cameraMatrix, const cv::Mat& distCoeffs = s_canonical.m_distCoeffs);
        virtual void Project(const Points3D& pts3d, Points2D& pts2d) const;
        bool SetCameraMatrix(const cv::Mat& cameraMatrix);
        bool SetDistCoeffs(const cv::Mat& distCoeffs);
        void SetValues(double fu, double fv, double uc, double vc, double k1, double k2, double p1, double p2, double k3);
        inline cv::Mat GetCameraMatrix() const { return m_cameraMatrix.clone(); }
        inline cv::Mat GetDistCoeffs() const { return m_distCoeffs; }
        void GetValues(double& fu, double& fv, double& uc, double& vc, double& k1, double& k2, double& p1, double& p2, double& k3) const;
        cv::Mat MakeProjectionMatrix(const EuclideanTransform& pose) const;
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);
        virtual String GetModelName() const { return "BOUGUET"; }
        virtual Vec ToVector() const;
        virtual bool FromVector(const Vec& v);
        virtual size_t GetDimension() const { return 9; }
    protected:
        static const BouguetModel s_canonical;
        cv::Mat m_cameraMatrix;
        cv::Mat m_distCoeffs;
    };
}
#endif // GEOMETRY_HPP

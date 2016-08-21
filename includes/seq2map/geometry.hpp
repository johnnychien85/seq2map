#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <seq2map\common.hpp>
#include <seq2map\solve.hpp>

namespace seq2map
{
    class EuclideanTransform : public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
        EuclideanTransform() :
            m_matrix(cv::Mat::eye(3, 4, CV_32F)),
            m_rvec(cv::Mat::zeros(3, 1, CV_32F)),
            m_rmat(m_matrix.rowRange(0, 3).colRange(0, 3)),
            m_tvec(m_matrix.rowRange(0, 3).colRange(3, 4)) {}
        EuclideanTransform(const cv::Mat& rotation, const cv::Mat& tvec);
        inline EuclideanTransform operator<<(const EuclideanTransform& tform) const;
        EuclideanTransform operator >> (const EuclideanTransform& tform) const;
        void Apply(Point3F& pt) const;
        void Apply(Points3F& pts) const;
        bool SetRotationMatrix(const cv::Mat& rmat);
        bool SetRotationVector(const cv::Mat& rvec);
        bool SetTranslation(const cv::Mat& tvec);
        bool SetTransformMatrix(const cv::Mat& matrix);
        inline cv::Mat GetRotationMatrix() const { return m_rmat; }
        inline cv::Mat GetRotationVector() const { return m_rvec; }
        inline cv::Mat GetTranslation() const { return m_tvec; }
        cv::Mat GetTransformMatrix(bool sqrMat = false, bool preMult = true) const;
        EuclideanTransform GetInverse() const;
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        static const EuclideanTransform Identity;
    protected:
        cv::Mat m_matrix;
        cv::Mat m_rmat;
        cv::Mat m_rvec;
        cv::Mat m_tvec;
    };

    class Motion
    {
    public:
        Motion() {}
        virtual ~Motion() {}
        size_t Update(const EuclideanTransform& tform);
        EuclideanTransform GetGlobalTransform(size_t to) const;
        EuclideanTransform GetLocalTransform(size_t from, size_t to) const;
    protected:
        std::vector<EuclideanTransform> m_tforms;
    };

    class ProjectionModel : public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
        typedef boost::shared_ptr<ProjectionModel> Ptr;
        virtual void Project(const Points3F& pts3d, Points2F& pts2d) const = 0;
        virtual bool Store(cv::FileStorage& fs) const = 0;
        virtual bool Restore(const cv::FileNode& fn) = 0;
        virtual String GetModelName() const = 0;
    };

    template<class T0, class T1> class PointMap
    {
    public:
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

    typedef PointMap<Point2F, Point2F> PointMap2Dto2D;
    typedef PointMap<Point3F, Point2F> PointMap3Dto2D;

    class MotionEstimation : public LeastSquaresProblem
    {
    public:
        MotionEstimation(const cv::Mat& K) : m_cameraMatrix(K.clone()), LeastSquaresProblem(0, 6) {}
        inline size_t AddObservation(const Point2F& pts2di, const Point2F& pts2dj) { return m_epi.Add(pts2di, pts2dj); }
        inline size_t AddObservation(const Point3F& pts3di, const Point2F& pts2dj) { return m_pnp.Add(pts3di, pts2dj); }
        inline PointMap2Dto2D GetEpipolarConds() const { return m_epi; }
        inline PointMap3Dto2D GetPerspectiveConds() const { return m_pnp; }
        virtual cv::Mat Initialise();
        virtual cv::Mat Evaluate(const cv::Mat& x0) const;
        virtual bool SetSolution(const cv::Mat& x);
        EuclideanTransform GetTransform() const { return m_transform; }
    protected:
        cv::Mat m_cameraMatrix;
        PointMap2Dto2D m_epi;
        PointMap3Dto2D m_pnp;
        EuclideanTransform m_transform;
    };

    class PointTriangulator
    {
    public:
        PointTriangulator(const cv::Mat& P0, const cv::Mat& P1) : m_projMatrix0(P0), m_projMatrix1(P1) {}
        virtual ~PointTriangulator() {}
        virtual void Triangulate(const PointMap2Dto2D& map, Points3F& x3d) = 0;
    protected:
        cv::Mat m_projMatrix0;
        cv::Mat m_projMatrix1;
    };

    class OptimalTriangulator : public PointTriangulator
    {
    public:
        OptimalTriangulator(const cv::Mat& P0, const cv::Mat& P1) : PointTriangulator(P0, P1) {}
        virtual void Triangulate(const PointMap2Dto2D& map, Points3F& x3d);
    };

    class MidPointTriangulator : public PointTriangulator
    {
    public:
        MidPointTriangulator(const cv::Mat& P0, const cv::Mat& P1) : PointTriangulator(P0, P1) {}
        virtual void Triangulate(const PointMap2Dto2D& map, Points3F& x3d);
    };

    class BouguetModel : public ProjectionModel
    {
    public:
        typedef boost::shared_ptr<BouguetModel> Ptr;

        BouguetModel(const cv::Mat& cameraMatrix = s_canonical.m_cameraMatrix, const cv::Mat& distCoeffs = s_canonical.m_distCoeffs);
        virtual void Project(const Points3F& pts3d, Points2F& pts2d) const;
        bool SetCameraMatrix(const cv::Mat& cameraMatrix);
        bool SetDistCoeffs(const cv::Mat& distCoeffs);
        inline cv::Mat GetCameraMatrix() const { return m_cameraMatrix.clone(); }
        cv::Mat MakeProjectionMatrix(const EuclideanTransform& pose) const;
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);
        virtual String GetModelName() const { return "BOUGUET";  }
    protected:
        static const BouguetModel s_canonical;
        cv::Mat m_cameraMatrix;
        cv::Mat m_distCoeffs;
    };
}
#endif // GEOMETRY_HPP

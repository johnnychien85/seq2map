#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <seq2map/common.hpp>
#include <seq2map/solve.hpp>

namespace seq2map
{
    cv::Mat skewsymat(const cv::Mat& x);

    /**
     * A geometry matrix wraps a cv::Mat that stores coordinates of n-dimensional elements.
     */
    class Geometry
    {
    public:
        enum Shape
        {
            ROW_MAJOR, // M elements in D-dimension space are arrnaged row-by-row, forming a M-by-D matrix.
            COL_MAJOR, // M elements in D-dimension space are arranged column-by-column, forming a D-by-M matrix.
            PACKED     // M-by-N elements in D-dimension space are arranged as a M-by-N-by-D matrix.
        };

        //
        // Constructor and destructor
        //
        
        /**
         * Explicit construction with a specified shape and optionally a geometry matrix.
         *
         * /param shape storage shape of elements in the geometry matrix.
         * /param geometry matrix.
         */
        Geometry(Shape shape, cv::Mat mat = cv::Mat()) : shape(shape), mat(mat) {}
        
        /**
         * Copy constructor with shared geometry matrix.
         */
        Geometry(Geometry& g) : shape(g.shape), mat(g.mat) {}

        /**
         * Copy constructor with a cloned geometry matrix.
         */
        Geometry(const Geometry& g) : shape(g.shape), mat(g.mat.clone()) {}

        /**
         * Destructor
         */
        virtual ~Geometry() {}

        //
        // Creation and conversion
        //
        static Geometry MakeHomogeneous(const Geometry& g, double w = 1.0f);
        static Geometry FromHomogeneous(const Geometry& g);

        /**
         * Consistency check.
         */
        bool IsConsistent(const Geometry& g) const;

        /**
         * Shallow copy assignment if possible, cloning will be avoided if no transpose is needed
         */
        Geometry& operator= (Geometry& g);

        /**
         * Deep copy assignment, always do cloning
         */
        Geometry& operator= (const Geometry& g);

        /**
         * Calculate per-element subtraction
         */
        Geometry operator- (const Geometry& g) const;

        /**
         * Rescale the geometry in-place to make the last dimension equal to one.
         * Note this method does not reduce the dimensionality by one.
         */
        Geometry& Dehomogenise();

        /**
         * Change the ordering of elements.
         */
        Geometry Reshape(Shape shape) const { return Geometry(shape) = *this; }
        Geometry Reshape(Shape shape)       { return Geometry(shape) = *this; }

        Geometry Reshape(const Geometry& g) const;
        Geometry Reshape(const Geometry& g);

        //
        // Accessors
        //
        size_t GetElements()  const;
        size_t GetDimension() const;

        cv::Mat mat;
        const Shape shape;

    private:
        /**
         * Reshape a cv::mat to fit target topology, internally used only.
         *
         * \param src Source matrix shape
         * \param dst Target matrix shape
         * \param rows Number of rows used when converting from Shape::ROW_MAJOR or Shape::COL_MAJOR to Shape::PACKED 
         * \return True if mat is re-allocated, false otherwise.
         */
        static bool Reshape(Shape src, Shape dst, cv::Mat& mat, size_t rows = 0);
    };

    /**
     * A metric measures distances between two geometry data.
     */
    class Metric
    : public Referenced<Metric> // by GeometricMapping
    {
    public:
        /**
         * Compute distance between two geometry data.
         *
         * \param first geometry data
         * \param second geometry data that have the same dimensionality of the first one.
         * \return The returned geometry has the same element of x and y and the dimension is always one.
         */
        virtual Geometry operator() (const Geometry& x, const Geometry& y) const = 0;

        /**
         * Compute norm of a geometry data (i.e. the distances to origin)
         *
         * \param x a geometry data.
         * \return Norm of x.
         */
        virtual Geometry operator() (const Geometry& x) const = 0;
    };

    /**
     * Euclidean distance.
     */
    class EuclideanMetric : public Metric
    {
    public:
        virtual Geometry operator() (const Geometry& x, const Geometry& y) const;
        virtual Geometry operator() (const Geometry& x) const;
    };

    /**
     * Generalised Euclidean distance
     */
    class MahalanobisMetric : public Metric
    {
    public:
        enum InverseCovarianceType
        {
            ISOTROPIC,              // the matrix is M-by-1
            ANISOTROPIC_ORTHOGONAL, // the matrix is M-by-D
            ANISOTROPIC_ROTATED     // the matrix is M-by-D*(D+1)/2
        };

        MahalanobisMetric() : icv(Geometry::ROW_MAJOR) {}

        virtual Geometry operator() (const Geometry& x, const Geometry& y) const;
        virtual Geometry operator() (const Geometry& x) const;

        Geometry icv; // inverse covariance coefficients for error modelling
    };

    /**
     * Mapping of geometry data in two different spaces.
     */
    struct GeometricMapping
    {
        GeometricMapping() : src(Geometry::ROW_MAJOR), dst(Geometry::ROW_MAJOR) {}

        bool IsConsistent() const { return src.GetElements() == dst.GetElements(); }

        Geometry src; // source geometry data
        Geometry dst; // target geometry data
        Metric::Own metric;
    };

    /**
     * A geometry transform can be applied to geometry data.
     */
    class GeometricTransform
    : public Persistent<cv::FileStorage, cv::FileNode>,
      public VectorisableD
    {
    public:
        /**
         * Transform a single precision point.
         */
        virtual Point3F& operator() (Point3F& pt) const = 0;

        /**
         * Transform a double precision point.
         */
        virtual Point3D& operator() (Point3D& pt) const = 0;

        /**
         * Transform a set of single precision points
         */
        virtual Points3F& operator() (Points3F& pts) const;

        /**
         * Transform a set of double precision points
         */
        virtual Points3D& operator() (Points3D& pts) const;

        /**
         * Transform a set of points in matrix form
         */
        virtual Geometry& operator() (Geometry& g) const = 0;
    };

    /**
     * An Euclidean transform actualises rigid transform in 3D Euclidean space.
     */
    class EuclideanTransform : public GeometricTransform
    {
    public:
        class Objective
        {
        public:
            virtual cv::Mat operator() (const EuclideanTransform& tform) const = 0;
        };

        //
        // Constructors
        //

        /**
         * Default constructor.
         */
        EuclideanTransform()
        : m_matrix(cv::Mat::eye(3, 4, CV_64F)),
          m_rvec(cv::Mat::zeros(3, 1, CV_64F)),
          m_rmat(m_matrix.rowRange(0, 3).colRange(0, 3)),
          m_tvec(m_matrix.rowRange(0, 3).colRange(3, 4)) {}

        /**
         * Construction from a rotation matrix and a translation vector
         */
        EuclideanTransform(const cv::Mat& rotation, const cv::Mat& tvec);

        /**
         * Destructor
         */
        virtual ~EuclideanTransform() {}

        //
        // Binary operators
        //

        /**
         * Append a transform to the current one and return the reulstant transform.
         */
        inline EuclideanTransform operator<<(const EuclideanTransform& tform) const { return EuclideanTransform(tform.m_rmat * m_rmat, tform.GetRotationMatrix() * m_tvec + tform.m_tvec); }

        /**
         * Prepend a transform to the current one and return the reulstant transform.
         */
        inline EuclideanTransform operator>>(const EuclideanTransform& tform) const { return tform << *this; }

        /**
         * Calculate the difference between the current transform and a target transform.
         */
        inline EuclideanTransform operator- (const EuclideanTransform& tform) const { return tform.GetInverse() >> (*this); }

        //
        // Accessors
        //

        /**
         * Set the rotation component given an SO(3) matrix.
         */
        bool SetRotationMatrix(const cv::Mat& rmat);

        /**
         * Set the rotation component given an rotation 3-vector.
         */
        bool SetRotationVector(const cv::Mat& rvec);

        /**
         * Set the translation component given a 3-vector.
         */
        bool SetTranslation(const cv::Mat& tvec);

        /**
         * Set the rotation and translation components given a transform matrix.
         */
        bool SetTransformMatrix(const cv::Mat& matrix);

        /**
         * Get the rotation as an SO(3) matrix.
         */
        inline cv::Mat GetRotationMatrix() const { return m_rmat; }

        /**
         * Get the rotation as a 3-vector.
         */
        inline cv::Mat GetRotationVector() const { return m_rvec; }

        /**
         * Get the translation.
         */
        inline cv::Mat GetTranslation() const { return m_tvec; }

        /**
         * Get the whole transform as a matrix.
         *
         * \param sqrMat set to true to return a 4-by-4 transform matrix to work with 3D homogeneous coordinates. Otherwise the homogeneous entries are omitted.
         * \param preMult set to true if the transform will be applied to a geometry matrix by pre-multiplication. Otherwise the matrix is transposed.
         * \param type type of returned cv::Mat
         */
        cv::Mat GetTransformMatrix(bool sqrMat = false, bool preMult = true, int type = CV_64F) const;

        //
        // Creation and conversion
        //

        /**
         * Get the corresponding essential matrix obtained by multplying rotation matrix by the skew matrix form of the translation vector.
         */
        cv::Mat ToEssentialMatrix() const;

        /**
         * Decompose an essential matrix and use the geometrically meaningful solution to derive the corresponding Euclidean transform.
         */
        bool FromEssentialMatrix(const cv::Mat& E, const GeometricMapping& m);

        /**
         * Compute the inverse transform.
         */
        EuclideanTransform GetInverse() const;

        //
        // Applying the transform
        //
        virtual Point3F& operator() (Point3F& pt) const;
        virtual Point3D& operator() (Point3D& pt) const;
        virtual Geometry& operator() (Geometry& g) const;

        using GeometricTransform::operator();

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        //
        // Vectorisation and de-vectorisation
        //
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

    /**
     * Motion represents a sequence of Euclidean transforms.
     */
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

    /**
     * A projection model maps points in 3D space to 2D image plane.
     */
    class ProjectionModel
    : public GeometricTransform,
      public Referenced<ProjectionModel>
    {
    public:
        enum ProjectiveSpace
        {
            EUCLIDEAN_2D,  // a 3D point (X,Y,Z) is mapped to an image point (x,y)
            EUCLIDEAN_3D,  // a 3D point (X,Y,Z) is rescaled to Z * (x,y,1)
            HOMOGENEOUS_3D // a 3D point (X,Y,Z) is mapped to an image point (x,y,1) with
        };

        //
        // Transforms
        //

        virtual Point3F& operator() (Point3F& pt) const = 0;
        virtual Point3D& operator() (Point3D& pt) const = 0;

        /**
         * Perform default forward projection
         */
        virtual Geometry& operator() (Geometry& g) const { return g = Project(g, HOMOGENEOUS_3D); }

        /**
         * Perform forward projection.
         *
         * \param g input geometry in 3D Euclidean space
         * \param space target space
         * \return geometry transformed either into image plane or to a re-scaled Euclidean space.
         */
        virtual Geometry Project(const Geometry& g, ProjectiveSpace space = EUCLIDEAN_2D) const = 0;

        /**
         * Perform backward projection.
         *
         * \param g input geometry in image plane
         * \return geometry representing directional vector 
         */
        virtual Geometry Backproject(const Geometry& g) const = 0;

        //
        // Accessors
        //
        virtual String GetModelName() const = 0;
    };

    /**
     * A delegation class modelling a projection process associated with a reference frame.
     */
    class PosedProjection : public ProjectionModel
    { 
    public:
        PosedProjection(const EuclideanTransform& pose, const ProjectionModel& proj) : pose(pose), proj(proj) {}

        virtual Point3F& operator() (Point3F& pt) const { return proj(pose(pt)); }
        virtual Point3D& operator() (Point3D& pt) const { return proj(pose(pt)); }

        virtual Geometry Project(const Geometry& g, ProjectiveSpace space = EUCLIDEAN_2D) const { return proj.Project(pose(Geometry(g)), space); }
        virtual Geometry Backproject(const Geometry& g) const { return pose.GetInverse()(Geometry::MakeHomogeneous(proj.Backproject(g), 0.0f)); }

        const EuclideanTransform& pose;
        const ProjectionModel& proj;

        virtual String GetModelName() const { return proj.GetModelName(); }

        virtual bool Store(cv::FileStorage& fs) const { return false; }
        virtual bool Restore(const cv::FileNode& fn) { return false; }

        //
        // Vectorisation and de-vectorisation
        //
        virtual Vec ToVector() const { return Vec(0); }
        virtual bool FromVector(const Vec&) { return false; }
        virtual size_t GetDimension() const { return 0; }
    };

    /**
     * Pinhole camera model describes the projection using a linear projective transform.
     */
    class PinholeModel : public ProjectionModel
    {
    public:
        //
        // Constructor and destructor
        //
        PinholeModel() : m_matrix(cv::Mat::eye(3, 3, CV_64F)) { SetValues(1.0f, 1.0f, 0.0f, 0.0f); }
        PinholeModel(const cv::Mat& K) : m_matrix(cv::Mat::eye(3, 3, CV_64F)) { SetCameraMatrix(K); }

        //
        // Creation and conversion
        //
        cv::Mat ToProjectionMatrix(const EuclideanTransform& tform) const;
        bool FromProjectionMatrix(const cv::Mat& P, const EuclideanTransform& tform);

        //
        // Accessors
        //
        bool SetCameraMatrix(const cv::Mat& cameraMatrix);
        inline cv::Mat GetCameraMatrix() const { return m_matrix.clone(); }
        cv::Mat GetInverseCameraMatrix() const;
        
        void SetValues(double fx, double fy, double cx, double cy);
        void GetValues(double& fx, double& fy, double& cx, double& cy) const;

        virtual String GetModelName() const { return "PINHOLE"; }

        //
        // Forward projection
        //
        virtual Point3F& operator() (Point3F& pt) const;
        virtual Point3D& operator() (Point3D& pt) const;

        virtual Geometry Project(const Geometry& g, ProjectiveSpace space = EUCLIDEAN_2D) const;

        //
        // Backward projection
        //
        virtual Geometry Backproject(const Geometry& g) const;

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        //
        // Vectorisation and de-vectorisation
        //
        virtual Vec ToVector() const;
        virtual bool FromVector(const Vec&);
        virtual size_t GetDimension() const { return 4; }

    protected:
        cv::Mat m_matrix;
    };

    /**
     * Bouguet model extending pinhole model takes nonlinear distortion into account.
     */
    class BouguetModel : public PinholeModel
    {
    public:
        /**
         *
         */
        enum DistortionModel
        {
            RADIAL_TANGENTIAL_DISTORTION = 4,  // 4-dof distortion including k1, k2, p1 and p2
            HIGH_RADIAL_DISTORTION       = 5,  // 5-dof distortion including k1, k2, p1, p2, and k3
            RATIONAL_RADIAL_DISTORTION   = 7,  // 7-dof distortion including k1, k2, p1, p2, k3, k4, k5 and k6
            THIN_PSISM_DISTORTION        = 12, // 12-dof distortion including k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3 and s4
            TILTED_SENSOR_DISTORTION     = 14  // 14-dof distortion including k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x and tau_y
        };

        //
        // Constructor and destructor
        //
        BouguetModel() : m_distCoeffs(14) { SetValues(1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); }
        BouguetModel(const cv::Mat& K, const cv::Mat& D);
        
        //
        // Accessors
        //
        bool SetDistortionCoeffs(const cv::Mat& D);
        cv::Mat GetDistortionCoeffs() const;

        inline void SetDistortionModel(DistortionModel model) { m_distModel = model; }
        inline DistortionModel GetDistortionModel() const { return m_distModel; }

        void SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2);
        void SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3);
        void SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6);
        void SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6, double s1, double s2, double s3, double s4);
        void SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6, double s1, double s2, double s3, double s4, double tx, double ty);

        void GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2) const;
        void GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3) const;
        void GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6) const;
        void GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6, double& s1, double& s2, double& s3, double& s4) const;
        void GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6, double& s1, double& s2, double& s3, double& s4, double& tx, double& ty) const;

        using PinholeModel::SetValues;
        using PinholeModel::GetValues;

        virtual String GetModelName() const { return "BOUGUET"; }

        //
        // Forward projection
        //
        virtual Point3F& operator() (Point3F& pt) const;
        virtual Point3D& operator() (Point3D& pt) const;
        virtual Geometry Project(const Geometry& g, ProjectiveSpace space = EUCLIDEAN_2D) const;

        using GeometricTransform::operator();

        //
        // Backward projection
        //
        virtual Geometry Backproject(const Geometry& g) const;

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        //
        // Vectorisation and de-vectorisation
        //
        virtual Vec ToVector() const;
        virtual bool FromVector(const Vec& v);
        virtual size_t GetDimension() const { return PinholeModel::GetDimension() + static_cast<size_t>(m_distModel); }

    protected:
        Vec m_distCoeffs;
        DistortionModel m_distModel;
    };
}
#endif // GEOMETRY_HPP

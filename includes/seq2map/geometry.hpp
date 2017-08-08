#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <seq2map/common.hpp>

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
            ROW_MAJOR, ///< M elements in D-dimension space are arrnaged row-by-row, forming a M-by-D matrix.
            COL_MAJOR, ///< M elements in D-dimension space are arranged column-by-column, forming a D-by-M matrix.
            PACKED     ///< M-by-N elements in D-dimension space are arranged as a M-by-N-by-D matrix.
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
         * Add an offset in a specific dimension; commonly used in numerical differentiation.
         */
        Geometry Step(size_t dim, double eps) const;

        /**
         * Rescale the geometry in-place to make the last dimension equal to one.
         * Note this method does not reduce the dimensionality by one.
         */
        Geometry& Dehomogenise();

        /**
         * Extract sub-elements.
         */
        Geometry operator[] (const Indices& indices) const;

        /**
         * Change the memory layout of elements.
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
        inline bool IsEmpty() const { return mat.empty(); }

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

    class EuclideanTransform;
    class MahalanobisMetric;

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
         * \return The returned geometry has the same number of elements as x and y, and the dimension is always one.
         */
        virtual Geometry operator() (const Geometry& x, const Geometry& y) const { return (*this)(x - y); }

        /**
         * Compute norm of a geometry data (i.e. the distances to origin)
         *
         * \param x a geometry data.
         * \return Norm of x.
         */
        virtual Geometry operator() (const Geometry& x) const = 0;

        /**
        * Make a deep copy of metric.
        * \return Pointer to the cloned metric.
        */
        virtual Metric::Own Clone() const = 0;

        /**
         * Extraction of sub-elements.
         *
         * \param indices list of element indices.
         * \return A new metric containing extracted sub-elements.
         */
        virtual Metric::Own operator[] (const Indices& indices) const = 0;

        /**
         * Combine two metrices.
         *
         * \param metric Metric to be combined.
         * \return A new metric obtained from two metrices.
         */
        virtual Metric::Own operator+ (const Metric& metric) const = 0;

        /**
         * Apply an Euclidean transform to the metric, optionally with the Jacobian of a second-level transform.
         * The Jacobian achieves generic transformation of the metric to another space by first-order linearisation.
         *
         * \param tform The transform to be applied.
         * \param jac Jacobian of a transformation function.
         * \return The transformed metric.
         */
        virtual Metric::Own Transform(const EuclideanTransform& tform, const Geometry& jac = Geometry(Geometry::ROW_MAJOR)) const = 0;

        /**
         * Simplify the metric to a degenerated form for faster evaluation.
         */
        virtual Metric::Own Reduce() const = 0;

        /**
         *
         */
        virtual boost::shared_ptr<MahalanobisMetric> ToMahalanobis(bool& native) = 0;

        /**
         *
         */
        virtual boost::shared_ptr<const MahalanobisMetric> ToMahalanobis() const = 0;

        /**
         *
         */
        virtual bool FromMahalanobis(const MahalanobisMetric& metric) = 0;
    };

    /**
     * Euclidean distance.
     */
    class EuclideanMetric : public Metric
    {
    public:
        EuclideanMetric(double scale = 1.0f) : scale(scale) {}

        virtual Geometry operator() (const Geometry& x) const;
        virtual Metric::Own Clone() const { return Metric::Own(new EuclideanMetric(scale)); }
        virtual Metric::Own operator[] (const Indices& indices) const { return Clone(); }
        virtual Metric::Own operator+ (const Metric& metric) const;
        virtual Metric::Own Transform(const EuclideanTransform& tform, const Geometry& jac = Geometry(Geometry::ROW_MAJOR)) const { return Clone(); }
        virtual Metric::Own Reduce() const { return Clone(); }
        virtual boost::shared_ptr<MahalanobisMetric> ToMahalanobis(bool& native) { return 0; }
        virtual boost::shared_ptr<const MahalanobisMetric> ToMahalanobis() const { return 0; }
        virtual bool FromMahalanobis(const MahalanobisMetric& metric) { return false; }

        double scale;
    };

    /**
     * Euclidean distance with per-element weightings.
     */
    class WeightedEuclideanMetric : public EuclideanMetric
    {
    public:
        WeightedEuclideanMetric(cv::Mat weights, double scale = 1.0f);

        virtual Geometry operator() (const Geometry& x) const;
        virtual Metric::Own Clone() const { return Metric::Own(new WeightedEuclideanMetric(m_weights.clone(), scale)); }
        virtual Metric::Own operator[] (const Indices& indices) const;
        virtual Metric::Own operator+ (const Metric& metric) const;

        virtual boost::shared_ptr<MahalanobisMetric> ToMahalanobis(bool& native);
        virtual boost::shared_ptr<const MahalanobisMetric> ToMahalanobis() const;
        virtual bool FromMahalanobis(const MahalanobisMetric& metric);

        const cv::Mat& GetWeights() const { return m_weights; }

        bool bayesian;

    private:
        cv::Mat m_weights;
    };

    /**
     * Generalised Euclidean distance
     */
    class MahalanobisMetric : public Metric
    {
    public:
        enum CovarianceType
        {
            ISOTROPIC,              ///< the matrix is M-by-1
            ANISOTROPIC_ORTHOGONAL, ///< the matrix is M-by-D
            ANISOTROPIC_ROTATED     ///< the matrix is M-by-D*(D+1)/2
        };

        MahalanobisMetric(CovarianceType type, size_t dims)
        : type(type), dims(dims), m_cov(Geometry::ROW_MAJOR), m_icv(Geometry::ROW_MAJOR) {}

        MahalanobisMetric(CovarianceType type, size_t dims, const cv::Mat& cov);

        virtual Geometry operator() (const Geometry& x) const;

        virtual Metric::Own Clone() const { return Metric::Own(new MahalanobisMetric(type, dims, m_cov.mat.clone())); }

        virtual Metric::Own operator[] (const Indices& indices) const;

        virtual Metric::Own operator+ (const Metric& metric) const;

        virtual Metric::Own Transform(const EuclideanTransform& tform, const Geometry& jac = Geometry(Geometry::ROW_MAJOR)) const;

        virtual Metric::Own Reduce() const;

        virtual boost::shared_ptr<MahalanobisMetric> ToMahalanobis(bool& native);

        virtual boost::shared_ptr<const MahalanobisMetric> ToMahalanobis() const;

        virtual bool FromMahalanobis(const MahalanobisMetric& metric);

        static boost::shared_ptr<MahalanobisMetric> Identity(size_t numel, size_t dims, int depth = CV_64F);

        /**
         * Update the current covariance by a new observation by summing up
         * two Gaussian distributions.
         *
         * \return Kalman gain.
         */
        cv::Mat Update(const MahalanobisMetric& metric);

        inline  size_t GetCovMatCols() const { return GetCovMatCols(type, dims); }
        cv::Mat GetInverseCovMat(const cv::Mat& cov) const;
        cv::Mat GetFullCovMat() const;
        cv::Mat GetFullCovMat(size_t index) const;
        bool SetCovarianceMat(const cv::Mat& cov);
        inline const Geometry& GetCovariance() const { return m_cov; };

        static size_t GetCovMatCols(CovarianceType type, size_t dims);

        const CovarianceType type; ///< shape of covariance matrix
        const size_t dims;         ///< dimensionality

    private:
        struct Metres
        {
            Speedometre tfm;
            Speedometre cov;
        };

        static double s_rcondThreshold;

        Geometry m_cov;         ///< error covariance coefficients
        mutable Geometry m_icv; ///< inverse covariance coefficients, automatically calculated from cov
        mutable Metres m_metres;
    };

    /**
     * A metric that composes of metrices in two different spaces, with transformations always applied to the first metric.
     */
    class DualMetric : public Metric
    {
    public:
        DualMetric(const Metric::ConstOwn& src, Metric::ConstOwn dst) : src(src), dst(dst) {}

        virtual Geometry operator() (const Geometry& x) const { return (*(*src + *dst))(x); }
        virtual Metric::Own Clone() const { return Metric::Own(new DualMetric(src->Clone(), dst->Clone())); }
        virtual Metric::Own operator[] (const Indices& indices) const { return Metric::Own(new DualMetric((*src)[indices], (*dst)[indices])); }
        virtual Metric::Own operator+  (const Metric& metric)   const { return Metric::Own(new DualMetric((*src) + metric, (*dst) + metric)); }

        virtual Metric::Own Transform(const EuclideanTransform& tform, const Geometry& jac) const { return Metric::Own(new DualMetric(src->Transform(tform, jac), dst)); }
        virtual Metric::Own Reduce() const { return (*src + *dst)->Reduce(); }

        virtual boost::shared_ptr<MahalanobisMetric> ToMahalanobis(bool& native) { return 0; }
        virtual boost::shared_ptr<const MahalanobisMetric> ToMahalanobis() const { return 0; }
        virtual bool FromMahalanobis(const MahalanobisMetric& metric) { return false; }

        const Metric::ConstOwn src;
        const Metric::ConstOwn dst;
    };

    /**
     * Mapping of geometry data in two different spaces.
     */
    struct GeometricMapping
    {
        /**
         * Incremental data builder
         */
         template<typename T0, typename T1> class Builder
         {
         public:
            /**
             * Constructor.
             */
            Builder() : m_size(0) {}

            //
            // Data construction
            //

            /**
             * Add a new correspondence with optional index of source.
             */
            size_t Add(const T0& pt0, const T1& pt1, size_t idx = INVALID_INDEX)
            {
                m_pts0.push_back(pt0);
                m_pts1.push_back(pt1);
                m_indices.push_back(idx);

                return m_size = m_pts0.size();
            }

            /**
             * Build a GeometricMapping from accumulated elements.
             */
            GeometricMapping Build() const
            {
                GeometricMapping mapping;

                mapping.src = Geometry(Geometry::PACKED, cv::Mat(m_pts0));
                mapping.dst = Geometry(Geometry::PACKED, cv::Mat(m_pts1));
                mapping.indices = m_indices;

                return mapping;
            }
            
            //
            // Accessors
            //
            inline size_t GetSize() const              { return m_size; }
            inline const std::vector<T0>& From() const { return m_pts0;  }
            inline const std::vector<T1>& To()   const { return m_pts1;  }

        private:
            size_t m_size;
            std::vector<size_t> m_indices;
            std::vector<T0> m_pts0;
            std::vector<T1> m_pts1;
        };

        typedef Builder<Point2D, Point2D> ImageToImageBuilder;
        typedef Builder<Point3D, Point2D> WorldToImageBuilder;
        typedef Builder<Point3D, Point3D> WorldToWorldBuilder;

        /**
         * Constructor
         */
        GeometricMapping(cv::Mat& srcData = cv::Mat(), cv::Mat& dstData = cv::Mat())
        : src(Geometry::ROW_MAJOR, srcData), dst(Geometry::ROW_MAJOR, dstData) {}

        /**
         * Copy constructor
         */
        GeometricMapping(GeometricMapping& mapping) : indices(mapping.indices), src(mapping.src), dst(mapping.dst), metric(mapping.metric) {}

        /**
         * Extract sub-elements.
         */
        GeometricMapping operator[] (const Indices& indices) const;

        /**
         * Check if the mapping is one-to-one.
         */
        bool IsConsistent() const;

        /**
         * Dimensionality check.
         *
         * \param d0 Dimension of source geometry.
         * \param d1 Dimension of target geometry.
         *
         * \return true is check passed, otherwise false.
         */
        bool Check(size_t d0, size_t d1) const;

        /**
         * Get number of correspondences in the mapping.
         */
        size_t GetSize() const { return IsConsistent() ? src.GetElements() : 0; }

        std::vector<size_t> indices; ///< optional reference
        Geometry src;                ///< source geometry data
        Geometry dst;                ///< target geometry data
        Metric::Own metric;          ///< associated metric to indicate the confidence of each correspondence
    };

    /**
     * A geometry transform can be applied to geometry data in 3-dimensional space.
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

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const = 0;
        virtual bool Restore(const cv::FileNode& fn) = 0;

        //
        // Vectorisation and de-vectorisation
        //
        virtual bool Store(Vec& v) const = 0;
        virtual bool Restore(const Vec& v) = 0;
        virtual size_t GetDimension() const = 0;
    };

    /**
     * A rotation transform rotate given points/vectors about its rotation axis.
     */
    class Rotation : public GeometricTransform
    {
    public:
        enum Parameterisation
        {
            RODRIGUES,
            EULER_ANGLES
        };

        //
        // Helpers
        //
        static inline double ToRadian(double deg) { return deg / 180.0f * CV_PI; }
        static inline double ToDegree(double rad) { return rad * 180.0f / CV_PI; }
        static cv::Mat RotX(double rad);
        static cv::Mat RotY(double rad);
        static cv::Mat RotZ(double rad);

        //
        // Constructors
        //
        Rotation(Parameterisation param, cv::Mat rmat);
        Rotation(Parameterisation param = RODRIGUES) : Rotation(param, cv::Mat::eye(3, 3, CV_64F)) {}

        //
        // Comparison
        //
        inline bool operator== (const Rotation& rhs) const { return cv::norm(m_rmat, rhs.m_rmat) < 1e-6; }
        inline bool operator!= (const Rotation& rhs) const { return !(*this == rhs); }

        inline bool IsIdentity() const { return *this == Identity; }

        //
        // Creation and conversion
        //
        bool FromMatrix(const cv::Mat& rmat);
        bool FromVector(const Vec& rvec) { return FromVector(cv::Mat(rvec)); }
        bool FromVector(const cv::Mat& rvec);
        bool FromAngles(double x, double y, double z);

        /**
         * Get the rotation as an SO(3) matrix.
         */
        inline cv::Mat ToMatrix() const { return m_rmat.clone(); }

        /**
         * Get the rotation as a 3-vector.
         */
        void ToVector(Vec& rvec) const;

        /**
         *
         */
        inline cv::Mat ToVector() const { return m_rvec.clone(); }

        /**
         * Get the rotation as three angles.
         */
        void ToAngles(double& x, double& y, double& z) const;

        //
        // Accessors
        //
        inline Parameterisation GetParameterisation() const { return m_param; }
        inline void SetParametersiation(Parameterisation param) { m_param = param; }

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
        virtual bool Store(Vec& v) const;
        virtual bool Restore(const Vec& v);
        virtual size_t GetDimension() const { return 3; }

        static const Rotation Identity;

    private:
        cv::Mat m_rmat; ///< the 3-by-3 rotation matrix managed by the class; might refer to an external variable
        cv::Mat m_rvec;
        Parameterisation m_param;
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
        EuclideanTransform(Rotation::Parameterisation rform = Rotation::EULER_ANGLES)
        : m_matrix(cv::Mat::eye(3, 4, CV_64F)),
          m_tvec(m_matrix.rowRange(0, 3).colRange(3, 4)),
          m_rotation(rform, m_matrix.rowRange(0, 3).colRange(0, 3)) {}

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
        EuclideanTransform operator<<(const EuclideanTransform& tform) const;

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
         * Set the translation component given a 3-vector.
         */
        bool SetTranslation(const cv::Mat& tvec);

        /**
         *
         */
        bool SetTranslation(const Vec& tvec);

        /**
         *
         */
        void SetTranslation(const cv::Vec3d& tvec) { cv::Mat(tvec, false).reshape(1, 3).copyTo(m_tvec); }

        /**
         * Set the rotation and translation components given a transform matrix.
         */
        bool SetTransformMatrix(const cv::Mat& matrix);

        /**
         *
         */
        inline Rotation& GetRotation() { return m_rotation; }

        /**
         *
         */
        inline const Rotation& GetRotation() const { return m_rotation; }

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

        /**
         *
         */
        bool IsIdentity() const { return m_rotation.IsIdentity() && cv::norm(m_tvec) == 0; }

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
        virtual Geometry& operator() (Geometry& g) const { return (*this)(g, g.GetDimension() == 3); }
        Geometry& operator() (Geometry& g, bool euclidean) const;

        using GeometricTransform::operator();

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        //
        // Vectorisation and de-vectorisation
        //
        virtual bool Store(Vec& v) const;
        virtual bool Restore(const Vec& v);
        virtual size_t GetDimension() const { return m_rotation.GetDimension() + 3; }

        static const EuclideanTransform Identity;

    protected:
        cv::Mat m_matrix;
        cv::Mat m_tvec;
        Rotation m_rotation;
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
            HOMOGENEOUS_3D // a 3D point (X,Y,Z) is mapped to an image point (x,y,1)
        };

        /**
         * Get a cloned projection model.
         */
        virtual ProjectionModel::Own Clone() const = 0;

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
         * Compute the Jacobians of 3D points and return them as a geometry in 6D space representing
         * dx/dX, dy/dX, dx/dY, dy/dY, dx/dZ and dy/dZ.
         *
         * \param g Point geometry in 3D space.
         *
         * \return Jacobian geometry in 6D space; the first, second, and third two dimensions represent
         *         partial derivatives with respect to X, Y and Z coordinates, respectively.
         */
        virtual Geometry GetJacobian(const Geometry& g) const;

        /**
         * Compute the Jacobians of 3D points given pre-calculated 2D projections.
         */
        virtual Geometry GetJacobian(const Geometry& g, const Geometry& proj) const;

        /**
         * Perform back-projection.
         *
         * \param g input geometry in image plane
         * \return geometry representing directional vector 
         */
        virtual Geometry Backproject(const Geometry& g) const = 0;

        /**
         * Project points into image plane and get the pixel values.
         *
         * \param g input geometry in 3D Euclidean space.
         * \param im image data being projected.
         * \return matrix of pixel values in the size of g with the number of channels of im.
         */
        virtual cv::Mat Project(const Geometry& g, const cv::Mat& im) const;

        //
        // Accessors
        //
        virtual String GetModelName() const = 0;

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const = 0;
        virtual bool Restore(const cv::FileNode& fn) = 0;

        //
        // Vectorisation and de-vectorisation
        //
        virtual bool Store(Vec& v) const = 0;
        virtual bool Restore(const Vec& v) = 0;
        virtual size_t GetDimension() const = 0;
    };

    /**
     * A delegation class modelling a projection process associated with a reference frame.
     */
    class PosedProjection
    : public ProjectionModel
    { 
    public:
        PosedProjection(const EuclideanTransform& pose, ProjectionModel::Own& proj) : pose(pose), proj(proj) {}

        virtual ProjectionModel::Own Clone() const { return ProjectionModel::Own(new PosedProjection(pose, proj->Clone())); }

        virtual Point3F& operator() (Point3F& pt) const { return (*proj)(pose(pt)); }
        virtual Point3D& operator() (Point3D& pt) const { return (*proj)(pose(pt)); }

        virtual Geometry Project(const Geometry& g, ProjectiveSpace space = EUCLIDEAN_2D) const { return proj->Project(pose(Geometry(g), true), space); }
        virtual Geometry Backproject(const Geometry& g) const { return pose.GetInverse().GetRotation()(proj->Backproject(g)); }

        virtual Geometry GetJacobian(const Geometry& g, const Geometry& proj) const { return this->proj->GetJacobian(pose(Geometry(g), true), proj); }

        virtual String GetModelName() const { return proj->GetModelName(); }

        virtual bool Store(cv::FileStorage& fs) const { return proj->Store(fs);   }
        virtual bool Restore(const cv::FileNode& fn)  { return proj->Restore(fn); }

        //
        // Vectorisation and de-vectorisation
        //
        virtual bool Store(Vec& v) const    { return proj->Store(v);   }
        virtual bool Restore(const Vec& v)  { return proj->Restore(v); }
        virtual size_t GetDimension() const { return proj->GetDimension(); }

        const EuclideanTransform& pose;
        ProjectionModel::Own proj;
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

        virtual ProjectionModel::Own Clone() const { return ProjectionModel::Own(new PinholeModel(m_matrix)); }

        //
        // Creation and conversion
        //
        cv::Mat ToProjectionMatrix(const EuclideanTransform& tform) const;
        bool FromProjectionMatrix(const cv::Mat& P, const EuclideanTransform& tform);

        //
        // Comparison
        //
        virtual bool operator== (const PinholeModel& rhs) const;
        virtual bool operator!= (const PinholeModel& rhs) const { return !(*this == rhs); }

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
        // Differentiation
        //
        virtual Geometry GetJacobian(const Geometry& g) const;
        virtual Geometry GetJacobian(const Geometry& g, const Geometry& proj) const { return GetJacobian(g); }

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        //
        // Vectorisation and de-vectorisation
        //
        virtual bool Store(Vec& v) const;
        virtual bool Restore(const Vec& v);
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
        
        virtual ProjectionModel::Own Clone() const { return ProjectionModel::Own(new BouguetModel(m_matrix, cv::Mat(m_distCoeffs, false))); }

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
        virtual bool Store(Vec& v) const;
        virtual bool Restore(const Vec& v);
        virtual size_t GetDimension() const { return PinholeModel::GetDimension() + static_cast<size_t>(m_distModel); }

    protected:
        Vec m_distCoeffs;
        DistortionModel m_distModel;
    };
}
#endif // GEOMETRY_HPP

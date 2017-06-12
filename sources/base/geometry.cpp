#include <seq2map/geometry.hpp>

using namespace seq2map;

const EuclideanTransform EuclideanTransform::Identity(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));

namespace seq2map
{
    cv::Mat skewsymat(const cv::Mat& x)
    {
        assert(x.total() == 3);

        cv::Mat x64f;
        x.convertTo(x64f, CV_64F);

        // normalisation to make sure ||x|| = 1
        x64f /= cv::norm(x64f);

        double* v = x64f.ptr<double>();

        cv::Mat y64f = (cv::Mat_<double>(3, 3) <<
             0.0f,  -v[2],  v[1],
             v[2],   0.0f, -v[0],
            -v[1],   v[0], 0.0f);

        cv::Mat y;
        y64f.convertTo(y, x.type());

        return y;
    }
}

//==[ Geometry ]==============================================================//

Geometry& Geometry::operator= (const Geometry& g)
{
    if (!Reshape(g.shape, shape, mat = g.mat))
    {
        // make a deep copy if mat remains a shallow copy of g.mat
        mat = mat.clone();
    }

    return *this;
}

Geometry& Geometry::operator= (Geometry& g)
{
    Reshape(g.shape, shape, mat = g.mat);
    return *this;
}

Geometry Geometry::operator- (const Geometry& g) const
{
    if (!IsConsistent(g)) return Geometry(shape);

    Geometry d(shape);

    if (shape == g.shape)
    {
        cv::subtract(mat, g.mat, d.mat);
    }
    else
    {
        cv::subtract(mat, g.Reshape(shape).mat, d.mat);
    }

    return d;
}

bool Geometry::IsConsistent(const Geometry& g) const
{
    if (GetDimension() != g.GetDimension() || GetElements() != g.GetElements())
    {
        E_WARNING << "inconsistent dimensionality";
        return false;
    }

    return true;
}

size_t Geometry::GetElements() const
{
    switch (shape)
    {
    case ROW_MAJOR: return mat.rows; break;
    case COL_MAJOR: return mat.cols; break;
    case PACKED:    return mat.rows * mat.cols; break;
    default:        return 0; // invalid shape
    }
}

size_t Geometry::GetDimension() const
{
    switch (shape)
    {
    case ROW_MAJOR: return mat.cols; break;
    case COL_MAJOR: return mat.rows; break;
    case PACKED:    return mat.channels(); break;
    default:        return 0; // invalid shape
    }
}

Geometry Geometry::Reshape(const Geometry& g) const
{
    if (!IsConsistent(g) || g.shape != Geometry::PACKED)
    {
        return Reshape(g.shape);
    }

    cv::Mat m = mat;
    
    if (!Reshape(shape, g.shape, m, g.mat.rows))
    {
        m = m.clone();
    }

    return Geometry(g.shape, m);
}

Geometry Geometry::Reshape(const Geometry& g)
{
    if (!IsConsistent(g) || g.shape != Geometry::PACKED)
    {
        return Reshape(g.shape);
    }

    cv::Mat m = mat;
    Reshape(shape, g.shape, m, g.mat.rows);

    return Geometry(g.shape, m);
}

bool Geometry::Reshape(Shape src, Shape dst, cv::Mat& mat, size_t rows)
{
    if (src == dst) // same shape, nothing to be done
    {
        return false;
    }

    if (src != PACKED && dst != PACKED) // simple transpose
    {
        mat = mat.t(); // element cloning
        return true;
    }

    switch (src)
    {
    case ROW_MAJOR:
        mat = mat.reshape(mat.cols, static_cast<int>(rows));
        return false;

    case COL_MAJOR:
        mat = cv::Mat(mat.t()).reshape(mat.rows, static_cast<int>(rows));
        return true;
    }

    switch (dst)
    {
    case ROW_MAJOR:
        mat = mat.reshape(1);
        return false;

    case COL_MAJOR:
        mat = mat.reshape(1).t();
        return true;
    }

    assert(0); // should never reach this point!
    return false;
}

Geometry Geometry::MakeHomogeneous(const Geometry& g, double w)
{
    const cv::Mat& src = g.mat;
    cv::Mat dst;

    switch (g.shape)
    {
    case ROW_MAJOR:
        dst = cv::Mat(src.rows, src.cols + 1, src.type());
        src.copyTo(dst.colRange(0, src.cols));
        dst.col(src.cols).setTo(w);
        break;

    case COL_MAJOR:
        dst = cv::Mat(src.rows + 1, src.cols, src.type());
        src.copyTo(dst.rowRange(0, src.rows));
        dst.row(src.rows).setTo(w);
        break;

    case PACKED:
        //cv::convertPointsToHomogeneous(src, dst);
        dst = cv::Mat(src.rows, src.cols, CV_MAKETYPE(src.depth(), src.channels() + 1));
        src.reshape(1).copyTo(dst.reshape(1).colRange(0, src.channels()));
        dst.reshape(1).col(src.channels()).setTo(w);
        break;
    }

    return Geometry(g.shape, dst);
}

Geometry Geometry::FromHomogeneous(const Geometry& g)
{
    const cv::Mat& src = g.mat;
    cv::Mat dst;

    switch (g.shape)
    {
    case ROW_MAJOR:
        dst = cv::Mat(src.rows, src.cols - 1, src.type());
        for (int j = 0; j < dst.cols; j++)
        {
            dst.col(j) = src.col(j) / src.col(dst.cols);
        }
        break;

    case COL_MAJOR:
        dst = cv::Mat(src.rows - 1, src.cols, src.type());
        for (int i = 0; i < dst.cols; i++)
        {
            dst.row(i) = src.row(i) / src.row(dst.rows);
        }
        break;

    case PACKED:
        //cv::convertPointsFromHomogeneous(src, dst);
        dst = cv::Mat(src.rows, src.cols, CV_MAKETYPE(src.depth(), src.channels() - 1));
        dst = dst.reshape(1); // to row major order
        for (int j = 0; j < dst.cols; j++)
        {
            dst.col(j) = src.col(j) / src.col(dst.cols);
        }
        dst = dst.reshape(src.channels() - 1, src.rows);
        break;
    }

    return Geometry(g.shape, dst);
}

Geometry& Geometry::Dehomogenise()
{
    if (shape == PACKED)
    {
        this->Reshape(ROW_MAJOR).Dehomogenise(); // operate on a shallow copy of myself
        return *this;
    }

    const int n = static_cast<int>(GetDimension()) - 1;

    switch (shape)
    {
    case ROW_MAJOR:
        for (int j = 0; j < mat.cols; j++)
        {
            mat.col(j) = mat.col(j) / mat.col(n);
        }
        break;

    case COL_MAJOR:
        for (int i = 0; i < mat.rows; i++)
        {
            mat.row(i) = mat.row(i) / mat.row(n);
        }
        break;
    }

    return *this;
}

//==[ Metric ]================================================================//

//==[ EuclideanMetric ]=======================================================//

Geometry EuclideanMetric::operator() (const Geometry& x, const Geometry& y) const
{
    return (*this)(x - y);
}

Geometry EuclideanMetric::operator() (const Geometry& x) const
{
    if (x.GetDimension() == 1)
    {
        return Geometry(x.shape, cv::abs(x.mat));
    }

    if (x.shape != Geometry::ROW_MAJOR)
    {
        return (*this)(x.Reshape(Geometry::ROW_MAJOR)).Reshape(x.shape);
    }

    cv::Mat d;

    // work only for row-major shape
    for (int j = 0; j < x.mat.cols; j++)
    {
        cv::Mat dj;
        
        cv::multiply(x.mat.col(j), x.mat.col(j), dj); // dj = dj .^ 2
        cv::add(d, dj, d);                            // d  = d + dj
    }

    cv::sqrt(d, d); // d = sqrt(d)

    return Geometry(x.shape, d);
}

//==[ MahalanobisMetric ]=====================================================//

Geometry MahalanobisMetric::operator() (const Geometry& x, const Geometry& y) const
{
    return (*this)(x - y);
}

Geometry MahalanobisMetric::operator() (const Geometry& x) const
{
    if (x.shape != Geometry::ROW_MAJOR)
    {
        return (*this)(x.Reshape(Geometry::ROW_MAJOR)).Reshape(x.shape);
    }

    /*
    const size_t d0 = err.rows;
    const size_t d1 = icv.GetDimension();

    const size_t n0 = err.cols;
    const size_t n1 = icv.GetElements();

    if (n1 == 0)
    {
        return err;
    }

    bool perElementEval = n1 != 1;

    cv::Mat dist = cv::Mat::zeros(1, n0, err.type());
    cv::Mat isgm = icv.mat;

    isgm.convertTo(isgm, CV_64F);

    if (!perElementEval)
    {
        if (d1 == 1)
        {
            for (size_t i = 0; i < d0; i++)
            {
                cv::Mat x = err.mat.row(i);
                dist += x.mul(x);
            }

            dist = dist * isgm.at<double>
        }
        else if (d1 == d0)
        {

        }
    }
    else
    {
        if (n0 != n1)
        {
            E_ERROR << "number of icv elements (n=" << n1 << ") does not match the length of error vector (n=" << n0 << ")";
            return err;
        }

        if (d1 == 1)
        {

        }
    }

    return dist;
    */

    throw std::exception("not implemented");
    return Geometry(x.shape);
}

//==[ GeometricTransform ]====================================================//

Points3F& GeometricTransform::operator() (Points3F& pts) const
{
    (*this)(Geometry(Geometry::PACKED, cv::Mat(pts))).mat.copyTo(cv::Mat(pts));
    return pts;
}

Points3D& GeometricTransform::operator() (Points3D& pts) const
{
    (*this)(Geometry(Geometry::PACKED, cv::Mat(pts))).mat.copyTo(cv::Mat(pts));
    return pts;
}

//==[ EuclideanTransform ]====================================================//

EuclideanTransform::EuclideanTransform(const cv::Mat& rotation, const cv::Mat& tvec) : EuclideanTransform()
{
    if (rotation.total() == 3)
    {
        SetRotationVector(rotation);
    }
    else
    {
        SetRotationMatrix(rotation);
    }

    SetTranslation(tvec);
}

bool EuclideanTransform::SetRotationMatrix(const cv::Mat& rmat)
{
    if (rmat.rows != 3 || rmat.cols != 3)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(rmat.size()), " rather than 3x3";
        return false;
    }

    cv::Mat rvec;

    cv::Rodrigues(rmat, rvec);
    SetRotationVector(rvec);

    return true;
}

bool EuclideanTransform::SetRotationVector(const cv::Mat& rvec)
{
    if (rvec.total() != 3)
    {
        E_ERROR << "given vector has " << rvec.total() << " element(s) rather than 3";
        return false;
    }

    cv::Mat _rvec, rmat;
    rvec.convertTo(_rvec, m_rvec.type());

    cv::Rodrigues(m_rvec = _rvec, rmat);
    rmat.copyTo(m_rmat);

    return true;
}

bool EuclideanTransform::SetTranslation(const cv::Mat& tvec)
{
    if (tvec.total() != 3)
    {
        E_ERROR << "given vector has " << tvec.total() << " element(s) rather than 3";
        return false;
    }

    cv::Mat _tvec;
    tvec.convertTo(_tvec, m_tvec.type());
    _tvec.reshape(0, 3).copyTo(m_tvec);

    return true;
}

bool EuclideanTransform::SetTransformMatrix(const cv::Mat& matrix)
{
    if ((matrix.rows != 3 && matrix.rows != 4) || matrix.cols != 4)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(matrix.size()) << " rather than 3x4 or 4x4";
    }

    // TODO: check the fourth row if a square matrix is passed
    // ...
    // ...

    const cv::Mat rmat = matrix.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tvec = matrix.rowRange(0, 3).colRange(3, 4);

    return SetRotationMatrix(rmat) && SetTranslation(tvec);
}

cv::Mat EuclideanTransform::GetTransformMatrix(bool sqrMat, bool preMult, int type) const
{
    if (!sqrMat && preMult) return m_matrix.clone(); // this is the intrinsic form

    cv::Mat matrix = cv::Mat::eye(sqrMat ? 4 : 3, 4, m_matrix.type());
    m_matrix.copyTo(matrix.rowRange(0, 3).colRange(0, 4));

    if (matrix.type() != type)
    {
        matrix.convertTo(matrix, type);
    }

    return preMult ? matrix : matrix.t();
}

cv::Mat EuclideanTransform::ToEssentialMatrix() const
{
    return skewsymat(m_tvec) * m_rmat;
}

bool EuclideanTransform::FromEssentialMatrix(const cv::Mat& E, const GeometricMapping& m)
{
    // TODO: finish this
    // ...
    // ..
    // .

    throw std::exception("not implemented");

    return false;
}

EuclideanTransform EuclideanTransform::GetInverse() const
{
    cv::Mat Rt = m_rmat.t();
    return EuclideanTransform(Rt, -Rt*m_tvec);
}

Point3F& EuclideanTransform::operator() (Point3F& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3F(
        (float)(m[0] * pt.x + m[3] * pt.y + m[6] * pt.z + m[9]),
        (float)(m[1] * pt.x + m[4] * pt.y + m[7] * pt.z + m[10]),
        (float)(m[2] * pt.x + m[5] * pt.y + m[8] * pt.z + m[11])
    );

    return pt;
}

Point3D& EuclideanTransform::operator() (Point3D& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3D(
        m[0] * pt.x + m[3] * pt.y + m[6] * pt.z + m[9],
        m[1] * pt.x + m[4] * pt.y + m[7] * pt.z + m[10],
        m[2] * pt.x + m[5] * pt.y + m[8] * pt.z + m[11]
    );

    return pt;
}

Geometry& EuclideanTransform::operator() (Geometry& g) const
{
    const size_t d = g.GetDimension();
    const int    m = g.mat.rows; // for reshaping a packed geometry

    if (d != 3 && d != 4)
    {
        E_ERROR << "geometry matrix must store either Euclidean 3D or homogeneous 4D coordinates (d=" << d << ")";
        return g;
    }

    bool homogeneous = (d == 4);

    if (!homogeneous)
    {
        g = Geometry::MakeHomogeneous(g);
    }

    switch (g.shape)
    {
    case Geometry::ROW_MAJOR:
        g.mat = g.mat * GetTransformMatrix(homogeneous, false, g.mat.type());
        break;
    case Geometry::COL_MAJOR:
        g.mat = GetTransformMatrix(homogeneous, true, g.mat.type()) * g.mat;
        break;
    case Geometry::PACKED:
        g.mat = g.mat.reshape(1);
        g.mat = g.mat * GetTransformMatrix(homogeneous, false, g.mat.type());
        g.mat = g.mat.reshape(homogeneous ? 4 : 3, m);
        break;
    }

    return g;
}

bool EuclideanTransform::Store(cv::FileStorage& fs) const
{
    fs << "rvec" << m_rvec;
    fs << "tvec" << m_tvec;

    return true;
}

bool EuclideanTransform::Restore(const cv::FileNode& fn)
{
    cv::Mat rvec, tvec;

    fn["rvec"] >> rvec;
    fn["tvec"] >> tvec;

    return SetRotationVector(rvec) && SetTranslation(tvec);
}

VectorisableD::Vec EuclideanTransform::ToVector() const
{
    Vec v(6);
    size_t i = 0;

    for (int j = 0; j < 3; j++) v[i++] = m_rvec.at<double>(j);
    for (int j = 0; j < 3; j++) v[i++] = m_tvec.at<double>(j);

    return v;
}

bool EuclideanTransform::FromVector(const Vec& v)
{
    if (v.size() != GetDimension()) return false;

    cv::Mat rvec(3, 1, CV_64F);
    cv::Mat tvec(3, 1, CV_64F);

    size_t i = 0;

    for (int j = 0; j < 3; j++) rvec.at<double>(j) = v[i++];
    for (int j = 0; j < 3; j++) tvec.at<double>(j) = v[i++];

    return SetRotationVector(rvec) && SetTranslation(tvec);
}

//==[ Motion ]================================================================//

size_t Motion::Update(const EuclideanTransform& tform)
{
    if (m_tforms.empty())
    {
        m_tforms.push_back(tform);
    }
    else
    {
        m_tforms.push_back(m_tforms.back() << tform);
    }

    return m_tforms.size();
}

bool Motion::Store(Path& path) const
{
    std::ofstream of(path.string().c_str(), std::ios::out);
    BOOST_FOREACH (const EuclideanTransform& tform, m_tforms)
    {
        cv::Mat M = tform.GetTransformMatrix();
        double* m = M.ptr<double>();

        for (size_t i = 0; i < 12; i++) of << m[i] << (i < 11 ? " " : "");
        of << std::endl;
    }

    return true;
}

bool Motion::Restore(const Path& path)
{
    m_tforms.clear();

    std::ifstream rf(path.string().c_str(), std::ios::in);

    if (!rf.is_open())
    {
        E_ERROR << "error reading " << path;
        return false;
    }

    for (String line; std::getline(rf, line); /**/)
    {
        std::stringstream ss(line);
        std::vector<double> m(
            (std::istream_iterator<double>(ss)), // begin
            (std::istream_iterator<double>()));  // end

        if (m.size() != 12)
        {
            E_ERROR << "error parsing " << path;
            E_ERROR << "line " << (m_tforms.size() + 1) << ": " << line;

            return false;
        }

        EuclideanTransform tform;
        tform.SetTransformMatrix(cv::Mat(m).reshape(1, 3));

        m_tforms.push_back(tform);
    }

    return true;
}

//==[ ProjectionModel ]=======================================================//

//==[ PinholeModel ]==========================================================//

Point3F& PinholeModel::operator() (Point3F& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3F(
        (float)(m[0] * pt.x / pt.z + m[2]),
        (float)(m[4] * pt.y / pt.z + m[5]),
        pt.z != 0 ? pt.z : 0
    );

    return pt;
}

Point3D& PinholeModel::operator() (Point3D& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3D(
        m[0] * pt.x / pt.z + m[2],
        m[4] * pt.y / pt.z + m[5],
        pt.z != 0 ? pt.z : 0
    );

    return pt;
}

Geometry PinholeModel::Project(const Geometry& g, ProjectiveSpace space) const
{
    Geometry proj(g.shape);

    if (g.GetDimension() != 3)
    {
        E_ERROR << "given geometry has to be 3D Euclidean (d=" << g.GetDimension() << ")";
        return proj;
    }

    cv::Mat K = m_matrix;

    if (K.type() != g.mat.type())
    {
        K.convertTo(K, g.mat.type());
    }

    // compute projection
    switch (g.shape)
    {
    case Geometry::ROW_MAJOR:
        proj.mat = g.mat * K.t();
        break;

    case Geometry::COL_MAJOR:
        proj.mat = K * g.mat;
        break;

    case Geometry::PACKED:
        proj.mat = cv::Mat(g.mat.reshape(1) * K.t()).reshape(3, g.mat.rows);
        break;
    }

    // de-homogenisation
    switch (space)
    {
    case EUCLIDEAN_3D:
        // no need to do normalisation
        break;

    case EUCLIDEAN_2D:
        // dehomonise and remove the last dimension
        proj = Geometry::FromHomogeneous(proj);
        break;

    case HOMOGENEOUS_3D:
        // economy in-place dehomonisation
        proj.Dehomogenise();
        break;
    }

    return proj;
}

Geometry PinholeModel::Backproject(const Geometry& g) const
{
    const size_t d = g.GetDimension();

    if (d != 2 || d != 3)
    {
        E_ERROR << "geometry matrix must store either Euclidean 2D or homogeneous 3D coordinates (d=" << d << ")";
        return Geometry(g.shape);
    }

    // the forward projection of the inverse camera would do the work
    return PinholeModel(GetInverseCameraMatrix()).Project(
        d == 2 ? Geometry::MakeHomogeneous(g) : g,
        EUCLIDEAN_3D
    );
}

cv::Mat PinholeModel::ToProjectionMatrix(const EuclideanTransform& tform) const
{
    cv::Mat P = m_matrix * tform.GetTransformMatrix(true, true, m_matrix.type());
    return P;
}

bool PinholeModel::FromProjectionMatrix(const cv::Mat& P, const EuclideanTransform& tform)
{
    cv::Mat K = P * tform.GetInverse().GetTransformMatrix(true, true, P.type());
    return SetCameraMatrix(P);
}

bool PinholeModel::SetCameraMatrix(const cv::Mat& K)
{
    if (K.rows != 3 || K.cols != 3 || K.channels() != 1)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(K.size()) << " rather than 3x3x1";
        return false;
    }

    cv::Mat K64f;
    K.convertTo(K64f, CV_64F);

    if (K64f.at<double>(0, 1) != 0 ||
        K64f.at<double>(1, 0) != 0 ||
        K64f.at<double>(2, 0) != 0 ||
        K64f.at<double>(2, 1) != 0 ||
        K64f.at<double>(2, 2) != 1)
    {
        E_ERROR << "the matrix has to follow the form:";
        E_ERROR << "| fx,  0, cx |";
        E_ERROR << "|  0, fy, cy |";
        E_ERROR << "|  0,  0,  1 |";

        return false;
    }

    m_matrix = K64f;
    return true;
}

cv::Mat PinholeModel::GetInverseCameraMatrix() const
{
    double fx, fy, cx, cy;
    GetValues(fx, fy, cx, cy);

    return cv::Mat_<double>(3, 3) << 1.0f / fx, 0, 0, 0, 1.0f / fy, -cx / fx, -cy / fy, 1;
}

void PinholeModel::SetValues(double fx, double fy, double cx, double cy)
{
    m_matrix.at<double>(0, 0) = fx;
    m_matrix.at<double>(1, 1) = fy;
    m_matrix.at<double>(0, 2) = cx;
    m_matrix.at<double>(1, 2) = cy;
}

void PinholeModel::GetValues(double& fx, double& fy, double& cx, double& cy) const
{
    fx = m_matrix.at<double>(0, 0);
    fy = m_matrix.at<double>(1, 1);
    cx = m_matrix.at<double>(0, 2);
    cy = m_matrix.at<double>(1, 2);
}

bool PinholeModel::Store(cv::FileStorage& fs) const
{
    Point2D f, c;

    GetValues(f.x, f.y, c.x, c.y);

    fs << "focalLengths"   << f;
    fs << "principalPoint" << c;

    return true;
}

bool PinholeModel::Restore(const cv::FileNode& fn)
{
    Point2D f, c;

    fn["focalLengths"]   >> f;
    fn["principalPoint"] >> c;

    SetValues(f.x, f.y, c.x, c.y);

    return true;
}

VectorisableD::Vec PinholeModel::ToVector() const
{
    Vec v(4);
    GetValues(v[0], v[1], v[2], v[3]);

    return v;
}

bool PinholeModel::FromVector(const Vec& v)
{
    if (v.size() != GetDimension()) return false;

    SetValues(v[0], v[1], v[2], v[3]);
    return true;
}

//==[ BouguetModel ]==========================================================//

BouguetModel::BouguetModel(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
    if (!SetCameraMatrix(cameraMatrix))   E_ERROR << "error setting camera matrix";
    if (!SetDistortionCoeffs(distCoeffs)) E_ERROR << "error setting distortion coefficients";
}

Point3F& BouguetModel::operator() (Point3F& pt) const
{
    throw std::exception("not implemented");
    return pt;
}

Point3D& BouguetModel::operator() (Point3D& pt) const
{
    throw std::exception("not implemented");
    return pt;
}

Geometry BouguetModel::Project(const Geometry& g, ProjectiveSpace space) const
{
    Geometry proj(g.shape);

    if (g.GetDimension() != 3)
    {
        E_ERROR << "given geometry has to be 3D Euclidean (d=" << g.GetDimension() << ")";
        return proj;
    }

    cv::projectPoints(g.shape == Geometry::PACKED ? g.mat.reshape(1) : g.mat,
        EuclideanTransform::Identity.GetRotationMatrix(),
        EuclideanTransform::Identity.GetTranslation(),
        m_matrix, m_distCoeffs, proj.mat);

    if (g.shape == Geometry::PACKED)
    {
        proj.mat = proj.mat.reshape(3, g.mat.rows); // restore to a MxNx3 matrix
    }

    switch (space)
    {
    case HOMOGENEOUS_3D:
    case EUCLIDEAN_3D:
        return Geometry::MakeHomogeneous(proj);
    }

    return proj;
}

Geometry BouguetModel::Backproject(const Geometry& g) const
{
    throw std::exception("not implemented");
    return Geometry(g.shape);
}

bool BouguetModel::SetDistortionCoeffs(const cv::Mat& D)
{
    cv::Mat D64f = D;
    Vec d;

    if (D.type() != CV_64F)
    {
        D.convertTo(D64f, CV_64F);
    }

    d.assign(D64f.datastart, D64f.dataend);

    double fx, fy, cx, cy;
    GetValues(fx, fy, cx, cy);

    switch (d.size())
    {
    case RADIAL_TANGENTIAL_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3]);
        break;
    case HIGH_RADIAL_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4]);
        break;
    case RATIONAL_RADIAL_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        break;
    case THIN_PSISM_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11]);
        break;
    case TILTED_SENSOR_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13]);
        break;
    default:
        E_ERROR << "given matrix has " << d.size() << " element(s) instead of 4, 5, 8, 12 or 14";
        return false;
    }

    return true;
}

cv::Mat BouguetModel::GetDistortionCoeffs() const
{
    int dof = static_cast<int>(m_distModel);
    return cv::Mat(m_distCoeffs).rowRange(0, dof).clone();
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2)
{
    SetValues(fx, fy, cx, cy);
    SetDistortionModel(RADIAL_TANGENTIAL_DISTORTION);

    m_distCoeffs[0] = k1;
    m_distCoeffs[1] = k2;
    m_distCoeffs[2] = p1;
    m_distCoeffs[3] = p2;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2);
    SetDistortionModel(HIGH_RADIAL_DISTORTION);

    m_distCoeffs[4] = k3;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3);
    SetDistortionModel(RATIONAL_RADIAL_DISTORTION);

    m_distCoeffs[5] = k4;
    m_distCoeffs[6] = k5;
    m_distCoeffs[7] = k6;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6, double s1, double s2, double s3, double s4)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6);
    SetDistortionModel(THIN_PSISM_DISTORTION);

    m_distCoeffs[8]  = s1;
    m_distCoeffs[9]  = s2;
    m_distCoeffs[10] = s3;
    m_distCoeffs[11] = s4;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6, double s1, double s2, double s3, double s4, double taux, double tauy)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4);
    SetDistortionModel(TILTED_SENSOR_DISTORTION);

    m_distCoeffs[12] = taux;
    m_distCoeffs[13] = tauy;
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2) const
{
    GetValues(fx, fy, cx, cy);

    k1 = m_distCoeffs[0];
    k2 = m_distCoeffs[1];
    p1 = m_distCoeffs[2];
    p2 = m_distCoeffs[3];
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2);

    if (m_distModel < HIGH_RADIAL_DISTORTION)
    {
        k3 = 0.0f;
    }
    else
    {
        k3 = m_distCoeffs[4];
    }
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3);

    if (m_distModel < RATIONAL_RADIAL_DISTORTION)
    {
        k4 = k5 = k6 = 0.0f;
    }
    else
    {
        k4 = m_distCoeffs[5];
        k5 = m_distCoeffs[6];
        k6 = m_distCoeffs[7];
    }
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6, double& s1, double& s2, double& s3, double& s4) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6);

    if (m_distModel < THIN_PSISM_DISTORTION)
    {
        s1 = s2 = s3 = s4 = 0.0f;
    }
    else
    {
        s1 = m_distCoeffs[8];
        s2 = m_distCoeffs[9];
        s3 = m_distCoeffs[10];
        s4 = m_distCoeffs[11];
    }
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6, double& s1, double& s2, double& s3, double& s4, double& taux, double& tauy) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4);

    if (m_distModel < TILTED_SENSOR_DISTORTION)
    {
        taux = tauy = 0.0f;
    }
    else
    {
        taux = m_distCoeffs[12];
        tauy = m_distCoeffs[13];
    }
}

bool BouguetModel::Store(cv::FileStorage & fs) const
{
    if (!PinholeModel::Store(fs)) return false;
    fs << "distCoeffs" << GetDistortionCoeffs();

    return true;
}

bool BouguetModel::Restore(const cv::FileNode & fn)
{
    if (!PinholeModel::Restore(fn)) return false;

    cv::Mat distCoeffs;
    fn["distCoeffs"]>> distCoeffs;

    return SetDistortionCoeffs(distCoeffs);
}

VectorisableD::Vec BouguetModel::ToVector() const
{
    throw std::exception("not implemented");
    return Vec();
}

bool BouguetModel::FromVector(const Vec& v)
{
    throw std::exception("not implemented");
    return false;
}

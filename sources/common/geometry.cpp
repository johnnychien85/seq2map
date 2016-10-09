#include <seq2map\geometry.hpp>

using namespace seq2map;

const EuclideanTransform EuclideanTransform::Identity(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
const BouguetModel BouguetModel::s_canonical(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(5, 1, CV_64F));

namespace seq2map
{
    cv::Mat eucl2homo(const cv::Mat& x, bool rowMajor)
    {
        assert(x.channels() == 1);
        cv::Mat y;

        if (rowMajor)
        {
            y = cv::Mat(x.rows, x.cols + 1, x.type());
            x.copyTo(y.colRange(0, x.cols));
            y.col(x.cols).setTo(1);
        }
        else
        {
            y = cv::Mat(x.rows + 1, x.cols, x.type());
            x.copyTo(y.rowRange(0, x.rows));
            y.row(x.rows).setTo(1);
        }

        return y;
    }

    cv::Mat homo2eucl(const cv::Mat& x, bool rowMajor)
    {
        assert(x.channels() == 1);
        cv::Mat y;

        if (rowMajor)
        {
            assert(x.cols > 1);
            size_t n = x.cols;
            y = cv::Mat::zeros(x.rows, n - 1, x.type());

            for (size_t j = 0; j < y.cols; j++)
            {
                y.col(j) = x.col(j) / x.col(n - 1);
            }
        }
        else
        {
            assert(x.rows > 1);
            size_t n = x.rows;
            y = cv::Mat::zeros(x.rows, n - 1, x.type());
            
            for (size_t i = 0; i < y.rows; i++)
            {
                y.row(i) = x.row(i) / x.row(n - 1);
            }
        }

        return y;
    }

    cv::Mat skewsymat(const cv::Mat& x)
    {
        assert(x.total() == 3);

        cv::Mat x64f;
        x.convertTo(x64f, CV_64F);

        // normalisation to make sure ||x|| = 1
        x64f /= norm(x64f);

        double* v = x64f.ptr<double>();

        cv::Mat y64f = (cv::Mat_<double>(3, 3) <<
             0.0f, -v[2],  v[1],
             v[2],  0.0f, -v[0],
            -v[1],  v[0],  0.0f);

        cv::Mat y;
        y64f.convertTo(y, x.type());

        return y;
    }
}

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

EuclideanTransform EuclideanTransform::operator<<(const EuclideanTransform& tform) const
{
    return EuclideanTransform(tform.m_rmat * m_rmat, tform.GetRotationMatrix() * m_tvec + tform.m_tvec);
}

EuclideanTransform EuclideanTransform::operator>>(const EuclideanTransform& tform) const
{
    return tform << *this;
}

void EuclideanTransform::Apply(Point3D& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3D(
        m[0] * pt.x + m[1] * pt.y + m[2]  * pt.z + m[3],
        m[4] * pt.x + m[5] * pt.y + m[6]  * pt.z + m[7],
        m[8] * pt.x + m[9] * pt.y + m[10] * pt.z + m[11]
    );
}

void EuclideanTransform::Apply(Points3D& pts) const
{
    cv::Mat G;
    cv::convertPointsToHomogeneous(pts, G);
    G = G.reshape(1) * GetTransformMatrix(false, false);
    G.reshape(3).copyTo(cv::Mat(pts));
}

bool EuclideanTransform::SetRotationMatrix(const cv::Mat& rmat)
{
    if (rmat.rows != 3 || rmat.cols != 3)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(rmat.size()), " rather than 3x3";
        return false;
    }

    cv::Rodrigues(rmat, m_rvec);
    SetRotationVector(m_rvec);

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

cv::Mat EuclideanTransform::GetTransformMatrix(bool sqrMat, bool preMult) const
{
    if (!sqrMat && preMult) return m_matrix;

    cv::Mat matrix;
    
    if (sqrMat)
    {
        matrix = cv::Mat::eye(4, 4, m_matrix.type());
        m_matrix.copyTo(matrix.rowRange(0, 3).colRange(0, 4));
    }
    else
    {
        matrix = m_matrix;
    }

    return preMult ? matrix : matrix.t();
}

cv::Mat EuclideanTransform::GetEssentialMatrix() const
{
    return skewsymat(m_tvec) * m_rmat;
}

EuclideanTransform EuclideanTransform::GetInverse() const
{
    cv::Mat Rt = m_rmat.t();
    return EuclideanTransform(Rt, -Rt*m_tvec);
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

    for (size_t j = 0; j < 3; j++) v[i++] = m_rvec.at<double>(j);
    for (size_t j = 0; j < 3; j++) v[i++] = m_tvec.at<double>(j);

    return v;
}

bool EuclideanTransform::FromVector(const Vec& v)
{
    if (v.size() != GetDimension()) return false;

    cv::Mat rvec(3, 1, CV_64F);
    cv::Mat tvec(3, 1, CV_64F);

    size_t i = 0;

    for (size_t j = 0; j < 3; j++) rvec.at<double>(j) = v[i++];
    for (size_t j = 0; j < 3; j++) tvec.at<double>(j) = v[i++];

    return SetRotationVector(rvec) && SetTranslation(tvec);
}

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
    std::ofstream of(path.string(), std::ios::out);
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

    std::ifstream rf(path.string(), std::ios::in);

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

VectorisableD::Vec MotionEstimation::Initialise()
{
    if (0 /*m_pnp.GetSize() > m_epi.GetSize()*/)
    {
        cv::Mat rvec, tvec;
        cv::solvePnP(m_pnp.From(), m_pnp.To(), m_cameraMatrix, cv::Mat(), rvec, tvec, false, CV_EPNP);
        m_transform.SetRotationVector(rvec);
        m_transform.SetTranslation(tvec);
    }
    else
    {
        cv::Mat E = cv::findEssentialMat(m_epi.From(), m_epi.To(), m_cameraMatrix);
        cv::Mat rmat, tvec;
        cv::recoverPose(E, m_epi.From(), m_epi.To(), m_cameraMatrix, rmat, tvec);
        m_transform.SetRotationMatrix(rmat);
        m_transform.SetTranslation(tvec);
    }

    //cv::Mat x0 = cv::Mat::zeros(6, 1, CV_64F);
    //m_transform.GetRotationVector().copyTo(x0.rowRange(0, 3));
    //m_transform.GetTranslation().copyTo(x0.rowRange(3, 6));

    m_invCameraMatrix = m_cameraMatrix.inv();
    m_conds = GetEvaluationSize();

    //cv::Mat W_pnp = cv::Mat(m_wpnp);
    //cv::Mat(W_pnp / cv::sum(W_pnp)[0] * m_wpnp.size()).copyTo(W_pnp);

    //cv::Mat x64f;
    //x0.convertTo(x64f, CV_64F);

    return m_transform.ToVector(); //x64f;
}

size_t MotionEstimation::GetEvaluationSize() const
{
    size_t m = 0;

    if (m_alpha < 1) m += m_separatedRpe ? m_pnp.GetSize() * 2 : m_pnp.GetSize();
    if (m_alpha > 0) m += m_sampsonError ? m_epi.GetSize() : m_epi.GetSize() * 2;

    return m;
}

VectorisableD::Vec MotionEstimation::Evaluate(const VectorisableD::Vec& x) const
{
    EuclideanTransform transform;

    transform.FromVector(x);

    //transform.SetRotationVector(x.rowRange(0, 3));
    //transform.SetTranslation(x.rowRange(3, 6));

    double a_rpe = 1 - m_alpha;
    double a_epi = m_alpha;

    a_rpe = a_rpe; ///** (m_pnp.GetSize() + m_epi.GetSize())*/ / m_pnp.GetSize();
    a_epi = a_epi; ///** (m_pnp.GetSize() + m_epi.GetSize())*/ / m_epi.GetSize();

    assert(a_rpe >= 0 && a_epi >= 0);

    cv::Mat rpe = a_rpe > 0 ? a_rpe * EvalReprojectionConds(transform) : cv::Mat();
    cv::Mat epi = a_epi > 0 ? a_epi * EvalEpipolarConds(transform)     : cv::Mat();

    if (rpe.empty()) return epi;
    if (epi.empty()) return rpe;

    cv::Mat y;
    cv::vconcat(rpe, epi, y);

    return y;
}

cv::Mat MotionEstimation::EvalEpipolarConds(const EuclideanTransform& transform) const
{
    cv::Mat x0 = cv::Mat(m_epi.From()).reshape(1);
    cv::Mat x1 = cv::Mat(m_epi.To()).reshape(1);

    cv::Mat K_inv = m_invCameraMatrix;
    cv::Mat F = K_inv.t() * transform.GetEssentialMatrix() * K_inv;
    cv::Mat Fx0 = eucl2homo(x0) * F.t();
    cv::Mat Fx1 = eucl2homo(x1) * F;
    cv::Mat nn0 = Fx0.col(0).mul(Fx0.col(0)) + Fx0.col(1).mul(Fx0.col(1));
    cv::Mat nn1 = Fx1.col(0).mul(Fx1.col(0)) + Fx1.col(1).mul(Fx1.col(1));
    cv::Mat xFx = Fx0.col(0).mul(x1.col(0)) + Fx0.col(1).mul(x1.col(1)) + Fx0.col(2);

    if (m_sampsonError)
    {
        // Sampson Approximation:
        //
        //              x0' * F * x1
        // ---------------------------------------
        // sqrt(Fx00^2 + Fx01^2 + Fx10^2 + Fx11^2)

        cv::Mat nn; cv::sqrt(nn0 + nn1, nn);

        return xFx.mul(1 / nn);
    }
    else
    {
        // Symmetric Geometric Error
        //
        //        x0' * F * x1              x0' * F * x1
        // ( ---------------------- , ---------------------- )
        //    sqrt(Fx00^2 + Fx01^2)    sqrt(Fx10^2 + Fx11^2)

        cv::Mat gerr = cv::Mat(xFx.rows, 2, xFx.type());
        cv::sqrt(nn0, nn0);
        cv::sqrt(nn1, nn1);

        gerr.col(0) = xFx.mul(1 / nn0);
        gerr.col(1) = xFx.mul(1 / nn1);
        
        return gerr.reshape(1, gerr.total());
    }
}

cv::Mat MotionEstimation::EvalReprojectionConds(const EuclideanTransform& transform) const
{
    Points3D pts3d = m_pnp.From();
    Points2D pts2d(m_pnp.GetSize());

    cv::projectPoints(pts3d, transform.GetRotationVector(), transform.GetTranslation(), m_cameraMatrix, cv::Mat(), pts2d);

    cv::Mat rpe = cv::Mat(cv::Mat(pts2d) - cv::Mat(m_pnp.To())).reshape(1);

    rpe.col(0) = rpe.col(0).mul(cv::Mat(m_wpnp));
    rpe.col(1) = rpe.col(1).mul(cv::Mat(m_wpnp));

    if (m_separatedRpe)
    {
        return rpe.reshape(1, rpe.total());
    }

    cv::multiply(rpe, rpe, rpe);
    cv::sqrt(cv::Mat(rpe.col(0) + rpe.col(1)), rpe);

    //rpe.convertTo(rpe, CV_64F);

    return rpe;
}

/*
bool MotionEstimation::SetSolution(const VectorisableD::Vec& x)
{
    //if (x.rows != 6 || x.cols != 1) return false;

    //cv::Mat x32f;
    //x.convertTo(x32f, CV_32F);

    //m_transform.SetRotationVector(x.rowRange(0, 3));
    //m_transform.SetTranslation(x.rowRange(3, 6));
    
    return x; // x32f
}*/

bool MotionEstimation::Store(Path& path) const
{
    std::ofstream of(path.string(), std::ios::out);

    of << mat2string(m_cameraMatrix, "egomo.K") << std::endl;
    of << mat2string(m_invCameraMatrix, "egomo.K_inv") << std::endl;
    of << mat2string(cv::Mat(m_pnp.From()).reshape(1), "egomo.pnp.src") << std::endl;
    of << mat2string(cv::Mat(m_pnp.To()  ).reshape(1), "egomo.pnp.dst") << std::endl;
    of << mat2string(cv::Mat(m_wpnp      ).reshape(1), "egomo.pnp.w")   << std::endl;
    of << mat2string(cv::Mat(m_upnp      ).reshape(1), "egomo.pnp.uid") << std::endl;
    of << mat2string(cv::Mat(m_epi.From()).reshape(1), "egomo.epi.src") << std::endl;
    of << mat2string(cv::Mat(m_epi.To()  ).reshape(1), "egomo.epi.dst") << std::endl;
    of << mat2string(cv::Mat(m_uepi      ).reshape(1), "egomo.epi.uid") << std::endl;
    of << mat2string(m_transform.GetTransformMatrix(), "egomo.M") << std::endl;

    return true;
}

void OptimalTriangulator::Triangulate(const PointMap2Dto2D& map, Points3D& pts, std::vector<double>& err)
{
    size_t n = map.GetSize();

    pts.clear();
    pts.resize(n);

    err.clear();
    err.resize(n);

    cv::Mat x4h, x3d(pts);
    cv::triangulatePoints(m_projMatrix0, m_projMatrix1, map.From(), map.To(), x4h);

    x3d = x3d.reshape(1);

    // convert homogeneous coordinates to Euclidean
    homo2eucl(x4h.t()).copyTo(x3d);

    for (size_t i = 0; i < n; i++)
    {
        err[i] = 0.0f;
    }
}

void MidPointTriangulator::DecomposeProjMatrix(const cv::Mat& P, cv::Mat& KRinv, cv::Mat& c)
{
    cv::Mat KR = P.rowRange(0, 3).colRange(0, 3);
    KRinv = KR.inv();
    c = -KRinv * P.rowRange(0, 3).col(3);
}

void MidPointTriangulator::Triangulate(const PointMap2Dto2D& map, Points3D& pts, std::vector<double>& err)
{
    size_t n = map.GetSize();

    pts.clear();
    pts.resize(n);

    err.clear();
    err.resize(n);

    cv::Mat x3d = cv::Mat(pts).reshape(1);

    cv::Mat KRinv0, c0;
    cv::Mat KRinv1, c1;
    
    DecomposeProjMatrix(m_projMatrix0, KRinv0, c0);
    DecomposeProjMatrix(m_projMatrix1, KRinv1, c1);
    
    //std::ofstream of("tri.m");
    //of << mat2string(cv::Mat(map.From()).reshape(1), "x0") << std::endl;
    //of << mat2string(cv::Mat(map.To())  .reshape(1), "x1") << std::endl;

    cv::Mat t = c1 - c0;
    cv::Mat m = c0 + c1;
    cv::Mat x0 = eucl2homo(cv::Mat(map.From()).reshape(1)) * KRinv0.t();
    cv::Mat x1 = eucl2homo(cv::Mat(map.To()  ).reshape(1)) * KRinv1.t();

    //x0.convertTo(x0, CV_64F);
    //x1.convertTo(x1, CV_64F);
    //t.convertTo(t, CV_64F);
    //m.convertTo(m, CV_64F);
   
    //of << mat2string(m_projMatrix0, "P0") << std::endl;
    //of << mat2string(m_projMatrix1, "P1") << std::endl;
    //of << mat2string(x0, "x0_h") << std::endl;
    //of << mat2string(x1, "x1_h") << std::endl;

    for (size_t i = 0; i < map.GetSize(); i++)
    {
        cv::Mat Bt = cv::Mat(2, 3, x0.type());
        x0.row(i).copyTo(Bt.row(0));
        x1.row(i).copyTo(Bt.row(1));

        cv::Mat At = Bt.clone();
        At.row(1) = -Bt.row(1);

        cv::Mat A = At.t();
        cv::Mat B = Bt.t();
        cv::Mat k = (At * A).inv() * At * t;
        cv::Mat g = (B * k + m) * 0.5f;
        cv::Mat d = (A * k - t);

        //g.convertTo(g, CV_32F);

        //if (i == 0)
        //{
        //    of << mat2string(A, "A0") << std::endl;
        //    of << mat2string((At*A).inv(), "A0Ai") << std::endl;
        //    of << mat2string(k, "k0") << std::endl;
        //}

        x3d.row(i) = g.t();
        err[i] = cv::norm(d);
    }

    //of << mat2string(x3d, "g") << std::endl;
}

BouguetModel::BouguetModel(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
    if (!SetCameraMatrix(cameraMatrix)) E_ERROR << "error setting camera matrix";
    if (!SetDistCoeffs  (distCoeffs  )) E_ERROR << "error setting distortion coefficients";
}

bool BouguetModel::SetCameraMatrix(const cv::Mat& cameraMatrix)
{
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(cameraMatrix.size()) << " rather than 3x3";
        return false;
    }

    cameraMatrix.convertTo(m_cameraMatrix, m_cameraMatrix.empty() ? cameraMatrix.type() : m_cameraMatrix.type());

    return true;
}

bool BouguetModel::SetDistCoeffs(const cv::Mat& distCoeffs)
{
    switch (distCoeffs.total())
    {
    case 4:
    case 5:
    case 8:
        distCoeffs.convertTo(m_distCoeffs, m_distCoeffs.empty() ? distCoeffs.type() : m_distCoeffs.type());
        return true;
    default:
        m_distCoeffs = s_canonical.m_distCoeffs.clone();
        return false;
    }
}

void BouguetModel::SetValues(double fu, double fv, double uc, double vc, double k1, double k2, double p1, double p2, double k3)
{
    m_cameraMatrix.at<double>(0, 0) = fu;
    m_cameraMatrix.at<double>(1, 1) = fv;
    m_cameraMatrix.at<double>(0, 2) = uc;
    m_cameraMatrix.at<double>(1, 2) = vc;
    m_distCoeffs.at<double>(0) = k1;
    m_distCoeffs.at<double>(1) = k2;
    m_distCoeffs.at<double>(2) = p1;
    m_distCoeffs.at<double>(3) = p2;
    m_distCoeffs.at<double>(4) = k3;
}

void BouguetModel::GetValues(double& fu, double& fv, double& uc, double& vc, double& k1, double& k2, double& p1, double& p2, double& k3) const
{
    fu = m_cameraMatrix.at<double>(0, 0);
    fv = m_cameraMatrix.at<double>(1, 1);
    uc = m_cameraMatrix.at<double>(0, 2);
    vc = m_cameraMatrix.at<double>(1, 2);
    k1 = m_distCoeffs.at<double>(0);
    k2 = m_distCoeffs.at<double>(1);
    p1 = m_distCoeffs.at<double>(2);
    p2 = m_distCoeffs.at<double>(3);
    k3 = m_distCoeffs.at<double>(4);
}

cv::Mat BouguetModel::MakeProjectionMatrix(const EuclideanTransform& pose) const
{
    cv::Mat P = cv::Mat::eye(3, 4, m_cameraMatrix.type());
    m_cameraMatrix.copyTo(P.rowRange(0, 3).colRange(0, 3));

    return cv::Mat(P * pose.GetTransformMatrix(true, true));
}

bool BouguetModel::Store(cv::FileStorage & fs) const
{
    fs << "cameraMatrix" << m_cameraMatrix;
    fs << "distCoeffs"   << m_distCoeffs;

    return true;
}

bool BouguetModel::Restore(const cv::FileNode & fn)
{
    cv::Mat cameraMatrix, distCoeffs;

    fn["cameraMatrix"] >> cameraMatrix;
    fn["distCoeffs"]   >> distCoeffs;

    return SetCameraMatrix(cameraMatrix) && SetDistCoeffs(distCoeffs);
}

void BouguetModel::Project(const Points3D& pts3d, Points2D& pts2d) const
{
    cv::projectPoints(pts3d,
        EuclideanTransform::Identity.GetRotationMatrix(),
        EuclideanTransform::Identity.GetTranslation(),
        m_cameraMatrix, m_distCoeffs, pts2d);
}

VectorisableD::Vec BouguetModel::ToVector() const
{
    Vec v(9);
    GetValues(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);

    return v;
}

bool BouguetModel::FromVector(const Vec& v)
{
    assert(v.size() == GetDimension());
    SetValues(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);

    return true;
}

#include <seq2map\geometry.hpp>

using namespace seq2map;

const EuclideanTransform EuclideanTransform::Identity(cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(3, 1, CV_32F));
const BouguetModel BouguetModel::s_canonical(cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(5, 1, CV_32F));

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

void EuclideanTransform::Apply(Point3F& pt) const
{
    const float* m = m_matrix.ptr<float>();

    pt = Point3F(
        m[0] * pt.x + m[1] * pt.y + m[2]  * pt.z + m[3],
        m[4] * pt.x + m[5] * pt.y + m[6]  * pt.z + m[7],
        m[8] * pt.x + m[9] * pt.y + m[10] * pt.z + m[11]
    );
}

void EuclideanTransform::Apply(Points3F& pts) const
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

    cv::Mat rvec32f, rmat;
    rvec.convertTo(rvec32f, CV_32F);

    cv::Rodrigues(m_rvec = rvec32f, rmat);
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

    cv::Mat tvec32f;
    tvec.convertTo(tvec32f, CV_32F);
    tvec32f.reshape(0, 3).copyTo(m_tvec);

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

EuclideanTransform Motion::GetGlobalTransform(size_t to) const
{
    return m_tforms[to];
}

EuclideanTransform Motion::GetLocalTransform(size_t from, size_t to) const
{
    return m_tforms[from].GetInverse() >> m_tforms[to];
}


cv::Mat MotionEstimation::Initialise()
{
    m_conds = m_pnp.GetSize();

    cv::Mat x0 = cv::Mat::zeros(6, 1, CV_32F);
    m_transform.GetRotationVector().copyTo(x0.rowRange(0, 3));
    m_transform.GetTranslation().copyTo(x0.rowRange(3, 6));

    cv::Mat x64f;
    x0.convertTo(x64f, CV_64F);

    return x64f;
}

cv::Mat MotionEstimation::Evaluate(const cv::Mat& x) const
{
    EuclideanTransform transform;
    transform.SetRotationVector(x.rowRange(0, 3));
    transform.SetTranslation(x.rowRange(3, 6));

    Points3F pts3d = m_pnp.From();
    Points2F pts2d(m_pnp.GetSize());

    //transform.Apply(pts3d);
    cv::projectPoints(pts3d, transform.GetRotationVector(), transform.GetTranslation(), m_cameraMatrix, cv::Mat(), pts2d);

    cv::Mat rpe = cv::Mat(cv::Mat(pts2d) - cv::Mat(m_pnp.To())).reshape(1);
    cv::multiply(rpe, rpe, rpe);
    cv::sqrt(cv::Mat(rpe.col(0) + rpe.col(1)), rpe);

    //x3dj = cv::Mat(x3dj.rowRange(0, 3).t()).reshape(3);
    //Points2F x2dj2(x2dj.size());
    //cv::projectPoints(x3dj, Mij.GetRotationVector(), Mij.GetTranslation(), K, cv::Mat(), x2dj2);
    //x3dj = x3dj.reshape(1);

    rpe.convertTo(rpe, CV_64F);

    return rpe;
}

bool MotionEstimation::SetSolution(const cv::Mat& x)
{
    if (x.rows != 6 || x.cols != 1) return false;

    cv::Mat x32f;
    x.convertTo(x32f, CV_32F);

    m_transform.SetRotationVector(x.rowRange(0, 3));
    m_transform.SetTranslation(x.rowRange(3, 6));
    
    return true;
}

void OptimalTriangulator::Triangulate(const PointMap2Dto2D& map, Points3F& pts)
{
    pts.clear();
    pts.resize(map.GetSize());

    cv::Mat x4h, x3d(pts);
    cv::triangulatePoints(m_projMatrix0, m_projMatrix1, map.From(), map.To(), x4h);

    x3d = x3d.reshape(1);

    // convert homogeneous coordinates to Euclidean
    x3d.col(0) = (x4h.row(0) / x4h.row(3)).t();
    x3d.col(1) = (x4h.row(1) / x4h.row(3)).t();
    x3d.col(2) = (x4h.row(2) / x4h.row(3)).t();
}

BouguetModel::BouguetModel(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
    if (!SetCameraMatrix(cameraMatrix)) E_ERROR << "error setting camera matrix";
    if (!SetDistCoeffs(distCoeffs))     E_ERROR << "error setting distortion coefficients";
}

bool BouguetModel::SetCameraMatrix(const cv::Mat& cameraMatrix)
{
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(cameraMatrix.size()) << " rather than 3x3";
        return false;
    }
    m_cameraMatrix = cameraMatrix.clone();
    return true;
}

bool BouguetModel::SetDistCoeffs(const cv::Mat& distCoeffs)
{
    switch (distCoeffs.total())
    {
    case 4:
    case 5:
    case 8:
        m_distCoeffs = distCoeffs.clone();
        return true;
    default:
        m_distCoeffs = s_canonical.m_distCoeffs.clone();
        return false;
    }
}

cv::Mat BouguetModel::MakeProjectionMatrix(const EuclideanTransform& pose) const
{
    cv::Mat P = cv::Mat::eye(3, 4, m_cameraMatrix.type());
    m_cameraMatrix.copyTo(P.rowRange(0, 3).colRange(0, 3));
    P = P * pose.GetTransformMatrix(true, true);

    return P;
}

bool BouguetModel::Store(cv::FileStorage & fs) const
{
    fs << "cameraMatrix" << m_cameraMatrix;
    fs << "distCoeffs" << m_distCoeffs;

    return true;
}

bool BouguetModel::Restore(const cv::FileNode & fn)
{
    cv::Mat cameraMatrix, distCoeffs;

    fn["cameraMatrix"] >> cameraMatrix;
    fn["distCoeffs"]   >> distCoeffs;

    return SetCameraMatrix(cameraMatrix) && SetDistCoeffs(distCoeffs);
}

void BouguetModel::Project(const Points3F& pts3d, Points2F& pts2d) const
{
    cv::projectPoints(pts3d, cv::Mat(), cv::Mat(), m_cameraMatrix, m_distCoeffs, pts2d);
}

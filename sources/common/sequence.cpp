#include <seq2map\sequence.hpp>

using namespace seq2map;

const BouguetIntrinsics BouguetIntrinsics::s_canonical(cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(5, 1, CV_32F));

void Pose::Transform(Points3F& points) const
{
    // TODO: finish the Euclidean transform
}

bool Pose::SetRotationMatrix(const cv::Mat& rmat)
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

bool Pose::SetRotationVector(const cv::Mat& rvec)
{
    if (rvec.total() != 3)
    {
        E_ERROR << "given vector has " << rvec.total() << " element(s) rather than 3";
        return false;
    }

    cv::Mat rmat;
    cv::Rodrigues(m_rvec = rvec, rmat);

    rmat.copyTo(m_rmat);

    return true;
}

bool Pose::SetTranslation(const cv::Mat& tvec)
{
    if (tvec.total() != 3)
    {
        E_ERROR << "given vector has " << tvec.total() << " element(s) rather than 3";
        return false;
    }

    tvec.reshape(0, 3).copyTo(m_tvec);
    return true;
}

bool Pose::SetMatrix(const cv::Mat& matrix)
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

cv::Mat Pose::GetMatrix(bool squareMatrix, bool preMult) const
{
    if (!squareMatrix && preMult) return m_matrix;

    cv::Mat matrix = cv::Mat::eye(4, 4, CV_32F);

    // TODO: finish the matrix formation code
    // m_matrix.copyTo(matrix.rowRange(0, 3).colRange(0, 4));
    // ...

    return matrix;
}

bool Pose::Store(cv::FileStorage& fs) const
{
    fs << "rvec" << m_rvec;
    fs << "tvec" << m_tvec;

    return true;
}

bool Pose::Restore(const cv::FileNode& fn)
{
    cv::Mat rvec, tvec;

    fn["rvec"] >> rvec;
    fn["tvec"] >> tvec;

    return SetRotationVector(rvec) && SetTranslation(tvec);
}

Frame Camera::GetFrame(size_t idx) const
{
    Frame frame(idx < m_imageStore.GetSize() ? idx : INVALID_INDEX);
    PersistentImage image;

    m_imageStore.Retrieve(idx, image);
    m_featureStore.Retrieve(idx, frame.features);

    frame.im = image.im;

    return frame;
}

void Camera::World2Image(const Points3F& worldPts, Points2F& imagePts) const
{
    Points3F cameraPts;

    World2Camera(worldPts, cameraPts);
    Camera2Image(cameraPts, imagePts);
}

void Camera::Camera2Image(const Points3F& cameraPts, Points2F& imagePts) const
{
    if (!m_intrinsics)
    {
        E_ERROR << "no intrinsics set";
        return;
    }

    m_intrinsics->Project(cameraPts, imagePts);
}

Camera::IntrinsicsFactory::IntrinsicsFactory()
{
    Factory::Register<BouguetIntrinsics>("BOUGUET");
}

BouguetIntrinsics::BouguetIntrinsics(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
    if (!SetCameraMatrix(cameraMatrix)) E_ERROR << "error setting camera matrix";
    if (!SetDistCoeffs(distCoeffs))     E_ERROR << "error setting distortion coefficients";
}

bool BouguetIntrinsics::SetCameraMatrix(const cv::Mat& cameraMatrix)
{
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(cameraMatrix.size()) << " rather than 3x3";
        return false;
    }
    m_cameraMatrix = cameraMatrix.clone();
    return true;
}

bool BouguetIntrinsics::SetDistCoeffs(const cv::Mat& distCoeffs)
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

bool seq2map::BouguetIntrinsics::Store(cv::FileStorage & fs) const
{
    fs << "cameraMatrix" << m_cameraMatrix;
    fs << "distCoeffs" << m_distCoeffs;

    return true;
}

bool seq2map::BouguetIntrinsics::Restore(const cv::FileNode & fn)
{
    cv::Mat cameraMatrix, distCoeffs;

    fn["cameraMatrix"] >> cameraMatrix;
    fn["distCoeffs"]   >> distCoeffs;

    return SetCameraMatrix(cameraMatrix) && SetDistCoeffs(distCoeffs);
}

void BouguetIntrinsics::Project(const Points3F& pts3d, Points2F& pts2d) const
{
    cv::projectPoints(pts3d, cv::Mat(), cv::Mat(), m_cameraMatrix, m_distCoeffs, pts2d);
}

bool Sequence::Store(Path& path) const
{
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);

    fs << "sequenceName" << m_seqName;
    fs << "sequencePath" << m_seqPath.string();
    fs << "grabber"      << m_grabberName;

    if (m_featureDxtor)
    {
        fs << "featureDetextractor" << "{:";
        m_featureDxtor->Store(fs);
        fs << "}";
    }

    fs << "cameras" << "[";
    BOOST_FOREACH(const Camera& cam, m_cameras)
    {
        Path imagesRelPath   = getRelativePath(cam.m_imageStore.GetRootPath(),   m_seqPath);
        Path featuresRelPath = getRelativePath(cam.m_featureStore.GetRootPath(), m_seqPath);

        fs << "{:";
        fs << "cameraName" << cam.m_cameraName;
        fs << "imageSize"  << cam.m_imageSize;
        fs << "extrinsics" << "{:"; cam.m_extrinsics.Store(fs); fs << "}";
        fs << "intrinsics" << "{:";
        if (cam.m_intrinsics) {
            fs << "model" << cam.m_intrinsics->GetModelName();
            cam.m_intrinsics->Store(fs);
        }
        fs << "}";
        fs << "imageStorage"   << "{:"; cam.m_imageStore.  Store(fs, imagesRelPath  ); fs << "}";
        fs << "featureStorage" << "{:"; cam.m_featureStore.Store(fs, featuresRelPath); fs << "}";
        fs << "}";
    }
    fs << "]";

    return true;
}

bool Sequence::Restore(const Path& path)
{
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    Clear();

    try
    {
        String seqPath;

        fs["sequenceName"] >> m_seqName;
        fs["sequencePath"] >> seqPath;
        fs["grabber"]      >> m_grabberName;

        m_seqPath = seqPath;

        // restore the feature detextractor
        if (!fs["featureDetextractor"].empty())
        {
            FeatureDetextractorFactory factory;
            FeatureDetextractorPtr dxtor = factory.Create(fs["featureDetextractor"]);

            if (dxtor)
            {
                m_featureDxtor = dxtor;
                E_INFO << "feature detector and extractor restored";
            }
        }

        // restore cameras
        cv::FileNode camsNode = fs["cameras"];
        for (cv::FileNodeIterator itr = camsNode.begin(); itr != camsNode.end(); itr++)
        {
            Camera cam;
            String intrinsicsModel;
            cv::FileNode fn = *itr;

            fn["cameraName"] >> cam.m_cameraName;
            fn["imageSize"]  >> cam.m_imageSize;
            fn["intrinsics"]["model"] >> intrinsicsModel;

            cam.m_extrinsics.Restore(fn["extrinsics"]);

            Camera::IntrinsicsFactory factory;
            Camera::Intrinsics::Ptr intrinsics = factory.Create(intrinsicsModel);

            if (!intrinsics)
            {
                E_ERROR << "error creating camera intrinsics \"" << intrinsicsModel << "\"";
                return false;
            }

            if (!intrinsics->Restore(fn["intrinsics"]))
            {
                E_ERROR << "error restoring intrinsics";
                return false;
            }

            if (!cam.m_imageStore.Restore(fn["imageStorage"]))
            {
                E_ERROR << "error restoring image storage";
                return false;
            }

            if (!cam.m_featureStore.Restore(fn["featureStorage"]))
            {
                E_ERROR << "error restoring feature storage";
                return false;
            }

            cam.SetIntrinsics(intrinsics);
            cam.m_imageStore.SetRootPath(m_seqPath / cam.m_imageStore.GetRootPath());
            cam.m_featureStore.SetRootPath(m_seqPath / cam.m_featureStore.GetRootPath());

            m_cameras.push_back(cam);
        }

    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught while restoring sequence";
        E_ERROR << ex.what();
        return false;
    }

    return true;
}
/*
void Sequence::WriteFileList(cv::FileStorage& fs, const String& node, const Path& root, const Strings& filenames)
{
    fs << node << "{:";
    fs << "root" << root.string();
    fs << "files" << "[";
    BOOST_FOREACH(const String& filename, filenames)
    {
        fs << filename;
    }
    fs << "]";
    fs << "}";
}
*/

void Sequence::Clear()
{
    m_seqPath = "";
    m_seqName = "";
    m_grabberName = "";
    m_cameras.clear();
}

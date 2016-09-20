#include <seq2map\sequence.hpp>

using namespace seq2map;

Camera::IntrinsicsFactory::IntrinsicsFactory()
{
    Register<BouguetModel>("BOUGUET");
}

void Camera::World2Image(const Points3D& worldPts, Points2D& imagePts) const
{
    Points3D cameraPts;

    World2Camera(worldPts, cameraPts);
    Camera2Image(cameraPts, imagePts);
}

void Camera::Camera2Image(const Points3D& cameraPts, Points2D& imagePts) const
{
    if (!m_intrinsics)
    {
        E_ERROR << "missing intrinsics";
        return;
    }

    m_intrinsics->Project(cameraPts, imagePts);
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
            ProjectionModel::Ptr intrinsics = factory.Create(intrinsicsModel);

            if (!intrinsics)
            {
                E_ERROR << "unknown camera model \"" << intrinsicsModel << "\"";
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

void Sequence::Clear()
{
    m_seqPath = "";
    m_seqName = "";
    m_grabberName = "";
    m_cameras.clear();
}

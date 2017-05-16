#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/make_shared.hpp>
#include <seq2map/sequence.hpp>

using namespace seq2map;

//==[ UUID ]==================================================================//

UUID::UUID() : m_uuid(boost::uuids::random_generator()())
{
}

UUID UUID::Generate(const String& seed)
{
    return UUID(boost::uuids::string_generator()(seed));
}

String UUID::ToString() const
{
    return boost::lexical_cast<String>(m_uuid);
}

bool UUID::FromString(const String& uuid)
{
    try
    {
        m_uuid = boost::lexical_cast<boost::uuids::uuid>(uuid);
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error converting string \"" << uuid << "\" to an uuid object";
        E_ERROR << ex.what();

        return false;
    }

    return true;
}

//==[ Entity ]================================================================//

bool Entity::Store(cv::FileStorage& fs) const
{
    fs << "name" << m_name;
    fs << "uuid" << m_uuid.ToString();

    return true;
}

bool Entity::Restore(const cv::FileNode& fn)
{
    try
    {
        String uuid;

        fn["name"] >> m_name;
        fn["uuid"] >> uuid;

        if (!m_uuid.FromString(uuid))
        {
            E_ERROR << "error restoring uuid";
            return false;
        }

    }
    catch (std::exception& ex)
    {
        E_ERROR << "error restoring entitiy";
        E_ERROR << ex.what();

        return false;
    }

    return true;
}

//==[ Sensor ]================================================================//

bool Sensor::Store(cv::FileStorage& fs) const
{
    if (!Entity::Store(fs))
    {
        E_ERROR << "error restoring entity";
        return false;
    }

    fs << "model" << m_model;
    fs << "extrinsics" << "{";
    bool success = m_extrinsics.Store(fs);
    fs << "}";

    return success;
}

bool Sensor::Restore(const cv::FileNode& fn)
{
    if (!Entity::Restore(fn))
    {
        E_ERROR << "error storing entity";
        return false;
    }

    fn["model"] >> m_model;

    return m_extrinsics.Restore(fn["extrinsics"]);
}

//==[ FeatureStore ]==========================================================//

bool FeatureStore::Create(const Path& root, size_t cam, FeatureDetextractor::Ptr dxtor)
{
    m_camIdx = cam;
    m_dxtor  = dxtor;

    return SequentialFileStore<ImageFeatureSet>::Create(root);
}

bool FeatureStore::Store(cv::FileStorage& fs) const
{
    if (!SequentialFileStore::Store(fs))
    {
        return false;
    }

    if (!m_dxtor) // no detextractor available?
    {
        return true;
    }

    fs << "camIdx" << m_camIdx;

    fs << "dxtor" << "{";
    bool dxtorStored = m_dxtor->Store(fs);
    fs << "}";

    if (!dxtorStored)
    {
        E_ERROR << "error storing feature detextractor";
    }

    return dxtorStored;
}

bool FeatureStore::Restore(const cv::FileNode& fn)
{
    if (!SequentialFileStore::Restore(fn))
    {
        return false;
    }

    fn["camIdx"] >> m_camIdx;

    cv::FileNode dxtorNode = fn["dxtor"];

    if (dxtorNode.empty())
    {
        E_INFO << "feature detextractor not found";
        m_dxtor.reset();

        return true;
    }

    m_dxtor = FeatureDetextractorFactory::GetInstance().Create(dxtorNode);

    if (!m_dxtor)
    {
        E_ERROR << "error creating feature detextractor from file node";
        return false;
    }

    return true;
}

//==[ DisparityStore ]========================================================//

LinearSpacedVec<double> DisparityStore::s_dspace16U(0.0f, 65535.0f);

/*
bool DisparityStore::Create(const Path& root, size_t priCamIdx, size_t secCamIdx, const Strings& filenames)
{
    m_priCamIdx = priCamIdx;
    m_secCamIdx = secCamIdx;

    return SequentialFileStore<PersistentImage>::Create(root, filenames);
}
*/

bool DisparityStore::Create(const Path& root, size_t priCamIdx, size_t secCamIdx, StereoMatcher::Ptr matcher)
{
    m_priCamIdx = priCamIdx;
    m_secCamIdx = secCamIdx;
    m_matcher   = matcher;

    if (m_matcher)
    {
        m_dspace = m_matcher->GetDisparitySpace();
        UpdateMappings();
    }

    return SequentialFileStore<PersistentImage>::Create(root);
}

bool DisparityStore::Store(cv::FileStorage& fs) const
{
    if (!SequentialFileStore::Store(fs))
    {
        return false;
    }

    fs << "priCamIdx" << m_priCamIdx;
    fs << "secCamIdx" << m_secCamIdx;

    fs << "dspace" << "{";
    {
        fs << "begin" << m_dspace.begin;
        fs << "end"   << m_dspace.end;
        fs << "segs"  << m_dspace.segs;
    }
    fs << "}";

    if (!m_matcher) // no stereo matcher available?
    {
        return true;
    }

    fs << "matcher" << "{";
    bool matcherStored = m_matcher->Store(fs);
    fs << "}";

    if (!matcherStored)
    {
        E_ERROR << "error storing stereo matcher";
        return false;
    }

    return true;
}

bool DisparityStore::Restore(const cv::FileNode& fn)
{
    if (!SequentialFileStore::Restore(fn))
    {
        return false;
    }

    try
    {
        fn["priCamIdx"] >> m_priCamIdx;
        fn["secCamIdx"] >> m_secCamIdx;

        cv::FileNode dspaceNode = fn["dspace"];
        dspaceNode["begin"] >> m_dspace.begin;
        dspaceNode["end"]   >> m_dspace.end;
        dspaceNode["segs"]  >> m_dspace.segs;

        UpdateMappings();

        cv::FileNode matcherNode = fn["matcher"];

        if (matcherNode.empty())
        {
            m_matcher.reset();
            return true;
        }

        m_matcher = StereoMatcherFactory::GetInstance().Create(matcherNode);

        if (!m_matcher)
        {
            E_ERROR << "error creating stereo matcher from file node";
            return false;
        }

        if (m_matcher->GetDisparitySpace() != m_dspace)
        {
            E_ERROR << "disparity space not compatible";
            return false;
        }
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error restoring from file node";
        E_ERROR << ex.what();

        return false;
    }

    return true;
}

bool DisparityStore::Append(Path& to, const PersistentImage& data) const
{
    cv::Mat dpm32F = data.im;
    cv::Mat dpm16U;

    if (dpm32F.type() != CV_32F)
    {
        E_ERROR << "given matrix is not a single precision disparity map";
        return false;
    }

    // linearily remap the disparity map to a 16U image
    dpm32F.convertTo(dpm16U, CV_16U, m_dspaceTo16U.alpha, m_dspaceTo16U.beta);

    return cv::imwrite(to.string(), dpm16U);
}

bool DisparityStore::Retrieve(const Path& from, PersistentImage& data) const
{
    cv::Mat dpm16U = cv::imread(from.string());

    if (dpm16U.empty())
    {
        E_ERROR << "error reading " << from;
        return false;
    }

    if (dpm16U.type() != CV_16U)
    {
        E_ERROR << "not a 16-bit single channel disparity image: " << from;
        return false;
    }

    // linearily remap the 16U image to a float disparity image
    dpm16U.convertTo(data.im, CV_32F, m_dspaceTo32F.alpha, m_dspaceTo32F.beta);

    return true;
}

void DisparityStore::UpdateMappings()
{
    const LinearSpacedVec<double>& src = m_dspace;
    const LinearSpacedVec<double>& dst = s_dspace16U;

    src.GetLinearMappingTo(dst, m_dspaceTo16U.alpha, m_dspaceTo16U.beta);
    dst.GetLinearMappingTo(src, m_dspaceTo32F.alpha, m_dspaceTo32F.beta);
}

//==[ Camera ]================================================================//

void Camera::IntrinsicsFactory::Init()
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

bool Camera::Store(cv::FileStorage& fs) const
{
    if (!Sensor::Store(fs))
    {
        E_ERROR << "error storing sensor";
        return false;
    }

    fs << "index" << (int) GetIndex();
    fs << "imageSize" << m_imageSize;
    
    if (m_intrinsics) {
        fs << "projection" << m_intrinsics->GetModelName();
        fs << "intrinsics" << "{";
        if (!m_intrinsics->Store(fs))
        {
            return false;
        }
        fs << "}";
    }

    fs << "imageStore" << "{";
    if (!m_imageStore.Store(fs))
    {
        return false;
    }
    fs << "}";

    return true;
}

bool Camera::Restore(const cv::FileNode& fn)
{
    if (!Sensor::Restore(fn))
    {
        E_ERROR << "error restoring sensor";
        return false;
    }

    String projectionModel;
    int index;

    fn["index"]      >> index;
    fn["projection"] >> projectionModel;
    fn["imageSize"]  >> m_imageSize;

    SetIndex(index);

    if (projectionModel.empty())
    {
        m_intrinsics.reset();
    }
    else
    {
        m_intrinsics = Camera::IntrinsicsFactory::GetInstance().Create(projectionModel);

        if (!m_intrinsics)
        {
            E_ERROR << "unknown camera model \"" << projectionModel << "\"";
            return false;
        }

        if (!m_intrinsics->Restore(fn["intrinsics"]))
        {
            E_ERROR << "error restoring intrinsics from file node";
            return false;
        }
    }

    return m_imageStore.Restore(fn["imageStore"]);
}


//==[ RectifiedStereo ]=======================================================//

RectifiedStereo::RectifiedStereo(const Camera& primary, const Camera& secondary)
{
    Create(boost::make_shared<const Camera>(primary), boost::make_shared<const Camera>(secondary));
}

bool RectifiedStereo::Create(const Camera::ConstPtr& primary, const Camera::ConstPtr& secondary)
{
    m_primary = primary;
    m_secondary = secondary;

    // Calculate geometric parameters..
    // ...
    // ..
    // .

    return true;
}

String RectifiedStereo::ToString() const
{
    std::stringstream ss;
    ss << "(" << m_primary->GetIndex() << "," << m_secondary->GetIndex() << ")";

    return ss.str();
}

bool RectifiedStereo::SetActiveStore(size_t store)
{
    if (store >= m_stores.size())
    {
        E_WARNING << "index out of bound (idx=" << store << ", items=" << m_stores.size() << ")";
        return false;
    }

    m_activeStore = store;
    return true;
}
/*
bool RectifiedStereo::Store(cv::FileStorage& fn) const
{
    if (!m_primary || !m_secondary)
    {
        E_ERROR << "missing camera(s)";
        return false;
    }

    fn << "primary"   << m_primary->GetIndex();
    fn << "secondary" << m_secondary->GetIndex();

    return true;
}

bool RectifiedStereo::Restore(const cv::FileNode& fs)
{

}
*/
cv::Mat RectifiedStereo::GetDepthMap(size_t frame, size_t store) const
{
    if (store >= m_stores.size())
    {
        E_ERROR << "store index " << store << " out of bound";
        return cv::Mat();
    }

    PersistentImage data;
    const SequentialFileStore<PersistentImage>& dpStore = m_stores[store];

    if (!dpStore.Retrieve(frame, data))
    {
        E_ERROR << "error restoring disparity map from store " << store << " frame " << frame;
        return cv::Mat();
    }

    // convert to disparities to a depth map
    cv::Mat dp = data.im;

    return dp;
}

//==[ Sequence ]==============================================================//

String Sequence::s_storeIndexFileName    = "index.yml";
String Sequence::s_featureStoreDirName   = "kpt";
String Sequence::s_disparityStoreDirName = "dpm";
String Sequence::s_mapStoreDirName       = "map";

void Sequence::Clear()
{
    m_rawPath = "";
    m_seqPath = "";
    m_seqName = "";
    m_vehicleName = "";
    m_grabberName = "";

    m_kptsDirName = s_featureStoreDirName;
    m_dispDirName = s_disparityStoreDirName;
    m_mapsDirName = s_mapStoreDirName;

    m_cameras.clear();
    m_stereo.clear();
}

bool Sequence::Store(Path& path) const
{
    if (dirExists(path))
    {
        path /= s_storeIndexFileName;
    }

    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        E_ERROR << "error writing to " << path;
        return false;
    }

    fs << "sequence"  << m_seqName;
    fs << "source"    << m_rawPath;
    fs << "vehicle"   << m_vehicleName;
    fs << "grabber"   << m_grabberName;
    fs << "keypoints" << m_kptsDirName;
    fs << "disparity" << m_dispDirName;
    fs << "map"       << m_mapsDirName;

    fs << "cameras" << "[";
    BOOST_FOREACH(const Camera& cam, m_cameras)
    {
        fs << "{";
        if (!cam.Store(fs))
        {
            E_ERROR << "error storing camera " << cam.GetName();
            return false;
        }
        fs << "}";
    }
    fs << "]";

    fs << "stereo" << "[";
    BOOST_FOREACH(const RectifiedStereo& stereo, m_stereo)
    {
        fs << "{";
        /*if (!stereo.Store(fs))
        {
            E_ERROR << "error storing stereo pair " << stereo;
            return false;
        }*/
        if (stereo.m_primary && stereo.m_secondary)
        {
            fs << "primary"   << (int) stereo.m_primary->GetIndex();
            fs << "secondary" << (int) stereo.m_secondary->GetIndex();
        }

        fs << "}";
    }
    fs << "]";

    return true;
}

bool Sequence::Restore(const Path& path)
{
    Path from = dirExists(path) ? path / s_storeIndexFileName : path;

    cv::FileStorage fs(from.string(), cv::FileStorage::READ);
    Clear();

    if (!fs.isOpened())
    {
        E_ERROR << "error reading " << from;
        return false;
    }

    try
    {
        fs["sequence"]  >> m_seqName;
        fs["source"]    >> m_rawPath;
        fs["vehicle"]   >> m_vehicleName;
        fs["grabber"]   >> m_grabberName;
        fs["keypoints"] >> m_kptsDirName;
        fs["disparity"] >> m_dispDirName;
        fs["map"]       >> m_mapsDirName;

        m_seqPath = fullpath(from.parent_path());

        // restore cameras
        cv::FileNode camsNode = fs["cameras"];

        m_cameras = Cameras(camsNode.size());

        for (cv::FileNodeIterator itr = camsNode.begin(); itr != camsNode.end(); itr++)
        {
            int idx;
            (*itr)["index"] >> idx;

            if (idx >= m_cameras.size())
            {
                E_ERROR << "camera index out of bound (idx=" << idx << ", size=" << m_cameras.size() << ")";
                return false;
            }

            if (!m_cameras[idx].Restore(*itr))
            {
                E_ERROR << "error restoring camera from file node";
                return false;
            }

            E_INFO << "camera " << idx << " restored";
        }

        // restore stereo pairs
        cv::FileNode stereoNode = fs["stereo"];
        for (cv::FileNodeIterator itr = stereoNode.begin(); itr != stereoNode.end(); itr++)
        {
            int primary, secondary;
            (*itr)["primary"]   >> primary;
            (*itr)["secondary"] >> secondary;

            if (primary   < 0 || primary   >= m_cameras.size() ||
                secondary < 0 || secondary >= m_cameras.size())
            {
                E_WARNING << "stereo pair (" << primary << "," << secondary << ") out of bound";
                E_WARNING << "#camera: " << m_cameras.size();

                continue;
            }

            m_stereo.push_back(RectifiedStereo(m_cameras[primary], m_cameras[secondary]));
        }
    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught while restoring sequence";
        E_ERROR << ex.what();

        return false;
    }

    size_t stores = ScanStores();
    E_INFO << "sequence successfully restored from " << from << " with " << stores << " store(s) found";

    return true;
}

bool Sequence::FindFeatureStore(size_t index, FeatureStore const* &store) const
{
    store = NULL;

    BOOST_FOREACH (const Camera& cam, m_cameras)
    {
        BOOST_FOREACH (const FeatureStore& f, cam.GetFeatureStores())
        {
            if (f.GetIndex() == index)
            {
                store = &f;
                return true;
            }
        }
    }

    return false;
}

size_t Sequence::ScanStores()
{
    size_t featureStores = 0, dispStores = 0;

    // remove all bound stores first
    BOOST_FOREACH(Camera& cam, m_cameras)
    {
        cam.m_featureStores.clear();
    }

    BOOST_FOREACH(RectifiedStereo& pair, m_stereo)
    {
        pair.m_stores.clear();
    }

    const Path featureStorePath   = m_seqPath / m_kptsDirName;
    const Path disparityStorePath = m_seqPath / m_dispDirName;

    // scan feature stores
    E_INFO << "scanning feature stores " << featureStorePath;
    Paths featureDirs = enumerateDirs(featureStorePath);

    BOOST_FOREACH (const Path& dir, featureDirs)
    {
        E_INFO << "trying " << dir;

        FeatureStore store;
        const Path from = dir / s_storeIndexFileName;

        if (!((Persistent<Path>&)store).Restore(from))
        {
            continue;
        }

        size_t camIdx = store.GetCameraIndex();

        if (camIdx >= m_cameras.size())
        {
            E_INFO << "feature store restored but abandoned";
            E_INFO << "reason being the owning camera index out of bound(idx = " << camIdx << ", size = " << m_cameras.size() << ")";

            continue;
        }
            
        Camera& cam = m_cameras[camIdx];

        if (store.GetItems() != cam.GetFrames())
        {
            E_INFO << "feature store restored but abandoned";
            E_INFO << "reason being mismatch of frame numbers (items=" << store.GetItems() << ", frames=" << cam.GetFrames() << ")";

            continue;
        }

        store.SetIndex(featureStores);
        cam.m_featureStores.push_back(store);
        featureStores++;

        E_INFO << "feature store loaded to camera " << cam.GetIndex();
    }

    // scan disparity stores
    E_INFO << "scanning disparity stores " << disparityStorePath;
    Paths disparityDirs = enumerateDirs(disparityStorePath);

    BOOST_FOREACH (const Path& dir, disparityDirs)
    {
        E_INFO << "trying " << dir;

        DisparityStore store;
        const Path from = dir / s_storeIndexFileName;

        if (!((Persistent<Path>&)store).Restore(from))
        {
            continue;
        }

        size_t priCamIdx = store.GetPrimaryCameraIndex();
        size_t secCamIdx = store.GetSecondaryCameraIndex();

        // search for the owning stereo pair
        bool found = false;
        BOOST_FOREACH (RectifiedStereo& pair, m_stereo)
        {
            if (pair.m_primary  ->GetIndex() == priCamIdx &&
                pair.m_secondary->GetIndex() == secCamIdx)
            {
                found = true;

                if (store.GetItems() == pair.m_primary->GetFrames())
                {
                    store.SetIndex(dispStores);
                    pair.m_stores.push_back(store);
                    dispStores++;

                    E_INFO << "disparity store loaded to stereo pair " << pair.ToString();
                }
                else
                {
                    E_INFO << "disparity store restored but abandoned";
                    E_INFO << "reason being mismatch of frame numbers (items=" << store.GetItems() << ", frames=" << pair.m_primary->GetFrames() << ")";
                }

                break;
            }
        }

        if (!found)
        {
            E_INFO << "disparity store restored but abandoned";
            E_INFO << "reason being missing stereo pair (" << priCamIdx << "," << secCamIdx << ")";
        }
    }

    return featureStores + dispStores;
}

//==[ Sequence::Builder ]=====================================================//

bool Sequence::Builder::Build(const Path& from, const String& name, const String& grabber, Sequence& seq) const
{
    seq.Clear();

    seq.m_rawPath = from;
    seq.m_seqName = name;
    seq.m_vehicleName = GetVehicleName(from);
    seq.m_grabberName = grabber;

    return BuildCamera(from, seq.m_cameras, seq.m_stereo);
}

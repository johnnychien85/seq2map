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

bool FeatureStore::Create(const Path& root, Camera::ConstOwn& camera, FeatureDetextractor::Own& dxtor)
{
    m_cam    = camera;
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

    fs << "camIdx" << (m_cam ? m_cam->GetIndex() : INVALID_INDEX);

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

bool DisparityStore::Create(const Path& root, RectifiedStereo::ConstOwn& stereo, StereoMatcher::ConstOwn& matcher)
{
    m_stereo  = stereo;
    m_matcher = matcher;

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

    Camera::ConstOwn priCam = m_stereo ? m_stereo->GetPrimaryCamera()   : Camera::ConstOwn();
    Camera::ConstOwn secCam = m_stereo ? m_stereo->GetSecondaryCamera() : Camera::ConstOwn();

    fs << "priCamIdx" << (priCam ? priCam->GetIndex() : INVALID_INDEX);
    fs << "secCamIdx" << (secCam ? secCam->GetIndex() : INVALID_INDEX);

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
    cv::Mat dpm16U = cv::imread(from.string(), cv::IMREAD_UNCHANGED);

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
    Register<PinholeModel>("PINHOLE");
    Register<BouguetModel>("BOUGUET");
}

/*
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
*/

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

    if (index != GetIndex())
    {
        E_ERROR << "index not matched";
        return false;
    }

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

bool RectifiedStereo::Create(Camera::ConstOwn& cam0, Camera::ConstOwn& cam1)
{
    m_priCam = cam0;
    m_secCam = cam1;

    // Calculate geometric parameters..
    // ...
    // ..
    // .

    return true;
}

String RectifiedStereo::ToString() const
{
    std::stringstream ss;
    ss << "(" << m_priCam->GetIndex() << "," << m_secCam->GetIndex() << ")";

    return ss.str();
}

/*
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
*/

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
    m_kptsStores.clear();
    m_dispStores.clear();
}

RectifiedStereo::ConstOwn Sequence::FindStereoPair(size_t priCamIdx, size_t secCamIdx) const
{
    BOOST_FOREACH(RectifiedStereo::ConstOwn pair, m_stereo)
    {
        if (!pair) continue;

        Camera::ConstOwn cam0 = pair->GetPrimaryCamera();
        Camera::ConstOwn cam1 = pair->GetSecondaryCamera();

        if (cam0 && cam1 && cam0->GetIndex() == priCamIdx && cam1->GetIndex() == secCamIdx)
        {
            return pair;
        }
    }

    return RectifiedStereo::ConstOwn();
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
    BOOST_FOREACH(const Camera::Map::value_type& pair, m_cameras)
    {
        if (!pair.second) continue; // ignore invalid reference
        const Camera& cam = *pair.second;

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
    BOOST_FOREACH(const RectifiedStereo::Own& ptr, m_stereo)
    {
        if (!ptr) continue; // ignore invalid reference

        const RectifiedStereo& stereo = *ptr;
        const Camera::ConstOwn& pri = stereo.m_priCam;
        const Camera::ConstOwn& sec = stereo.m_secCam;

        if (!pri || !sec)
        {
            E_ERROR << "stereo pair " << stereo.ToString() << " referencing missing camera(s)";;
            continue;
        }

        fs << "{";
        fs << "primary"   << pri->GetIndex();
        fs << "secondary" << sec->GetIndex();
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

        for (cv::FileNodeIterator itr = camsNode.begin(); itr != camsNode.end(); itr++)
        {
            size_t idx;
            (*itr)["index"] >> idx;

            Camera::Own& cam = m_cameras[idx];

            if (cam)
            {
                E_ERROR << "camera " << cam->GetIndex() << " already exists";
                return false;
            }

            cam = Camera::New(idx);

            if (!cam->Restore(*itr))
            {
                E_ERROR << "error restoring camera " << cam->GetIndex() << " from file node";
                return false;
            }

            E_INFO << "camera " << idx << " restored";
        }

        // restore stereo pairs
        cv::FileNode stereoNode = fs["stereo"];

        for (cv::FileNodeIterator itr = stereoNode.begin(); itr != stereoNode.end(); itr++)
        {
            size_t priCamIdx, secCamIdx;
            (*itr)["primary"]   >> priCamIdx;
            (*itr)["secondary"] >> secCamIdx;

            Camera::Own& priCam = m_cameras[priCamIdx];
            Camera::Own& secCam = m_cameras[secCamIdx];

            if (!priCam)
            {
                E_ERROR << "referecing to a missing primary camera";
            }

            if (!secCam)
            {
                E_ERROR << "referencing to a missing secondary camera";
            }

            RectifiedStereo::Own stereo = RectifiedStereo::Own(new RectifiedStereo(Camera::ConstOwn(priCam), Camera::ConstOwn(secCam)));
            std::pair<RectifiedStereo::Set::iterator, bool> result = m_stereo.insert(stereo);

            if (!result.second)
            {
                E_WARNING << "stereo pair " << stereo->ToString() << " already exists";
            }
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

size_t Sequence::ScanStores()
{
    const Path featureStorePath   = m_seqPath / m_kptsDirName;
    const Path disparityStorePath = m_seqPath / m_dispDirName;

    // scan feature stores
    Paths featureDirs = enumerateDirs(featureStorePath);
    E_INFO << "scanning " << featureDirs.size() << " feature store(s) in " << featureStorePath;

    BOOST_FOREACH (const Path& dir, featureDirs)
    {
        E_INFO << "processing " << dir;

        FeatureStore::Own store = FeatureStore::New(m_kptsStores.size());
        const Path from = dir / s_storeIndexFileName;

        cv::FileStorage fs(from.string(), cv::FileStorage::READ);
        cv::FileNode fn = fs.root();

        if (!fs.isOpened() || !store->Restore(fn))
        {
            E_INFO << "failed restoring from " << from;
            continue;
        }

        size_t camIdx;
        fn["camIdx"] >> camIdx;

        Camera::ConstOwn cam = m_cameras[camIdx];

        if (!cam)
        {
            E_INFO << "feature store restored but abandoned";
            E_INFO << "reason being the owning camera is missing (idx = " << camIdx << ")";

            continue;
        }
            
        if (store->GetItems() != cam->GetFrames())
        {
            E_INFO << "feature store restored but abandoned";
            E_INFO << "reason being mismatch of frame numbers (items=" << store->GetItems() << ", frames=" << cam->GetFrames() << ")";

            continue;
        }

        store->m_cam = cam;
        m_kptsStores[store->GetIndex()] = store;

        E_INFO << "feature store " << store->GetIndex() << " loaded to camera " << cam->GetIndex();
    }

    // scan disparity stores
    Paths disparityDirs = enumerateDirs(disparityStorePath);
    E_INFO << "scanning " << disparityDirs.size() << " disparity store(s) in " << disparityStorePath;

    BOOST_FOREACH (const Path& dir, disparityDirs)
    {
        E_INFO << "processing " << dir;

        DisparityStore::Own store = DisparityStore::New(m_dispStores.size());
        const Path from = dir / s_storeIndexFileName;

        cv::FileStorage fs(from.string(), cv::FileStorage::READ);
        cv::FileNode fn = fs.root();

        if (!fs.isOpened() || !store->Restore(fn))
        {
            E_INFO << "failed restoring from " << from;
            continue;
        }

        size_t priCamIdx, secCamIdx;

        fn["priCamIdx"] >> priCamIdx;
        fn["secCamIdx"] >> secCamIdx;

        // search for the owning stereo pair
        RectifiedStereo::ConstOwn pair = FindStereoPair(priCamIdx, secCamIdx);

        if (!pair)
        {
            E_INFO << "disparity store restored but abandoned";
            E_INFO << "reason being missing stereo pair (" << priCamIdx << "," << secCamIdx << ")";

            continue;
        }

        if (store->GetItems() != GetFrames())
        {
            E_INFO << "disparity store restored but abandoned";
            E_INFO << "reason being mismatch of frame numbers (items=" << store->GetItems() << ", frames=" << GetFrames() << ")";

            continue;
        }

        store->m_stereo = pair;
        m_dispStores[store->GetIndex()] = store;

        E_INFO << "disparity store " << store->GetIndex() << " loaded to stereo pair " << pair->ToString();
    }

    return m_kptsStores.size() + m_dispStores.size();
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

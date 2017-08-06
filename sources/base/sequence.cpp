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

template<typename T> 
Geometry Camera::GetImagePoints() const
{
    Geometry g(Geometry::PACKED);

    int type = CV_MAKE_TYPE(cv::DataType<T>::depth, 3);
    g.mat = cv::Mat(m_imageSize.height, m_imageSize.width, type);

    typedef cv::Point3_<T> ptype;

    for (int i = 0; i < g.mat.rows; i++)
    {
        for (int j = 0; j < g.mat.cols; j++)
        {
            g.mat.at<ptype>(i, j) = ptype(j, i, 1);
        }
    }

    return g;
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

RectifiedStereo::Configuration RectifiedStereo::GetConfiguration(const EuclideanTransform& rel, double& baseline)
{
    static const double EPS = 1e-2;

    cv::Mat tvec = rel.GetTranslation();

    if (rel.GetRotation() != EuclideanTransform::Identity.GetRotation())
    {
        E_WARNING << "inconsistent rotational extrinsics";
        return UNKNOWN;
    }

    // determine configuration
    double t[3];
    t[0] = tvec.at<double>(0);
    t[1] = tvec.at<double>(1);
    t[2] = tvec.at<double>(2);

    if (t[0] > 0 && t[0] > t[1] && t[0] > t[1])
    {
        double ty = std::abs(t[1]);
        double tz = std::abs(t[2]);

        if (ty > EPS)
        {
            E_WARNING << "invalid left-right geometry : the length of ty = " << ty << " is not zero";
            return UNKNOWN;
        }

        if (tz > EPS)
        {
            E_WARNING << "invalid left-right geometry : the length of tz = " << tz << " is not zero";
            return UNKNOWN;
        }

        baseline = t[0];
        return LEFT_RIGHT;
    }
    else if (t[1] > 0 && t[1] > t[0] && t[1] > t[2])
    {
        double tx = std::abs(t[0]);
        double tz = std::abs(t[2]);

        if (tx > EPS)
        {
            E_WARNING << "invalid top-bottom geometry : the length of tz = " << tx << " is not zero";
            return UNKNOWN;
        }

        if (tz > EPS)
        {
            E_WARNING << "invalid top-bottom geometry : the length of tz = " << tz << " is not zero";
            return UNKNOWN;
        }

        baseline = t[1];
        return TOP_BOTTOM;
    }
    else if (t[2] > 0)
    {
        double tx = std::abs(t[0]);
        double ty = std::abs(t[1]);

        if (tx > EPS)
        {
            E_WARNING << "invalid back-forward geometry : the length of tz = " << tx << " is not zero";
            return UNKNOWN;
        }

        if (ty > EPS)
        {
            E_WARNING << "invalid back-forward geometry : the length of ty = " << ty << " is not zero";
            return UNKNOWN;
        }

        baseline = t[2];
        return BACK_FORWARD;
    }

    E_WARNING << "unable to determine the geometric configuration from relative pose";
    return UNKNOWN;
}

Geometry RectifiedStereo::Backproject(const cv::Mat& dp, const Indices& idx) const
{
    if (dp.rows != m_rays.mat.rows || dp.cols != m_rays.mat.cols || dp.channels() != 1)
    {
        E_ERROR << "given disparity map has an invalid dimension of " << dp.rows << "x" << dp.cols << "x" << dp.channels();
        E_ERROR << "expected " << m_rays.mat.rows << "x" << m_rays.mat.cols << "x1";

        return Geometry(m_rays.shape);
    }

    const int m = static_cast<int>(idx.empty() ? m_rays.GetElements() : idx.size());

    const cv::Mat src = (idx.empty() ? m_rays : m_rays[idx]).mat.reshape(1, m);
    /***/ cv::Mat dst = cv::Mat(src.rows, src.cols, src.type());
    /***/ cv::Mat dpm = idx.empty() ? dp.reshape(1, m) : cv::Mat();

    if (!idx.empty())
    {
        const cv::Mat dpv = dp.reshape(1, dp.total());
        dpm = cv::Mat(m, 1, dp.depth());

        int j = 0;
        BOOST_FOREACH (size_t i, idx)
        {
            dpv.row(static_cast<int>(i)).copyTo(dpm.row(j++));
        }
    }

    if (dpm.depth() != dst.depth())
    {
        dpm.convertTo(dpm, dst.depth());
    }

    cv::divide(src.col(2), dpm, dst.col(2));          // z = k/d
    cv::multiply(src.col(0), dst.col(2), dst.col(0)); // x = z*x
    cv::multiply(src.col(1), dst.col(2), dst.col(1)); // y = z*y

    return idx.empty() ?
        Geometry(Geometry::ROW_MAJOR, dst).Reshape(m_rays) :
        Geometry(Geometry::ROW_MAJOR, dst);
}

StructureEstimation::Estimate RectifiedStereo::Backproject(const cv::Mat& dp, const cv::Mat& var, const Indices& idx) const
{
    StructureEstimation::Estimate estimate(Backproject(dp, idx));

    if (!var.empty())
    {
        if (var.rows != dp.rows || var.cols != dp.cols || var.channels() != 1)
        {
            E_ERROR << "given variance map is ill-formed (shape=" << var.rows << "x" << var.cols << "x" << var.channels() << ")";
            E_ERROR << "the estimated structure will have no error metric";

            return estimate;
        }
    }

    // the Jacobian matrix:
    //
    //          / 1/f  0  -x/d \
    // J = Z(d) |  0  1/f -y/d |
    //          \  0   0  -1/d /
    //             Jx  Jy  Jdp

    const double f = m_depthDispRatio / m_baseline;

    cv::Mat dp2; cv::multiply(dp, dp, dp2);
    cv::Mat jxy = estimate.structure.mat.col(2) * (1.0f/f);
    cv::Mat jdp = Backproject(-dp2, idx).mat;

    // lazy way.. using the magnitude of the dominating ellipsoidal axis
    // cv::Mat cov = cv::Mat(jdp.rows, 1, jdp.depth());
    // cv::reduce(jac.mul(jac), cov, 1, cv::REDUCE_SUM);
    // cv::sqrt(cov, cov);
    // estimate.metric = Metric::Own(new WeightedEuclideanMetric(1 / cov));

    // rigorous way.. calculating full covariance matrix with all three ellipsoidal axes
    MahalanobisMetric* metric = new MahalanobisMetric(MahalanobisMetric::ANISOTROPIC_ROTATED, 3);
    cv::Mat cov = cv::Mat(estimate.structure.GetElements(), 6, estimate.structure.mat.depth());

    // apply the variance map if provided
    if (!var.empty())
    {
        cv::Mat sigma;

        cv::sqrt(Geometry(Geometry::PACKED, var)[idx].mat, sigma);
        cv::multiply(jdp, sigma, jdp);
    }

    cov.col(0) = jdp.col(0).mul(jdp.col(0)) + jxy.mul(jxy);
    cov.col(1) = jdp.col(0).mul(jdp.col(1));
    cov.col(2) = jdp.col(0).mul(jdp.col(2));
    cov.col(3) = jdp.col(1).mul(jdp.col(1)) + jxy.mul(jxy);
    cov.col(4) = jdp.col(1).mul(jdp.col(2));
    cov.col(5) = jdp.col(2).mul(jdp.col(2));

    if (!metric->SetCovarianceMat(cov))
    {
        E_ERROR << "error setting covariance matrix";
    }
    else
    {
        estimate.metric = Metric::Own(metric);
    }

    return estimate;
}

bool RectifiedStereo::Create(Camera::ConstOwn& pri, Camera::ConstOwn& sec)
{
    Clear();

    // check pointers valadity
    if (!pri || !sec)
    {
        E_WARNING << "missing camera(s)";
        return false;
    }

    m_priCam = pri;
    m_secCam = sec;

    // check intrinsics
    boost::shared_ptr<const PinholeModel> priProj = boost::dynamic_pointer_cast<const PinholeModel>(pri->GetIntrinsics());
    boost::shared_ptr<const PinholeModel> secProj = boost::dynamic_pointer_cast<const PinholeModel>(sec->GetIntrinsics());

    if (!priProj || !secProj)
    {
        E_WARNING << "missing intrinsics";
        return false;
    }

    if (*priProj != *secProj)
    {
        E_WARNING << "inconsistent pinhole projection parameters";
        return false;
    }

    // check extrinsics
    // find the relative transform from the secondary camera to the primary
    EuclideanTransform rel = pri->GetExtrinsics() - sec->GetExtrinsics();

    // try to identify geometric configuration and the baseline
    if ((m_config = GetConfiguration(rel, m_baseline)) == UNKNOWN)
    {
        E_WARNING << "invalid geometrical configuration";
        return false;
    }

    // calculate corresponding disparity-depth conversion factor
    double fx, fy, cx, cy;
    priProj->GetValues(fx, fy, cx, cy);

    switch (m_config)
    {
    case LEFT_RIGHT:
        m_depthDispRatio = m_baseline * fx; // z = \frac{bf}{d}
        break;
    case TOP_BOTTOM:
        m_depthDispRatio = m_baseline * fy; // z = \frac{bf}{d}
        break;
    case BACK_FORWARD:
        throw std::exception("not implemented");
        break;
    }

    // calculate back-projected rays
    m_rays = priProj->Backproject(pri->GetImagePoints<double>());
    m_rays.Reshape(Geometry::ROW_MAJOR).mat.col(2).setTo(m_depthDispRatio);

    // all good
    return true;
}

void RectifiedStereo::Clear()
{
    m_priCam.reset();
    m_secCam.reset();

    m_config = UNKNOWN;
    m_baseline = 0;
    m_depthDispRatio = 0;
}

String RectifiedStereo::ToString() const
{
    std::stringstream ss;
    ss << "(" << m_priCam->GetIndex() << "," << m_secCam->GetIndex() << ")";

    return ss.str();
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
    m_kptsStores.clear();
    m_dispStores.clear();
}

RectifiedStereo::ConstOwn Sequence::GetStereoPair(size_t priCamIdx, size_t secCamIdx) const
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

            E_TRACE << "camera " << idx << " restored";
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
    E_TRACE << "sequence successfully restored from " << from << " with " << stores << " store(s) found";

    return true;
}

size_t Sequence::ScanStores()
{
    const Path featureStorePath   = m_seqPath / m_kptsDirName;
    const Path disparityStorePath = m_seqPath / m_dispDirName;

    // scan feature stores
    Paths featureDirs = enumerateDirs(featureStorePath);
    E_TRACE << "scanning " << featureDirs.size() << " feature store(s) in " << featureStorePath;

    BOOST_FOREACH (const Path& dir, featureDirs)
    {
        E_TRACE << "processing " << dir;

        FeatureStore::Own store = FeatureStore::New(m_kptsStores.size());
        const Path from = dir / s_storeIndexFileName;

        cv::FileStorage fs(from.string(), cv::FileStorage::READ);
        cv::FileNode fn = fs.root();

        if (!fs.isOpened() || !store->Restore(fn))
        {
            E_WARNING << "failed restoring from " << from;
            continue;
        }

        size_t camIdx;
        fn["camIdx"] >> camIdx;

        Camera::ConstOwn cam = m_cameras[camIdx];

        if (!cam)
        {
            E_TRACE << "feature store restored but abandoned";
            E_TRACE << "reason being the owning camera is missing (idx = " << camIdx << ")";

            continue;
        }
            
        if (store->GetItems() != cam->GetFrames())
        {
            E_TRACE << "feature store restored but abandoned";
            E_TRACE << "reason being mismatch of frame numbers (items=" << store->GetItems() << ", frames=" << cam->GetFrames() << ")";

            continue;
        }

        store->m_cam = cam;
        m_kptsStores[store->GetIndex()] = store;

        E_TRACE << "feature store " << store->GetIndex() << " loaded to camera " << cam->GetIndex();
    }

    // scan disparity stores
    Paths disparityDirs = enumerateDirs(disparityStorePath);
    E_TRACE << "scanning " << disparityDirs.size() << " disparity store(s) in " << disparityStorePath;

    BOOST_FOREACH (const Path& dir, disparityDirs)
    {
        E_TRACE << "processing " << dir;

        DisparityStore::Own store = DisparityStore::New(m_dispStores.size());
        const Path from = dir / s_storeIndexFileName;

        cv::FileStorage fs(from.string(), cv::FileStorage::READ);
        cv::FileNode fn = fs.root();

        if (!fs.isOpened() || !store->Restore(fn))
        {
            E_WARNING << "failed restoring from " << from;
            continue;
        }

        size_t priCamIdx, secCamIdx;

        fn["priCamIdx"] >> priCamIdx;
        fn["secCamIdx"] >> secCamIdx;

        // search for the owning stereo pair
        RectifiedStereo::ConstOwn pair = GetStereoPair(priCamIdx, secCamIdx);

        if (!pair)
        {
            E_TRACE << "disparity store restored but abandoned";
            E_TRACE << "reason being missing stereo pair (" << priCamIdx << "," << secCamIdx << ")";

            continue;
        }

        if (store->GetItems() != GetFrames())
        {
            E_TRACE << "disparity store restored but abandoned";
            E_TRACE << "reason being mismatch of frame numbers (items=" << store->GetItems() << ", frames=" << GetFrames() << ")";

            continue;
        }

        store->m_stereo = pair;
        m_dispStores[store->GetIndex()] = store;

        E_TRACE << "disparity store " << store->GetIndex() << " loaded to stereo pair " << pair->ToString();
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

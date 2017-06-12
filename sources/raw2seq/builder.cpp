#include <iomanip>
#include <boost/algorithm/string/predicate.hpp>
#include "builder.hpp"

using namespace std;
namespace po = boost::program_options;

//==[ KittiOdometryBuilder ]====================================================

size_t KittiOdometryBuilder::ParseIndex(const String& varname, size_t offset)
{
    size_t idx;
    std::istringstream iss(String(varname.begin() + offset, varname.end()));
    iss >> idx;

    // check if all the entries of the variable name have benn
    //  succesfully converted to an integer
    return iss.eof() ? idx : INVALID_INDEX;
}

KittiOdometryBuilder::Calib::Calib(const Path& from)
: m_okay(false)
{
    ifstream in(from.string().c_str(), std::ios::in);

    if (!in.is_open())
    {
        E_ERROR << "error reading " << from.string();
        return;
    }

    String line;
    size_t i = 0;

    while (getline(in, line))
    {
        Strings toks = explode(line, ' ');
        i++;

        if (toks.empty()) continue; // skip an empty line

        if (toks[0].back() != ':')
        {
            E_WARNING << "error parsing " << from << " line " << i;
            E_WARNING << "the line begins with an invalid variable name \"" << toks[0] << "\"";

            continue;
        }

        // strip the colon char
        toks[0].pop_back();

        // VAR: x0 x1 x2 ...
        String var = toks.front();
        Strings x = Strings(toks.begin() + 1, toks.end());

        if (var.front() == 'P')
        {
            size_t idx = ParseIndex(var, 1);

            if (idx == INVALID_INDEX)
            {
                E_ERROR << "error parsing " << from << " line " << i;
                E_ERROR << "the matrix index cannot be determined from \"" << var << "\"";

                continue;
            }

            if (idx + 1 > P.size())
            {
                P.resize(idx + 1);
            }

            P[idx] = strings2mat(x, cv::Size(4, 3));

        }
        else if (boost::equals(var, "Tr"))
        {
            Tr = strings2mat(x, cv::Size(4, 3));
        }
        else
        {
            E_WARNING << "error parsing " << from << " line " << i;
            E_WARNING << "the line begins with an unknown variable name \"" << toks[0] << "\"";
        }
    }

    m_okay = true;
    E_INFO << "succesfully parsed " << from;
}

void KittiOdometryBuilder::WriteParams(cv::FileStorage& fs) const
{
    fs << "pose" << m_posePath;
}

bool KittiOdometryBuilder::ReadParams(const cv::FileNode& fn)
{
    fn["pose"] >> m_posePath;
    return true;
}

Parameterised::Options KittiOdometryBuilder::GetOptions(int flag)
{
    Options o("KITTI odometry dataset builder option");
    o.add_options()
        ("pose", po::value<String>(&m_posePath)->default_value(""), "Path to sequence's pose file.");

    return o;
}

bool KittiOdometryBuilder::BuildCamera(const Path& from, Camera::Map& cams, RectifiedStereo::Set& stereo) const
{
    Path calPath = from / "calib.txt";

    if (!fileExists(calPath))
    {
        E_ERROR << "missing calibration file " << calPath;
        return false;
    }

    Calib calib(calPath);

    if (calib.P.empty())
    {
        E_INFO << "no camera found, scanning aborted";
        return false;
    }

    E_INFO << "found " << calib.P.size() << " camera(s)";

    // KITTI odometry dataset should come with 2 rectified cameras
    assert(calib.P.size() == 2);

    const size_t ref = 0; // index of the referenced camera; KITTI uses the first camera (cam0)
    const cv::Mat K = calib.P[ref].rowRange(0, 3).colRange(0, 3);
    const cv::Mat K_inv = K.inv();

    cams.clear();

    for (size_t i = 0; i < calib.P.size(); i++)
    {
        E_INFO << "scanning files for camera " << i << "..";

        Camera::Own cam = Camera::New(i);

        // intrinsic parameters..
        cam->SetIntrinsics(PinholeModel::Own(new PinholeModel(K)));

        // extrinsic parameters..
        // only cameras other than the referenced one need this
        if (i != ref)
        {
            cam->GetExtrinsics().SetTransformMatrix(K_inv * calib.P[i]);
        }

        // search for images
        stringstream ss; ss << "image_" << i;
        String imageDirName = ss.str();
        Path imageDirPath = from / imageDirName;
        const String imageFileExt = ".png";

        cam->SetName(imageDirName);

        if (!dirExists(imageDirPath))
        {
            E_INFO << "skipped missing image folder " << imageDirPath;
            continue;
        }

        Paths imageFiles = enumerateFiles(imageDirPath, imageFileExt);
        Strings filelist;
        ImageStore& imageStore = cam->GetImageStore();

        BOOST_FOREACH(const Path& imageFile, imageFiles)
        {
            filelist.push_back(imageFile.filename().string());
        }

        imageStore.Create(imageDirPath, filelist);

        // try to retrieve the first frame
        PersistentImage frame = cam->GetImageStore()[0];
        cam->SetImageSize(frame.im.size());

        E_INFO << imageStore.GetItems() << " image(s) located for camera " << i;

        cam->Join(cams);
    }

    // create the left-right stereo pair
    stereo.insert(RectifiedStereo::Create(cams[0], cams[1]));

    return true;
}

//==[ KittiRawDataBuilder ]=====================================================

KittiRawDataBuilder::CalibCam2Cam::CalibCam2Cam(const Path& from)
: m_okay(false)
{
    ifstream in(from.string().c_str(), std::ios::in);

    if (!in.is_open())
    {
        E_ERROR << "error reading " << from;
        return;
    }

    String line;
    size_t i = 0;

    while (getline(in, line))
    {
        Strings toks = explode(line, ' ');
        i++;

        if (toks.empty()) continue; // skip an empty line

        if (toks[0].back() != ':')
        {
            E_WARNING << "error parsing " << from << " line " << i;
            E_WARNING << "the line begins with an invalid variable name \"" << toks[0] << "\"";

            continue;
        }

        // strip the colon char
        toks[0].pop_back();

        // VAR: x0 x1 x2 ...
        String var = toks.front();
        Strings x = Strings(toks.begin() + 1, toks.end());

        Strings varToks = explode(var, '_');

        if (varToks.size() < 2 || varToks[0].size() != 1) continue; // skip an irrelevant variable name

        bool rect  = varToks.size() > 1 && boost::equals(varToks[1], "rect");
        size_t idx = KittiOdometryBuilder::ParseIndex(varToks.back(), 0);

        if (idx == INVALID_INDEX)
        {
            continue;
        }

        if (idx + 1 > data.size())
        {
            data.resize(idx + 1);
        }

        char initial = varToks[0][0];

        if (rect)
        {
            switch (initial)
            {
            case 'S': data[idx].S_rect = strings2mat(x, cv::Size(1, 2)); break;
            case 'R': data[idx].R_rect = strings2mat(x, cv::Size(3, 3)); break;
            case 'P': data[idx].P_rect = strings2mat(x, cv::Size(4, 3)); break;
            }
        }
        else
        {
            switch (initial)
            {
            case 'S': data[idx].S = strings2mat(x, cv::Size(1, 2)); break;
            case 'K': data[idx].K = strings2mat(x, cv::Size(3, 3)); break;
            case 'D': data[idx].D = strings2mat(x, cv::Size(1, 5)); break;
            case 'R': data[idx].R = strings2mat(x, cv::Size(3, 3)); break;
            case 'T': data[idx].T = strings2mat(x, cv::Size(1, 3)); break;
            }
        }
    }

    m_okay = true;
    E_INFO << "succesfully parsed " << from.string();
}

KittiRawDataBuilder::CalibRigid::CalibRigid(const Path& from)
: m_okay(false)
{
    ifstream in(from.string().c_str(), std::ios::in);

    if (!in.is_open())
    {
        E_ERROR << "error reading " << from;
        return;
    }

    String line;
    size_t i = 0;

    while (getline(in, line))
    {
        Strings toks = explode(line, ' ');
        i++;

        if (toks.empty()) continue; // skip an empty line

        if (toks[0].back() != ':')
        {
            E_WARNING << "error parsing " << from << " line " << i;
            E_WARNING << "the line begins with an invalid variable name \"" << toks[0] << "\"";

            continue;
        }

        // strip the colon char
        toks[0].pop_back();

        // VAR: x0 x1 x2 ...
        String var = toks.front();
        Strings x = Strings(toks.begin() + 1, toks.end());

        switch (var[0])
        {
        case 'R': R = strings2mat(x, cv::Size(3, 3)); break;
        case 'T': T = strings2mat(x, cv::Size(3, 1)); break;
        }
    }

    if (tform.SetRotationMatrix(R) && tform.SetTranslation(T))
    {
        m_okay = true;
        E_INFO << "succesfully parsed " << from.string();
    }
}

void KittiRawDataBuilder::WriteParams(cv::FileStorage& fs) const
{
    fs << "rectified" << m_rectified;
    fs << "cam2cam" << m_cam2cam;
    fs << "imu2lid" << m_imu2lid;
    fs << "lid2cam" << m_lid2cam;
}

bool KittiRawDataBuilder::ReadParams(const cv::FileNode& fn)
{
    fn["rectified"] >> m_rectified;
    fn["cam2cam"] >> m_cam2cam;
    fn["imu2lid"] >> m_imu2lid;
    fn["lid2cam"] >> m_lid2cam;

    return true;
}

Parameterised::Options KittiRawDataBuilder::GetOptions(int flag)
{
    Options o("KITTI raw data builder options");
    o.add_options()
        ("synced",      po::bool_switch(&m_rectified)->default_value(false), "The sequence is synchronised and rectified.")
        ("cam-to-cam",  po::value<String>(&m_cam2cam)->default_value("calib_cam_to_cam.txt"),  "Path to the camera calibration file.")
        ("imu-to-velo", po::value<String>(&m_imu2lid)->default_value("calib_imu_to_velo.txt"), "Path to the IMU-to-LiDAR extrinsics file.")
        ("velo-to-cam", po::value<String>(&m_lid2cam)->default_value("calib_velo_to_cam.txt"), "Path to the LiDAR-to-camera extrinsics file.");

    return o;
}

bool KittiRawDataBuilder::BuildCamera(const Path& from, Camera::Map& cams, RectifiedStereo::Set& stereo) const
{
    CalibCam2Cam cam2cam(m_cam2cam);
    CalibRigid imu2lid(m_imu2lid);
    CalibRigid lid2cam(m_lid2cam);

    if (!cam2cam.IsOkay() || !imu2lid.IsOkay() || !lid2cam.IsOkay())
    {
        E_ERROR << "error reading configurations";
        return false;
    }

    EuclideanTransform imu2cam = imu2lid.tform >> lid2cam.tform;
    EuclideanTransform cam2imu = imu2cam.GetInverse();

    size_t ncams = cam2cam.data.size();
    bool rectified = m_rectified;

    if (!rectified) // automatically detect if the sequence is processed
    {
        Strings seqDirNameToks = explode(from.filename().string(), '_');
        if (boost::equals(seqDirNameToks.back(), "sync"))
        {
            E_WARNING << "synced + rectified sequence detected despite --synced is not set";
            rectified = true;
        }
    }

    if (ncams == 0)
    {
        E_INFO << "no camera found, scanning aborted";
        return false;
    }

    E_INFO << "found " << ncams << " camera(s)";

    assert(ncams == 4); // KITTI raw dataset should come with 4 cameras

    const size_t ref = 0; // index of the referenced camera; KITTI uses the first camera (idx=0)
    const cv::Mat K = cam2cam.data[ref].P_rect.rowRange(0, 3).colRange(0, 3);
    const cv::Mat K_inv = K.inv();

    cams.clear();

    for (size_t i = 0; i < ncams; i++)
    {
        E_INFO << "scanning files for camera " << i << "..";

        Camera::Own cam = Camera::New(i);

        // intrinsic parameters..
        if (rectified)
        {
            PinholeModel* intrinsics = new PinholeModel;

            intrinsics->SetCameraMatrix(cam2cam.data[i].K);
            cam->SetIntrinsics(PinholeModel::Own(intrinsics));
        }
        else
        {
            BouguetModel* intrinsics = new BouguetModel;
            
            intrinsics->SetCameraMatrix(cam2cam.data[i].K);
            intrinsics->SetDistortionCoeffs(cam2cam.data[i].D);
            cam->SetIntrinsics(BouguetModel::Own(intrinsics));
        }

        // extrinsic parameters..
        if (!rectified)
        {
            cam->GetExtrinsics().SetRotationMatrix(cam2cam.data[i].R);
            cam->GetExtrinsics().SetTranslation(cam2cam.data[i].T);
        }
        else if (i != ref)
        {
            cv::Mat RT = K_inv * cam2cam.data[i].P_rect;
            cam->GetExtrinsics().SetTransformMatrix(RT);

            // TODO: take R_rect into account
            // ...
        }

        // search for images
        stringstream ss; ss << "image_" << std::setfill('0') << std::setw(2) << i;
        const String imageDirName = ss.str();
        const Path   imageDirPath = from / imageDirName / "data";
        const String imageFileExt = ".png";

        cam->SetName(imageDirName);
        cam->SetModel(i < 2 ? "Point Grey Flea 2 (FL2-14S3M-C)" : "Point Grey Flea 2 (FL2-14S3C-C)");

        if (!dirExists(imageDirPath))
        {
            E_INFO << "skipped missing image folder " << imageDirPath.string();
            continue;
        }

        cam->GetImageStore().FromExistingFiles(imageDirPath, imageFileExt);

        cv::Mat S = rectified ? cam2cam.data[i].S_rect : cam2cam.data[i].S;
        cv::Size imageSize((int) S.at<float>(0), (int) S.at<float>(1));
        cam->SetImageSize(imageSize);

        E_INFO << cam->GetImageStore().GetItems() << " image(s) located for camera " << i;

        cam->Join(cams);
    }

    // create six possible stereo pairs
    stereo.insert(RectifiedStereo::Create(cams[0], cams[1])); // canonical pair 0
    stereo.insert(RectifiedStereo::Create(cams[2], cams[3])); // canonical pair 1
    stereo.insert(RectifiedStereo::Create(cams[0], cams[3])); // long-baseline cross-pair 0
    stereo.insert(RectifiedStereo::Create(cams[2], cams[1])); // long-baseline cross-pair 1
    stereo.insert(RectifiedStereo::Create(cams[2], cams[0])); // short-baseline cross-pair 0
    stereo.insert(RectifiedStereo::Create(cams[3], cams[1])); // short-baseline cross-pair 1

    return true;
}

//==[ EurocBuilder ]============================================================

bool EurocMavBuilder::ReadConfig(const Path& from, cv::FileStorage& to)
{
    std::ifstream fs(from.string());
    std::stringstream ss;

    if (!fs.is_open())
    {
        E_ERROR << "error opening file stream " << from;
        return false;
    }

    ss << "%YAML:1.0" << std::endl; // OpenCV needs this clue..
    ss << "---"       << std::endl;
    ss << fs.rdbuf(); // concatenate the body

    // parse the content of YML file in memory
    return to.open(ss.str(), cv::FileStorage::READ | cv::FileStorage::MEMORY);
}

String EurocMavBuilder::GetVehicleName(const Path& from) const
{
    Path confPath = from / "body.yaml";
    cv::FileStorage fs;

    if (!ReadConfig(confPath, fs))
    {
        E_ERROR << "error parsing body configuration from " << confPath;
        return "UNKNOWN";
    }

    return fs["comment"];
}

bool EurocMavBuilder::BuildCamera(const Path& from, Camera::Map& cams, RectifiedStereo::Set& stereo) const
{
    for (size_t k = 0; k < 2; k++)
    {
        std::stringstream ss;
        ss << "cam" << k;

        Path confPath = from / ss.str() / "sensor.YAML";
        cv::FileStorage fs;

        if (!ReadConfig(confPath, fs))
        {
            E_ERROR << "error reading camera configuration from " << confPath;
            return false;
        }

        String sensorType;
        fs["sensor_type"] >> sensorType;

        if (sensorType != "camera")
        {
            E_ERROR << "wrong sensor type \"" << sensorType << "\"";
            return false;
        }

        Camera::Own cam = Camera::New(k);
        String modelName;
        cv::Size imageSize;
        std::vector<double> extrinsics, intrinsics, distCoeffs;
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

        fs["comment"]      >> modelName;
        fs["resolution"]   >> imageSize;
        fs["T_BS"]["data"] >> extrinsics;
        fs["intrinsics"]   >> intrinsics;
        fs["distortion_coefficients"] >> distCoeffs;

        if (extrinsics.size() != 16)
        {
            E_ERROR << "error reading extrinsics, the parsed vector has a length of " << extrinsics.size() << " instead of 16";
            return false;
        }

        if (intrinsics.size() != 4)
        {
            E_ERROR << "error reading intrinsics, the parsed vector has a length of " << intrinsics.size() << " instead of 4";
            return false;
        }

        if (distCoeffs.size() != 4)
        {
            E_ERROR << "error reading distortion coefficients, the parsed vector has a length of " << distCoeffs.size() << " instead of 4";
            return false;
        }

        E_INFO << cv::Mat(distCoeffs).total();


        cameraMatrix.at<double>(0, 0) = intrinsics[0];
        cameraMatrix.at<double>(1, 1) = intrinsics[1];
        cameraMatrix.at<double>(0, 2) = intrinsics[2];
        cameraMatrix.at<double>(1, 2) = intrinsics[3];

        cam->SetName(ss.str());
        cam->SetModel(modelName);
        cam->SetImageSize(imageSize);
        cam->GetExtrinsics().SetTransformMatrix(cv::Mat(extrinsics).reshape(1, 4));
        cam->SetIntrinsics(BouguetModel::Own(new BouguetModel(cameraMatrix, cv::Mat(distCoeffs))));

        const Path imageDirPath = from / ss.str() / "data";
        const String imageFileExt = ".png";

        cam->GetImageStore().FromExistingFiles(imageDirPath, imageFileExt);
        E_INFO << cam->GetImageStore().GetItems() << " image(s) located for camera " << k;

        cam->Join(cams);
    }

    return true;
}

//==[ SeqBuilderFactory ]=======================================================

SeqBuilderFactory::SeqBuilderFactory()
{
    Register<KittiOdometryBuilder>("KITTI_ODOMETRY");
    Register<KittiRawDataBuilder> ("KITTI_RAW");
    Register<EurocMavBuilder>     ("EUROC");
}

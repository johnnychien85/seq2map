#include <iomanip>
#include <boost/algorithm/string/predicate.hpp>
#include "scanner.hpp"

using namespace std;

size_t KittiOdometryScanner::ParseIndex(const String& varname, size_t offset)
{
    size_t idx;
    std::istringstream iss(String(varname.begin() + offset, varname.end()));
    iss >> idx;

    // check if all the entries of the variable name have benn
    //  succesfully converted to an integer
    return iss.eof() ? idx : INVALID_INDEX;
}

KittiOdometryScanner::Calib::Calib(const Path& calPath)
{
    ifstream in(calPath.string(), std::ios::in);

    if (!in.is_open())
    {
        E_ERROR << "error reading \"" << calPath.string() << "\"";
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
            E_WARNING << "error parsing \"" << calPath.string() << "\" line " << i;
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
                E_ERROR << "error parsing \"" << calPath.string() << "\" line " << i;
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
            E_WARNING << "error parsing \"" << calPath.string() << "\" line " << i;
            E_WARNING << "the line begins with an unknown variable name \"" << toks[0] << "\"";
        }
    }

    E_INFO << "succesfully parsed \"" << calPath.string() << "\"";
}

bool KittiOdometryScanner::Scan(const Path& seqPath, const Path& calPath, const Path& motPath, Sequence& seq)
{
    Calib calib(calPath);

    if (calib.P.empty())
    {
        E_INFO << "no camera found, scanning aborted";
        return false;
    }

    E_INFO << "found " << calib.P.size() << " camera(s)";
    seq.Clear();

    Cameras& cams = seq.GetCameras();
    const size_t ref = 0; // index of the referenced camera; KITTI uses the first camera (idx=0)
    const cv::Mat K = calib.P[ref].rowRange(0, 3).colRange(0, 3);
    const cv::Mat K_inv = K.inv();

    cams.resize(calib.P.size());

    for (size_t i = 0; i < calib.P.size(); i++)
    {
        E_INFO << "scanning files for camera " << i << "..";

        Camera& cam = cams.at(i);

        // intrinsic parameters..
        BouguetModel::Ptr intrinsics = BouguetModel::Ptr(new BouguetModel());
        intrinsics->SetCameraMatrix(K);

        cam.SetIntrinsics(intrinsics);

        // extrinsic parameters..
        //  only cameras other than the referenced one need
        if (i != ref)
        {
            cv::Mat RT = K_inv * calib.P[i];
            cam.GetExtrinsics().SetTransformMatrix(RT);
        }

        // search for images
        stringstream ss; ss << "image_" << i;
        String imageDirName = ss.str();
        Path imageDirPath = seqPath / imageDirName;
        const String imageFileExt = ".png";
        
        cam.SetName(imageDirName);

        if (!dirExists(imageDirPath))
        {
            E_INFO << "skipped because image folder \"" << imageDirPath.string() << "\" not found";
            continue;
        }

        Paths imageFiles = enumerateFiles(imageDirPath, imageFileExt);
        Camera::ImageStorage& imageStore = cam.GetImageStorage();

        imageStore.SetRootPath(imageDirPath);
        imageStore.Allocate(imageFiles.size());

        BOOST_FOREACH(const Path& imageFile, imageFiles)
        {
            imageStore.Add(imageFile.filename().string());
        }

        // try to retrieve the first frame
        Frame frame = cam[0];
        cam.SetImageSize(frame.im.size());

        E_INFO << imageStore.GetSize() << " image(s) located for camera " << i;
    }

    return true;
}


KittiRawDataScanner::Calib::Calib(const Path& calPath)
{
    ifstream in(calPath.string(), std::ios::in);

    if (!in.is_open())
    {
        E_ERROR << "error reading \"" << calPath.string() << "\"";
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
            E_WARNING << "error parsing \"" << calPath.string() << "\" line " << i;
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
        size_t idx = KittiOdometryScanner::ParseIndex(varToks.back(), 0);

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

    E_INFO << "succesfully parsed \"" << calPath.string() << "\"";
}

bool KittiRawDataScanner::Scan(const Path& seqPath, const Path& calPath, const Path& motPath, Sequence& seq)
{
    Calib calib(calPath);
    size_t ncams = calib.data.size();

    Strings seqNameToks = explode(seqPath.filename().string(), '_');
    bool rectified = boost::equals(seqNameToks.back(), "sync");

    if (ncams == 0)
    {
        E_INFO << "no camera found, scanning aborted";
        return false;
    }

    E_INFO << "found " << ncams << " camera(s)";
    seq.Clear();

    Cameras& cams = seq.GetCameras();
    const size_t ref = 0; // index of the referenced camera; KITTI uses the first camera (idx=0)
    const cv::Mat K = calib.data[ref].P_rect.rowRange(0, 3).colRange(0, 3);
    const cv::Mat K_inv = K.inv();

    cams.resize(ncams);

    for (size_t i = 0; i < ncams; i++)
    {
        E_INFO << "scanning files for camera " << i << "..";

        Camera& cam = cams.at(i);

        // intrinsic parameters..
        BouguetModel::Ptr intrinsics = BouguetModel::Ptr(new BouguetModel());

        if (!rectified)
        {
            intrinsics->SetCameraMatrix(calib.data[i].K);
            intrinsics->SetDistCoeffs(calib.data[i].D);
        }
        else
        {
            intrinsics->SetCameraMatrix(K);
        }
        
        cam.SetIntrinsics(intrinsics);

        // extrinsic parameters..
        if (!rectified)
        {
            cam.GetExtrinsics().SetRotationMatrix(calib.data[i].R);
            cam.GetExtrinsics().SetTranslation(calib.data[i].T);
        }
        else if (i != ref)
        {
            cv::Mat RT = K_inv * calib.data[i].P_rect;
            cam.GetExtrinsics().SetTransformMatrix(RT);

            // TODO: take R_rect into account
            // ...
        }

        // search for images
        stringstream ss; ss << "image_" << std::setfill('0') << std::setw(2) << i;
        String imageDirName = ss.str();
        Path imageDirPath = seqPath / imageDirName / "data";
        const String imageFileExt = ".png";

        cam.SetName(imageDirName);

        if (!dirExists(imageDirPath))
        {
            E_INFO << "skipped because image folder \"" << imageDirPath.string() << "\" not found";
            continue;
        }

        Paths imageFiles = enumerateFiles(imageDirPath, imageFileExt);
        Camera::ImageStorage& imageStore = cam.GetImageStorage();

        imageStore.SetRootPath(imageDirPath);
        imageStore.Allocate(imageFiles.size());

        BOOST_FOREACH(const Path& imageFile, imageFiles)
        {
            imageStore.Add(imageFile.filename().string());
        }

        cv::Mat S = rectified ? calib.data[i].S_rect : calib.data[i].S;
        cv::Size imageSize((int) S.at<float>(0), (int) S.at<float>(1));
        cam.SetImageSize(imageSize);

        E_INFO << imageStore.GetSize() << " image(s) located for camera " << i;
    }

    return true;
}

ScannerFactory::ScannerFactory()
{
    Register<KittiOdometryScanner>("KITTI_ODOMETRY");
    Register<KittiRawDataScanner> ("KITTI_RAW");
}

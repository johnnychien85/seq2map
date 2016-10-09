#include <fstream>
#include <iomanip>
#include <calibn/calibprofile.hpp>
#include <calibn/calibgraph.hpp>
#include <calibn/helpers.hpp>

bool CalibProfile::Create(const Path& imageRootPath, const String& imageList, size_t cams, size_t imgs, const String& def)
{
    if (!m_pattern.FromString(def))
    {
        E_ERROR << "error parsing calibration pattern definitions";
        return false;
    }

    E_INFO << "calibration pattern parsed"; // << m_pattern.ToString();
    E_INFO << "scanning image files in " << imageRootPath;

    if (!m_imageFiles.FromPattern(imageList, cams, imgs) &&
        !m_imageFiles.FromFile   (imageList, cams, imgs))
    {
        E_ERROR << "cannot decide image file list from \"" << imageList << "\"";
        return false;
    }

    if (m_imageFiles.IsOkay())
    {
        E_ERROR << "profile creation failed due to invalid image file list";
        return false;
    }

    // set the root path; we need this to perform file check
    m_imageFiles.SetRoot(imageRootPath);

    if (!m_imageFiles.CheckImageFiles())
    {
        E_ERROR << "profile creation failed due to a missing file";
        return false;
    }

    m_numCameras = cams;
    m_numImages  = imgs;
    m_cams.clear();

    for (size_t cam = 0; cam < cams; cam++)
    {
        m_cams.push_back(CalibCam(cam));
    }
}

bool CalibProfile::Build(bool adaptive, bool normalise, bool fastcheck, size_t subpx)
{
    if (m_cams.empty() || !m_imageFiles.IsOkay() || !m_pattern.IsOkay())
    {
        E_ERROR << "profile not initialised properly";
        return false;
    }

    E_INFO << "start building calibration data for " << m_numCameras << " camera(s) from " << m_numImages << " image(s)";
    size_t newImgIdx = 0;

    // set up chessboard detection parameters
    CalibData::detectionFlags = 0;
    if (adaptive)  CalibData::detectionFlags += CV_CALIB_CB_ADAPTIVE_THRESH;
    if (normalise) CalibData::detectionFlags += CV_CALIB_CB_NORMALIZE_IMAGE;
    if (fastcheck) CalibData::detectionFlags += CV_CALIB_CB_FAST_CHECK;
    
    CalibData::subPixelWinSize = cv::Size(subpx, subpx);

    for (size_t img = 0; img < m_numImages; img++)
    {
        std::cout << "creating calibration data from image #" << img << "..";

        CalibDataSet dataset(m_numCameras);
        size_t hits = 0;

        for (size_t cam = 0; cam < m_numCameras; cam++)
        {
            dataset[cam].FromImage(newImgIdx, m_imageFiles(cam, img), m_pattern);

            if (dataset[cam].IsOkay())
            {
                hits++;
                std::cout << ".." << cam;
            }
            else
            {
                std::cout << "..x";
            }
        }

        if (hits == 0)
        {
            std::cout << "..REJECTED" << std::endl;
            continue;
        }

        for (size_t cam = 0; cam < m_numCameras; cam++)
        {
            if (!m_cams[cam].AddCalibData(dataset[cam]))
            {
                E_ERROR << "error adding calibration data of image #" << img << " to cam #" << cam;
                return false;
            }
        }

        newImgIdx++;
        std::cout << "..DONE" << std::endl;
    }

    m_numImages = newImgIdx;

    return true;
}

bool CalibProfile::Calibrate(bool pairOptim)
{
    if (m_numCameras == 1) // monocular calibration
    {
        if (m_cams[0].Calibrate() < 0)
        {
            return false;
        }

        m_params = CalibBundleParams(1, 0);
        m_params.Intrinsics.push_back(m_cams[0].GetIntrinsics());
        m_params.ImagePoses = m_cams[0].GetImagePoses();

        return true;
    }

    //CalibReport report(m_reportPath, "");
    m_graph.Create(m_cams, 0);

    if (!m_graph.IsOkay()) // something wrong with the connectivity?
    {
        E_ERROR << "calibration not possible due to disconnected node(s) in the calibration graph";
        return false;
    }
    
    if (!m_graph.Initialise(pairOptim))
    {
        E_ERROR << "error establishing initial solutions";
        return false;
    }
}

bool CalibProfile::Optimise(size_t iter, double eps, size_t threads)
{
    if (!m_graph.IsOkay()) return false;
    cv::TermCriteria term(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, iter, eps);

    return m_graph.Optimise(term, threads);
}

bool CalibProfile::Store(Path& path) const
{
    try
    {
        cv::FileStorage f(path.string(), cv::FileStorage::WRITE);

        f << "calibn" << "{:";
        f << "numCameras" << (int) m_numCameras;
        f << "numImages"  << (int) m_numImages;
        f << "pattern"    << "{:" << "size" << m_pattern.PatternSize << "metric" << m_pattern.PatternMetric << "}";

        f << "cams" << "[:";
        for (size_t cam = 0; cam < m_cams.size(); cam++)
        {
            if (!m_cams[cam].Write(f)) return false;
        }
        f << "]";
        
        f << "params";
        if (!m_params.Store(f))
        {
            return false;
        }
        f << "}";
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error storing profile to " << path;
        E_ERROR << ex.what();

        return false;
    }

    E_INFO << "profile saved to " << path.string();
    return true;
}

bool CalibProfile::Store(Path& path) const
{
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    return Store(fs);
}

bool CalibProfile::Restore(const Path& path)
{
    try
    {
        cv::FileStorage fs(path.str(), cv::FileStorage::READ);

        if (!fs.isOpened()) // can't open file?
        {
            return false;
        }

        cv::FileNode fn = fs["calibn"];

        int x;
        fn["numCameras"] >> x; m_numCameras = (size_t) x;
        fn["numImages"]  >> x; m_numImages  = (size_t) x;
        fn["pattern"]["size"]   >> m_pattern.PatternSize;
        fn["pattern"]["metric"] >> m_pattern.PatternMetric;

        cv::FileNode camNode = fn["cams"];

        for (cv::FileNodeIterator camItr = camNode.begin();
             camItr != camNode.end(); camItr++)
        {
            CalibCam cam(*camItr);

            if (cam.GetSize() != m_numImages)
            {
                E_ERROR << "error loading profile, the number of loaded images of camera #" << cam.GetIndex() << " is not consistent with the definition";
                E_ERROR << "expected " << m_numImages << " image(s), while loaded " << cam.GetSize();

                return false;
            }

            m_cams.push_back(cam);
        }

        if (m_cams.size() != m_numCameras)
        {
            E_ERROR << "error loading profile, the number of loaded cameras is not consistent with the definition";
            E_ERROR << "expected " << m_numCameras << " camera(s), while loaded " << m_cams.size();

            return false;
        }

        // TODO: sort cameras according to their indices
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error restoring profile from " << path;
        E_ERROR << ex.what();

        return false;
    }

    return true;
}

bool CalibProfile::ExportToCalibDir(const string& path)
{
    Size imageSize = m_cams[0].GetData(0).GetImageSize();
    m_params.InitUndistortRectifyMaps(0, 1, imageSize);

    mkdir(path);

    string mfilePath = fullPath(path, "cam.m");
    ofstream fk(mfilePath);

    fk << "% Projection matrix / matrices" << endl;
    fk << "P = zeros(3,4," << m_cams.size() << ");" << endl;
    fk << "R = zeros(3,3," << m_cams.size() << ");" << endl;

    for (size_t k = 0; k < m_cams.size(); k++)
    {
        stringstream ss; ss << "c" << k << ".xml";
        FileStorage fc(fullPath(path, ss.str()), FileStorage::WRITE);

        Mat R = m_params.GetRectR(k);
        Mat P = m_params.GetRectP(k);
        
        fc << "imageSize" << imageSize;
        fc << "K" << m_params.Intrinsics[k].CameraMatrix;
        fc << "D" << m_params.Intrinsics[k].DistortionMatrix;
        fc << "R" << R;
        fc << "P" << P;

        fk << "P(:,:," << (k + 1) << ") = [";
        for (size_t i = 0; i < P.rows; i++)
        {
            bool lastRow = i == P.rows - 1;
            for (size_t j = 0; j < P.cols; j++)
            {
                bool lastCol = j == P.cols - 1;
                fk << fixed << setprecision(5) << P.at<double>(i,j) << (lastCol ? "" : ", ");
            }
            if (!lastRow) fk << "; ";
        }
        fk << "];" << endl;

        fk << "R(:,:," << (k + 1) << ") = [";
        for (size_t i = 0; i < R.rows; i++)
        {
            bool lastRow = i == R.rows - 1;
            for (size_t j = 0; j < R.cols; j++)
            {
                bool lastCol = j == R.cols - 1;
                fk << fixed << setprecision(5) << R.at<double>(i,j) << (lastCol ? "" : ", ");
            }
            if (!lastRow) fk << "; ";
        }
        fk << "];" << endl;
    }

    fk << endl;
    fk << "% Image pose(s)" << endl;
    fk << "obj = [" << endl;
    BOOST_FOREACH (const CalibExtrinsics& imagePose, m_params.ImagePoses)
    {
        vector<double> e = imagePose.ToVector();

        fk << " ";
        for (size_t i = 0; i < e.size(); i++)
        {
            fk << fixed << setprecision(5) << e[i] << (i == e.size() - 1 ? "" : ", ");
        }
        fk << ";" << endl;
    }
    fk << "];" << endl;

    return true;
}

String CalibProfile::Summary() const
{
    return "??";
}

bool ImageFileList::FromPattern(const String& pattern, size_t cams, size_t imgs)
{
    if (cams == 0 || imgs == 0 || !CheckPattern(pattern)) return false;

    m_list.clear();
    m_list.resize(cams);
    
    for (size_t cam = 0; cam < cams; cam++)
    {
        m_list[cam].resize(imgs);

        String imagePattern = pattern;
        replace(imagePattern, "%1", toString((int)cam));

        for (size_t img = 0; img < imgs; img++)
        {
            String filename = imagePattern;
            bool   success  = replace(filename, "%2", toString((int)img));

            if (!success)
            {
                m_list.clear(); // this should make no difference
                E_ERROR << "error replacing image index in \"" << filename << "\"";

                return false;
            }

            m_list[cam][img] = filename;
        }
    }

    return true;
}

bool ImageFileList::FromFile(const Path& path, size_t cams, size_t imgs)
{
    std::ifstream f(path.string());

    if (!f.is_open())
    {
        E_ERROR << "error opening list file " << path;
        return false;
    }

    m_list.clear();
    m_list.resize(cams);

    for (size_t cam = 0; cam < cams; cam++)
    {
        m_list[cam].resize(imgs);

        for (size_t img = 0; img < imgs; img++)
        {
            String line;

            if (!getline(f, line))
            {
                m_list.clear(); // this should make no difference
                            
                E_ERROR << "error parsing list file " << path;
                E_ERROR << "reached EOF while trying to read image " << img << " of camera " << cam;

                return false;
            }

            m_list[cam][img] = line;
        }
    }

    return true;
}

bool ImageFileList::CheckImageFiles()
{
    if (!IsOkay())
    {
        E_ERROR << "image file check aborted due to empty file list";
        return false;
    }

    size_t cams = m_list.size();
    size_t imgs = m_list.begin()->size();

    // consistency check
    for (size_t cam = 1 /* no need to check cam #0 */; cam < cams; cam++)
    {
        if (m_list[cam].size() != imgs)
        {
            E_ERROR << "image file check aborted due to inconsistent number of image";
            return false;
        }
    }

    std::cout << "Checking image file.." << std::endl;
    
    for (size_t img = 0; img < imgs; img++)
    {
        for (size_t cam = 0; cam < cams; cam++)
        {
            Path imagePath = this->operator(cam, img);
            cv::Mat im = cv::imread(imagePath);

            if (im.empty())
            {
                std::cout << "..x (cam" << cam << ")..FAILED" << std::endl;
                E_ERROR << "error reading image from " << imagePath;

                return false;
            }
        }
        std::cout << ".." << img;
    }
    std::cout << "..DONE" << std::endl;

    return true;
}

bool ImageFileList::CheckPattern(const String& pattern)
{
    String test = m_pattern;
    return replace(test, "%1", "0") && replace(test, "%2", "0");
}

/*
CalibProfile::CalibProfile(const string& profilePath, const string& reportPath)
: m_params(CalibBundleParams::NullParams), m_reportPath(reportPath)
{
    if (Load(m_path = profilePath))
    {
        return;
    }

    m_cams.clear();
    m_numCameras = 0;
    m_numImages = 0;
}
*/

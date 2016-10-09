#include <calibn/calibcam.hpp>
#include <calibn/helpers.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

size_t CalibIntrinsics::NumParameters = 9; // fc,fu,uc,vc,k1,k2,p1,p2,k3
size_t CalibExtrinsics::NumParameters = 6; // rx,ry,rz,tx,ty,tz
CalibBundleParams CalibBundleParams::NullParams(0,0);
TermCriteria Calibratable::OptimTermCriteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-3);

std::vector<double> CalibIntrinsics::ToVector() const
{
    Mat K = CameraMatrix, D = DistortionMatrix;
    vector<double> vec(NumParameters);

    assert(K.size() == Size(3,3) && D.size() == Size(5,1));

    size_t i = 0;
    vec[i++] = K.at<double>(0,0); // fu
    vec[i++] = K.at<double>(1,1); // fv
    vec[i++] = K.at<double>(0,2); // uc
    vec[i++] = K.at<double>(1,2); // uv
    vec[i++] = D.at<double>(0,0); // kappa1
    vec[i++] = D.at<double>(0,1); // kappa2
    vec[i++] = D.at<double>(0,2); // p1
    vec[i++] = D.at<double>(0,3); // p2
    vec[i++] = D.at<double>(0,4); // kappa3

    return vec;
}

bool CalibIntrinsics::FromVector(const std::vector<double>& vec)
{
    if (vec.size() != NumParameters)
    {
        return false;
    }

    Mat& K = CameraMatrix;
    Mat& D = DistortionMatrix;

    assert(K.size() == Size(3,3) && D.size() == Size(5,1));

    size_t i = 0;
    K.at<double>(0,0) = vec[i++]; // fu
    K.at<double>(1,1) = vec[i++]; // fv
    K.at<double>(0,2) = vec[i++]; // uc
    K.at<double>(1,2) = vec[i++]; // vu
    D.at<double>(0,0) = vec[i++]; // kappa1
    D.at<double>(0,1) = vec[i++]; // kappa2
    D.at<double>(0,2) = vec[i++]; // p1
    D.at<double>(0,3) = vec[i++]; // p2
    D.at<double>(0,4) = vec[i++]; // kappa3

    return true;
}

bool CalibIntrinsics::Write(FileStorage& fn) const
{
    vector<double> vec = ToVector();
    fn << "focalLength"	<< vector<double>(vec.begin() + 0, vec.begin() + 2);
    fn << "imageCentre"	<< vector<double>(vec.begin() + 2, vec.begin() + 4);
    fn << "distCoeffs"	<< vector<double>(vec.begin() + 4, vec.end());

    return true;
}

std::vector<double> CalibExtrinsics::ToVector() const
{
    Mat R = Rotation;
    Mat t = Translation;
    vector<double> vec(NumParameters);

    assert(R.size() == Size(3,3) && t.size() == Size(1,3));

    Mat a; Rodrigues(R,a); // to angle-axis representation

    size_t i = 0;
    vec[i++] = a.at<double>(0,0); // ax
    vec[i++] = a.at<double>(1,0); // ay
    vec[i++] = a.at<double>(2,0); // az
    vec[i++] = t.at<double>(0,0); // tx
    vec[i++] = t.at<double>(1,0); // ty
    vec[i++] = t.at<double>(2,0); // tz

    return vec;
}

bool CalibExtrinsics::FromVector(const vector<double>& vec)
{
    if (vec.size() != NumParameters)
    {
        return false;
    }

    Mat& R = Rotation;
    Mat& t = Translation;
    Mat a = Mat(3, 1, CV_64F);

    assert(R.size() == Size(3,3) && t.size() == Size(1,3));

    size_t i = 0;
    a.at<double>(0,0) = vec[i++]; // ax
    a.at<double>(1,0) = vec[i++]; // ay
    a.at<double>(2,0) = vec[i++]; // az
    t.at<double>(0,0) = vec[i++]; // tx
    t.at<double>(1,0) = vec[i++]; // ty
    t.at<double>(2,0) = vec[i++]; // tz

    Rodrigues(a,R); // to matrix representation

    return true;
}

Mat CalibExtrinsics::GetMatrix4x4() const
{
    Mat M = Mat::eye(4, 4, Rotation.type());
    Rotation.copyTo(M.rowRange(0,3).colRange(0,3));
    Translation.copyTo(M.rowRange(0,3).colRange(3,4));

    return M;
}

bool CalibExtrinsics::Write(FileStorage& fn) const
{
    vector<double> vec = ToVector();
    fn << "rotation"	<< vector<double>(vec.begin() + 0, vec.begin() + 3);
    fn << "translation"	<< vector<double>(vec.begin() + 3, vec.begin() + 6);

    return true;
}

void CalibBundleParams::Create(size_t numCams, size_t refCamIdx)
{
    m_numCams = numCams;
    m_refCamIdx = refCamIdx;
}

std::vector<double> CalibBundleParams::ToVector() const
{
    assert(m_numCams == Intrinsics.size() && m_numCams == Extrinsics.size());

    size_t m = Intrinsics.size();
    size_t n = Extrinsics.size() - 1;
    size_t k = ImagePoses.size();
    size_t d = m * CalibIntrinsics::NumParameters + (n + k) * CalibExtrinsics::NumParameters;
    std::vector<double> vec;

    for (CalibIntrinsicsList::const_iterator itr = Intrinsics.begin(); itr != Intrinsics.end(); itr++)
    {
        append(vec, itr->ToVector());
    }

    CalibExtrinsicsList extrinsicsPoses; // extrinsics + image poses
    extrinsicsPoses.reserve((n + k) * CalibExtrinsics::NumParameters);

    // add all extrinsics excluding the reference camera's one because it's always identity
    append(extrinsicsPoses, Extrinsics);
    extrinsicsPoses.erase(extrinsicsPoses.begin() + m_refCamIdx);

    // add all image poses afterwards
    append(extrinsicsPoses, ImagePoses);

    // append all extrinsics (camera extrinsics + image extrinsics)
    for (CalibExtrinsicsList::const_iterator itr = extrinsicsPoses.begin(); itr != extrinsicsPoses.end(); itr++)
    {
        append(vec, itr->ToVector());
    }

    assert (vec.size() == d); // size check

    return vec;
}

bool CalibBundleParams::FromVector(const vector<double>& vec)
{
    size_t m = m_numCams;
    size_t n = m - 1;
    size_t k = (vec.size() - m * CalibIntrinsics::NumParameters - n * CalibExtrinsics::NumParameters) / CalibExtrinsics::NumParameters;
    size_t d = m * CalibIntrinsics::NumParameters + (n + k) * CalibExtrinsics::NumParameters;

    assert (vec.size() == d); // size check

    Intrinsics = CalibIntrinsicsList(m);
    CalibExtrinsicsList extrinsicsPoses = CalibExtrinsicsList(n+k);

    vector<double>::const_iterator v = vec.begin();

    for (CalibIntrinsicsList::iterator itr = Intrinsics.begin(); itr != Intrinsics.end(); itr++)
    {
        if (!itr->FromVector(vector<double>(v, v + itr->GetSize()))) return false;
        v += itr->GetSize();
    }

    for (CalibExtrinsicsList::iterator itr = extrinsicsPoses.begin(); itr != extrinsicsPoses.end(); itr++)
    {
        if (!itr->FromVector(vector<double>(v, v + itr->GetSize()))) return false;
        v += itr->GetSize();
    }

    // split up extrinsics and image poses
    Extrinsics = CalibExtrinsicsList(extrinsicsPoses.begin(), extrinsicsPoses.begin() + n);
    ImagePoses = CalibExtrinsicsList(extrinsicsPoses.begin() + n, extrinsicsPoses.end());

    // append an identity extrinsics of the reference camera
    Extrinsics.insert(Extrinsics.begin() + m_refCamIdx, CalibExtrinsics());

    return true;
}

ImagePointList CalibBundleParams::Project(const ObjectPointList& pts3d, size_t camIdx, size_t imgIdx) const
{
    ImagePointList y;
    CalibExtrinsics E = ImagePoses[imgIdx].Concatenate(Extrinsics[camIdx]);
    projectPoints(pts3d, E.Rotation, E.Translation, Intrinsics[camIdx].CameraMatrix, Intrinsics[camIdx].DistortionMatrix.t(), y);

    return y;
}

ImagePointList CalibBundleParams::ProjectUndistort(const ObjectPointList& pts3d, size_t camIdx, size_t imgIdx) const
{
    return UndistortPoints(Project(pts3d, camIdx, imgIdx), camIdx);
    /*
    assert(m_rectOkay);

    Mat P = m_rectP[camIdx] * ImagePoses[imgIdx].GetMatrix4x4();
    Mat x, y;

    convertPointsToHomogeneous(pts3d, x);
    convertPointsFromHomogeneous(P * x, y);

    return y;
    */
}

ImagePointList CalibBundleParams::UndistortPoints(const ImagePointList& pts2d, size_t camIdx) const
{
    assert(m_rectOkay);
    ImagePointList udst2d;

    undistortPoints(pts2d, udst2d, Intrinsics[camIdx].CameraMatrix, Intrinsics[camIdx].DistortionMatrix, m_rectR[camIdx], m_rectP[camIdx]);

    return udst2d;
}

bool CalibBundleParams::Store(cv::FileStorage& fn) const
{
    size_t m = Intrinsics.size();
    size_t n = Extrinsics.size();
    size_t k = ImagePoses.size();

    if (m != n)
    {
        return false;
    }

    fn << "{:";
    fn << "cams" << "[:";
    for (size_t cam = 0; cam < m; cam++)
    {
        fn << "{:" << "index" << (int) cam; 
        if (!Intrinsics[cam].Write(fn) || !Extrinsics[cam].Write(fn))
        {
            return false;
        }
        fn << "}";
    }
    fn << "]";

    fn << "imagePoses" << "[:";
    for (size_t img = 0; img < k; img++)
    {
        fn << "{:" << "index" << (int) img;
        if (!ImagePoses[img].Write(fn))
        {
            return false;
        }
        fn << "}";
    }
    fn << "]";
    fn << "}";

    return true;
}

void CalibBundleParams::InitUndistortRectifyMaps(size_t cam0, size_t cam1, const Size& imageSize)
{
    assert(m_numCams > 0);

    if (m_numCams == 1) // monocular case
    {
        InitMonocularUndistortMap(imageSize);
        return;
    }

    // for binocular and multinocular cases
    InitCollinearUndistortMaps(cam0, cam1, imageSize);
}

void CalibBundleParams::InitMonocularUndistortMap(const Size& imageSize)
{
    assert(m_numCams == 1);

    m_rectMaps1 = vector<Mat>(1);
    m_rectMaps2 = vector<Mat>(1);
    m_rectR = vector<Mat>(1);
    m_rectP = vector<Mat>(1);

    m_rectR[0] = Mat::eye(3, 3, CV_64F);
    m_rectP[0] = Mat::zeros(3, 4, CV_64F);

    Mat K = m_rectP[0].colRange(0,3).rowRange(0,3);
    double alpha = 0;

    getOptimalNewCameraMatrix(Intrinsics[0].CameraMatrix, Intrinsics[0].DistortionMatrix, imageSize, alpha).copyTo(K);

    initUndistortRectifyMap(
            Intrinsics[0].CameraMatrix, // camera matrix
            Intrinsics[0].DistortionMatrix, // distortion coeffs
            Mat(),						// rotation (identity for monocular case)
            K,                          // new camera matrix
            imageSize,					// new image size (unchanged)
            CV_16SC2,					// store x and y coordinates
            m_rectMaps1[0],				// ..in one 2-channel matrix
            m_rectMaps2[0]);			// ..interpolation coefficients?

    m_rectOkay = true;
}

void CalibBundleParams::InitCollinearUndistortMaps(size_t cam0, size_t cam1, const Size& imageSize)
{
    m_rectMaps1 = vector<Mat>(m_numCams);
    m_rectMaps2 = vector<Mat>(m_numCams);
    m_rectR = vector<Mat>(m_numCams);
    m_rectP = vector<Mat>(m_numCams);

    assert(m_numCams > 1);
    
    CalibExtrinsics E01 = Extrinsics[cam0].GetInverse().Concatenate(Extrinsics[cam1]); // transformation from cam0 to cam1
    Mat Q;
    Mat K0 = Intrinsics[cam0].CameraMatrix, D0 = Intrinsics[cam0].DistortionMatrix;
    Mat K1 = Intrinsics[cam1].CameraMatrix, D1 = Intrinsics[cam1].DistortionMatrix;

    double alpha = 0;
    int flags = CALIB_ZERO_DISPARITY;

    // solve stereo rectification first
    stereoRectify(
        K0, D0, K1, D1,
        imageSize, E01.Rotation, E01.Translation,
        m_rectR[cam0], m_rectR[cam1], m_rectP[cam0], m_rectP[cam1], Q, flags, alpha);

    //initUndistortRectifyMap(K0, D0, m_rectR[cam0], m_rectP[cam0], imageSize, CV_16SC2, m_rectMaps1[cam0], m_rectMaps2[cam0]);
    //initUndistortRectifyMap(K1, D1, m_rectR[cam1], m_rectP[cam1], imageSize, CV_16SC2, m_rectMaps1[cam1], m_rectMaps2[cam1]);

    for (size_t cam2 = 0; cam2 < m_numCams; cam2++)
    {
        Mat K2 = Intrinsics[cam2].CameraMatrix, D2 = Intrinsics[cam2].DistortionMatrix;

        if (cam2 != cam0 && cam2 != cam1) // do 3-view collinear rectification for each camera excepting cam0 and cam1
        {
            CalibExtrinsics E02 = Extrinsics[cam0].GetInverse().Concatenate(Extrinsics[cam2]); // transformation from cam0 to cam1

            rectify3Collinear(
                K0, D0, K1, D1, K2, D2,
                Mat(), Mat(), imageSize,
                E01.Rotation, E01.Translation, E02.Rotation, E02.Translation,
                m_rectR[cam0], m_rectR[cam1], m_rectR[cam2],
                m_rectP[cam0], m_rectP[cam1], m_rectP[cam2], Q, alpha, imageSize, NULL, NULL, flags);
        }

        initUndistortRectifyMap(K2, D2, m_rectR[cam2], m_rectP[cam2], imageSize, CV_16SC2, m_rectMaps1[cam2], m_rectMaps2[cam2]);
    }

    m_rectOkay = true;
}

Mat CalibBundleParams::Rectify(size_t camIdx, const Mat& im) const
{
    assert(RectifyInitialised() && !m_rectMaps1[camIdx].empty() && !m_rectMaps2[camIdx].empty());

    Mat imRect;
    remap(im, imRect, m_rectMaps1[camIdx], m_rectMaps2[camIdx], INTER_CUBIC, BORDER_CONSTANT, Scalar());

    return imRect;
}

CalibCam::CalibCam(const FileNode& fn)
{
    try
    {
        int index;
        fn["index"] >> index;

        SetIndex((size_t) index);

        FileNode dataNode = fn["data"];
        for (FileNodeIterator itr = dataNode.begin(); itr != dataNode.end(); itr++)
        {
            CalibData data(*itr);
            if (!AddCalibData(data))
            {
                return;
            }
        }

        // TODO: sort calibration data according to their indices
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error restoring camera data";
        E_ERROR << ex.what();

        return;
    }
}

bool CalibCam::AddCalibData(CalibData& data)
{
    if (data.IsOkay())
    {
        if (m_imageSize == Size(0,0))
        {
            m_imageSize = data.GetImageSize();
        }
        else if(data.GetImageSize() != m_imageSize)
        {
            cerr << "Trying to add calibration data to a camera with inconsistent image size." << endl;
            cerr << "Size of added image " << data.GetImageSize().width << "x" << data.GetImageSize().height << " differs from " <<  m_imageSize.width << "x" << m_imageSize.height << endl;

            return false;
        }
    }

    m_data.push_back(data);
    return true;
}

double CalibCam::Calibrate()
{
    size_t nImages = m_data.size();
    m_imageIndcies.clear();

    // build point correspondences
    std::vector<ImagePointList>  imagePoints;
    std::vector<ObjectPointList> objectPoints;
    
    for (CalibDataSet::iterator data = m_data.begin(); data != m_data.end(); data++)
    {
        if (!data->IsOkay())
        {
            continue;
        }

        assert(data->GetImageSize() == m_imageSize);
        
        m_imageIndcies.push_back(data->GetIndex());
        imagePoints.   push_back(data->GetImagePoints());
        objectPoints.  push_back(data->GetObjectPoints());
    }

    if (m_imageIndcies.size() < 3)
    {
        E_ERROR << "calibration of cam #" << GetIndex() << " is not possible from " << m_imageIndcies.size() << " view(s)";
        return -1;
    }

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    // monocular calibration
    m_rpe = calibrateCamera(
        objectPoints, imagePoints, m_imageSize,
        m_intrinsics.CameraMatrix, m_intrinsics.DistortionMatrix, rvecs, tvecs,
        0, OptimTermCriteria);

    // remap extrinsics
    m_extrinsics = CalibExtrinsicsList(m_data.size());

    for (size_t i = 0; i < m_imageIndcies.size(); i++)
    {
        size_t idx = m_imageIndcies[i];
        cv::Mat rmat_i;
        cv::Rodrigues(rvecs[i], rmat_i);
        m_extrinsics[idx] = CalibExtrinsics(rmat_i, tvecs[i]);
    }

    return m_rpe;
}

bool CalibCam::Write(FileStorage& f) const
{
    f << "{:";
    f << "index" << (int) GetIndex();
    f << "data" << "[:";
    for (size_t img = 0; img < m_data.size(); img++)
    {
        if (!m_data[img].Store(f)) return false;
    }
    f << "]";
    f << "}";
    return true;
}

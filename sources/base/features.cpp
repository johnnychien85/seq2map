#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/join.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <seq2map/features.hpp>
#include <seq2map/features_opencv.hpp>

using namespace cv;
using namespace seq2map;

namespace po = boost::program_options;

const seq2map::String HetergeneousDetextractor::s_detectorFileNodeName  = "detection";
const seq2map::String HetergeneousDetextractor::s_extractorFileNodeName = "extraction";
const seq2map::String ImageFeatureSet::s_fileMagicNumber = "IMKPTDSC"; // which means IMage KeyPoinTs & DeSCriptors
const char ImageFeatureSet::s_fileHeaderSep = ' ';

FeatureDetectorFactory::FeatureDetectorFactory()
{
    Factory::Register<GFTTFeatureDetector>     ("GFTT" );
    Factory::Register<FASTFeatureDetector>     ("FAST" );
    Factory::Register<AGASTFeatureDetector>    ("AGAST");
    Factory::Register<ORBFeatureDetextractor>  ("ORB"  );
    Factory::Register<BRISKFeatureDetextractor>("BRISK");
    Factory::Register<KAZEFeatureDetextractor> ("KAZE" );
    Factory::Register<AKAZEFeatureDetextractor>("AKAZE");
#ifdef WITH_XFEATURES2D // non-free feature detectors .....
    Factory::Register<StarFeatureDetector>     ("STAR" );
    Factory::Register<MSDFeatureDetector>      ("MSD"  );
    Factory::Register<SIFTFeatureDetextractor> ("SIFT" );
    Factory::Register<SURFFeatureDetextractor> ("SURF" );
#endif // WITH_XFEATURES2D ................................
}

FeatureExtractorFactory::FeatureExtractorFactory()
{
    Factory::Register<ORBFeatureDetextractor>  ("ORB"  );
    Factory::Register<BRISKFeatureDetextractor>("BRISK");
    Factory::Register<KAZEFeatureDetextractor> ("KAZE" );
    Factory::Register<AKAZEFeatureDetextractor>("AKAZE");
#ifdef WITH_XFEATURES2D // non-free descriptor extractors..
    Factory::Register<SIFTFeatureDetextractor> ("SIFT" );
    Factory::Register<SURFFeatureDetextractor> ("SURF" );
    Factory::Register<BRIEFFeatureExtractor>   ("BRIEF");
    Factory::Register<DAISYFeatureExtractor>   ("DAISY");
    Factory::Register<FREAKFeatureExtractor>   ("FREAK");
    Factory::Register<LATCHFeatureExtractor>   ("LATCH");
    Factory::Register<LUCIDFeatureExtractor>   ("LUCID");
#endif // WITH_XFEATURES2D ................................
}

void FeatureDetextractorFactory::Init()
{
    Factory::Register<ORBFeatureDetextractor>  ("ORB"  );
    Factory::Register<BRISKFeatureDetextractor>("BRISK");
    Factory::Register<KAZEFeatureDetextractor> ("KAZE" );
    Factory::Register<AKAZEFeatureDetextractor>("AKAZE");
#ifdef WITH_XFEATURES2D // non-free feature detextractors..
    Factory::Register<SIFTFeatureDetextractor> ("SIFT" );
    Factory::Register<SURFFeatureDetextractor> ("SURF" );
#endif // WITH_XFEATURES2D ................................
}

ImageFeature ImageFeatureSet::GetFeature(const size_t idx)
{
    assert(idx < m_keypoints.size());
    return ImageFeature(m_keypoints[idx], m_descriptors.row((int)idx));
}

// ImageFeature ImageFeatureSet::GetFeature(const size_t idx) const
// {
//    assert(idx < m_keypoints.size());
//    return ImageFeature(m_keypoints[idx], m_descriptors.row((int)idx).clone());
// }

seq2map::String ImageFeatureSet::NormType2String(int type)
{
    switch (type)
    {
    case NORM_INF:      return "INF";      break;
    case NORM_L1:       return "L1";       break;
    case NORM_L2:       return "L2";       break;
    case NORM_L2SQR:    return "L2SQR";    break;
    case NORM_HAMMING:  return "HAMMING";  break;
    case NORM_HAMMING2: return "HAMMING2"; break;
    }

    E_WARNING << "unknown norm type " << type;

    return "UNKNOWN";
}

seq2map::String ImageFeatureSet::MatType2String(int type)
{
    switch (type)
    {
    case CV_8U:       return "8U";  break;
    case CV_8S:       return "8S";  break;
    case CV_16U:      return "16U"; break;
    case CV_16S:      return "16S"; break;
    case CV_32S:      return "32S"; break;
    case CV_32F:      return "32F"; break;
    case CV_64F:      return "64F"; break;
    case CV_USRTYPE1: return "USR"; break;
    }

    E_WARNING << "unknown matrix type " << type;

    return MatType2String(CV_USRTYPE1);
}

int seq2map::ImageFeatureSet::String2MatType(const seq2map::String& type)
{
    if      (type == "8U")   return CV_8U;
    else if (type == "8S")   return CV_8S;
    else if (type == "16U")  return CV_16U;       
    else if (type == "16S")  return CV_16S;  
    else if (type == "32S")  return CV_32S;  
    else if (type == "32F")  return CV_32F;  
    else if (type == "64F")  return CV_64F; 
    else if (type == "USR")  return CV_USRTYPE1;  

    E_WARNING << "unknown type string \"" << type << "\"";

    return String2MatType("USR");
}

int seq2map::ImageFeatureSet::String2NormType(const seq2map::String& type)
{

    if      (type == "INF")      return NORM_INF;
    else if (type == "L1")       return NORM_L1;
    else if (type == "L2")       return NORM_L2;       
    else if (type == "L2SQR")    return NORM_L2SQR;    
    else if (type == "HAMMING")  return NORM_HAMMING;  
    else if (type == "HAMMING2") return NORM_HAMMING2; 

    E_WARNING << "unknown string type \"" << type << "\"";

    return -1;
}

bool seq2map::ImageFeatureSet::Store(Path& path) const
{
    std::ofstream os(path.string().c_str(), std::ios::out | std::ios::binary);

    if (!os.is_open())
    {
        E_ERROR << "error opening output stream";
        return false;
    }

    // write the magic number first
    os << s_fileMagicNumber;

    // the header
    os << "CV3" << s_fileHeaderSep;
    os << NormType2String(m_normType)          << s_fileHeaderSep;
    os << MatType2String(m_descriptors.type()) << s_fileHeaderSep;

    os.write((char*)&m_descriptors.rows, sizeof m_descriptors.rows);
    os.write((char*)&m_descriptors.cols, sizeof m_descriptors.cols);

    // the key points section
    BOOST_FOREACH(const KeyPoint& kp, m_keypoints)
    {
        os.write((char*)&kp.pt.x,     sizeof kp.pt.x);
        os.write((char*)&kp.pt.y,     sizeof kp.pt.y);
        os.write((char*)&kp.response, sizeof kp.response);
        os.write((char*)&kp.octave,   sizeof kp.octave);
        os.write((char*)&kp.angle,    sizeof kp.angle);
        os.write((char*)&kp.size,     sizeof kp.size);
    }

    // feature vectors
    os.write((char*)m_descriptors.data, m_descriptors.elemSize() * m_descriptors.total());

    // end of file
    os.close();

    return true;
}

bool ImageFeatureSet::Restore(const Path& path)
{
    std::ifstream is(path.string().c_str(), std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        E_ERROR << "error opening input stream";
        return false;
    }

    char magicNumber[8];
    String version;
    String normType;
    String matType;
    Size2i matSize;

    is.read((char*)&magicNumber, sizeof magicNumber);

    if (!boost::equal(magicNumber, ImageFeatureSet::s_fileMagicNumber))
    {
        E_ERROR << "magic number not found";
        return false;
    }

    getline(is, version,  ImageFeatureSet::s_fileHeaderSep);
    getline(is, normType, ImageFeatureSet::s_fileHeaderSep);
    getline(is, matType,  ImageFeatureSet::s_fileHeaderSep);

    if (!boost::equals(version, "CV3"))
    {
        E_ERROR << "unknown file version";
        return false;
    }

    m_normType = String2NormType(normType);

    is.read((char*)&matSize.height, sizeof matSize.height);
    is.read((char*)&matSize.width,  sizeof matSize.width);

    m_keypoints.clear();
    m_keypoints.reserve(matSize.height);

    for (int i = 0; i < matSize.height; i++)
    {
        cv::KeyPoint kp;

        is.read((char*)&kp.pt.x,     sizeof kp.pt.x);
        is.read((char*)&kp.pt.y,     sizeof kp.pt.y);
        is.read((char*)&kp.response, sizeof kp.response);
        is.read((char*)&kp.octave,   sizeof kp.octave);
        is.read((char*)&kp.angle,    sizeof kp.angle);
        is.read((char*)&kp.size,     sizeof kp.size);

        m_keypoints.push_back(kp);
    }

    m_descriptors = Mat::zeros(matSize.height, matSize.width, String2MatType(matType)); 
    is.read((char*)m_descriptors.data, m_descriptors.elemSize() * m_descriptors.total());

    // TODO: add file corruption check (e.g. reaching EoF pre-maturely)
    // ..
    // ..

    return true;
}

FeatureDetextractor::Ptr FeatureDetextractorFactory::Create(const seq2map::String& detectorName, const seq2map::String& extractorName)
{
    FeatureDetextractor::Ptr dxtor;

    if (detectorName == extractorName)
    {
        const String& dxtorName = detectorName; // = extractorName
        dxtor = Factory::Create(dxtorName);

        if (!dxtor) E_ERROR << "feature detector-and-extractor unknown: " << dxtorName;
    }
    else
    {
        FeatureDetector::Ptr  detector = m_detectorFactory.Create(detectorName);
        FeatureExtractor::Ptr xtractor = m_extractorFactory.Create(extractorName);

        if (!detector) E_ERROR << "feature detector unknown: " << detectorName;
        if (!xtractor) E_ERROR << "descriptor extractor unknown: " << extractorName;

        if (detector && xtractor)
        {
            dxtor = FeatureDetextractor::Ptr(new HetergeneousDetextractor(detector, xtractor));
        }
    }

    if (dxtor)
    {
        dxtor->m_keypointType   = detectorName;
        dxtor->m_descriptorType = extractorName;
    }

    return dxtor;
}

FeatureDetextractor::Ptr FeatureDetextractorFactory::Create(const cv::FileNode& fn)
{
    FeatureDetextractor::Ptr dxtor;
    String keypointType, descriptorType;

    try
    {
        fn["keypoint"]   >> keypointType;
        fn["descriptor"] >> descriptorType;

        dxtor = Create(keypointType, descriptorType);

        if (dxtor) dxtor->Restore(fn);
    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught while creating feature detextractor";
        E_ERROR << ex.what();
    }

    return dxtor;
}

bool FeatureDetextractor::Store(cv::FileStorage& fs) const
{
    fs << "keypoint"   << m_keypointType;
    fs << "descriptor" << m_descriptorType;
    fs << "parameters" << "{";
    WriteParams(fs);
    fs << "}";

    return true;
}

bool FeatureDetextractor::Restore(const cv::FileNode& fn)
{
    String keypointType, descriptorType;

    try
    {
        fn["keypoint"]   >> keypointType;
        fn["descriptor"] >> descriptorType;

        bool compatible = boost::equals(keypointType,   m_keypointType) &&
                          boost::equals(descriptorType, m_descriptorType);
        if (!compatible)
        {
            E_ERROR << "incompatible feature keypoint and/or descriptor";
            E_ERROR << m_keypointType << " and " << m_descriptorType << " expected";
            E_ERROR << "while " << keypointType << " and " << descriptorType << " present";

            return false;
        }

        if (!ReadParams(fn["parameters"]))
        {
            E_ERROR << "error reading parameters from file node";
            return false;
        }
    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught while restoring feature detextractor";
        E_ERROR << ex.what();

        return false;
    }

    return true;
}

void HetergeneousDetextractor::WriteParams(cv::FileStorage& fs) const
{
    fs << s_detectorFileNodeName << "{";
    m_detector->WriteParams(fs);
    fs << "}";

    fs << s_extractorFileNodeName << "{";
    m_extractor->WriteParams(fs);
    fs << "}";
}

bool HetergeneousDetextractor::ReadParams(const cv::FileNode& fn)
{
    return m_detector ->ReadParams(fn[s_detectorFileNodeName] ) &&
        m_extractor->ReadParams(fn[s_extractorFileNodeName]);
}

void HetergeneousDetextractor::ApplyParams()
{
    m_detector ->ApplyParams();
    m_extractor->ApplyParams();
}

Parameterised::Options HetergeneousDetextractor::GetOptions(int flag)
{
    if (flag & DETECTION_OPTIONS ) return m_detector ->GetOptions(flag);
    if (flag & EXTRACTION_OPTIONS) return m_extractor->GetOptions(flag);

    return Options();
}

ImageFeatureSet HetergeneousDetextractor::DetectAndExtractFeatures(const Mat& im) const
{
    assert(m_detector && m_extractor);
    KeyPoints keypoints = m_detector->DetectFeatures(im);
    return m_extractor->ExtractFeatures(im, keypoints);
}

KeyPoints CvFeatureDetectorAdaptor::DetectFeatures(const cv::Mat& im) const
{
    KeyPoints keypoints;
    m_cvDetector->detect(im, keypoints);

    return keypoints;
}

ImageFeatureSet CvFeatureExtractorAdaptor::ExtractFeatures(const Mat& im, KeyPoints& keypoints) const
{
    Mat descriptors;
    m_cvExtractor->compute(im.clone(), keypoints, descriptors);

    return ImageFeatureSet(keypoints, descriptors, m_cvExtractor->defaultNorm());
}

ImageFeatureSet CvFeatureDetextractorAdaptor::DetectAndExtractFeatures(const Mat& im) const
{
    KeyPoints keypoints;
    Mat descriptors;

    m_cvDxtor->detectAndCompute(im, Mat(), keypoints, descriptors);

    return ImageFeatureSet(keypoints, descriptors, m_cvDxtor->defaultNorm());
}

template<class T>
void CvSuperDetextractorAdaptor<T>::SetCvSuperDetextractorPtr(CvDextractorPtr cvDxtor)
{
    SetCvDetectorPtr    (cvDxtor);
    SetCvExtractorPtr   (cvDxtor);
    SetCvDetextractorPtr(cvDxtor);
}

Indices ImageFeatureMap::Select(int mask) const
{
    Indices indices;
    for (size_t i = 0; i < m_matches.size(); i++)
    {
        if (m_matches[i].state & mask) indices.push_back(i);
    }
    return indices;
}

void ImageFeatureMap::Draw(Mat& canvas)
{
    const KeyPoints& src = m_src.GetKeyPoints();
    const KeyPoints& dst = m_dst.GetKeyPoints();

    cv::Scalar inlierColour  = cv::Scalar(  0, 255,   0);
    cv::Scalar outlierColour = cv::Scalar(200, 200, 200);

    BOOST_FOREACH (const FeatureMatch& match, m_matches)
    {
        if (match.state & FeatureMatch::RATIO_TEST_FAILED) continue;

        bool inlier = match.state & FeatureMatch::INLIER;
        cv::line(canvas, src[match.srcIdx].pt, dst[match.dstIdx].pt, inlier ? inlierColour : outlierColour);
    }
}

Mat ImageFeatureMap::Draw(const cv::Mat& src, const cv::Mat& dst)
{
    Mat canvas = imfuse(src, dst);
    Draw(canvas);

    return canvas;
}

ImageFeatureMap FeatureMatcher::MatchFeatures(const ImageFeatureSet& src, const ImageFeatureSet& dst)
{
    ImageFeatureMap map(src, dst);

    if (src.GetNormType() != dst.GetNormType())
    {
        int stype = src.GetNormType(), dtype = dst.GetNormType();

        E_ERROR << "given feature sets have different norm types";
        E_ERROR << "the first set has:        " << ImageFeatureSet::NormType2String(stype) << " (" << stype << ")";
        E_ERROR << "while the second set has: " << ImageFeatureSet::NormType2String(dtype) << " (" << dtype << ")";

        return map;
    }

    int metric = src.GetNormType(); // == dst.GetNormType();

    bool normalisation = metric != NORM_HAMMING && metric != NORM_HAMMING2;
    cv::Mat srcDescriptors = normalisation ? NormaliseDescriptors(src.GetDescriptors()) : src.GetDescriptors();
    cv::Mat dstDescriptors = normalisation ? NormaliseDescriptors(dst.GetDescriptors()) : dst.GetDescriptors();

    // Perform descriptor matching in feature space
    map.m_matches = MatchDescriptors(srcDescriptors, dstDescriptors, metric);

    // no match found?
    if (map.m_matches.empty()) return map;

    // Perform symmetry test
    if (m_symmetric)
    {
        FeatureMatches& forward = map.m_matches;
        FeatureMatches backward = MatchDescriptors(dstDescriptors, srcDescriptors, metric);

        AutoSpeedometreMeasure measure(m_symmetryTestMetre, map.m_matches.size());
        RunSymmetryTest(forward, backward);
    }

    Indices inliers = map.Select(FeatureMatch::INLIER);

    BOOST_FOREACH(FeatureMatchFilter* filter, m_filters)
    {
        size_t total = inliers.size();
        size_t passed = 0;

        AutoSpeedometreMeasure measure(m_filteringMetre, total);

        if (!filter->Filter(map, inliers))
        {
            E_INFO << "filtering stopped because a filter failed";
            break;
        }

        passed = inliers.size();
        E_INFO << "survival rate: " << passed << " / " << total;

        if (passed == 0)
        {
            E_INFO << "filtering stopped because no inlier survived";
            break;
        }
    }

    return map;
}

FeatureMatcher& FeatureMatcher::AddFilter(FeatureMatchFilter& filter)
{
    m_filters.push_back(&filter);
    return *this;
};

cv::Mat FeatureMatcher::NormaliseDescriptors(const cv::Mat& desc)
{
    cv::Mat normalised = cv::Mat(desc.rows, desc.cols, desc.type());
    for (int i = 0; i < desc.rows; i++)
    {
        cv::normalize(desc.row(i), normalised.row(i));
    }

    return normalised;
}

FeatureMatches FeatureMatcher::MatchDescriptors(const Mat& src, const Mat& dst, int metric)
{
    FeatureMatches matches;
    matches.reserve(src.rows);

    const bool ratioTest = m_maxRatio > 0.0f && m_maxRatio < 1.0f;
    const int k = ratioTest ? 2 : 1;
    std::vector<std::vector<DMatch> > knn;

    if (m_useGpu && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(metric);
        
        {
            AutoSpeedometreMeasure measure(m_descMatchingMetre, src.rows + dst.rows);
            cv::cuda::GpuMat D1, D2;

            D1.upload(src);
            D2.upload(dst);

            matcher->knnMatch(D1, D2, knn, k);
        }
    }
    else
    {
        Ptr<cv::DescriptorMatcher> matcher = m_exhaustive ?
            Ptr<DescriptorMatcher>(new BFMatcher(metric)) : Ptr<DescriptorMatcher>(new FlannBasedMatcher());

        {
            AutoSpeedometreMeasure measure(m_descMatchingMetre, src.rows + dst.rows);
            matcher->knnMatch(src, dst, knn, k);
        }
    }

    if (ratioTest)
    {
        AutoSpeedometreMeasure measure(m_ratioTestMetre, knn.size());
        BOOST_FOREACH(const std::vector<DMatch>& match, knn)
        {
            float ratio = match[0].distance / match[1].distance;
            int flag = ratio < m_maxRatio ? FeatureMatch::INLIER : FeatureMatch::RATIO_TEST_FAILED;

            matches.push_back(FeatureMatch(match[0].queryIdx, match[0].trainIdx, match[0].distance, flag));
        }
    }
    else
    {
        BOOST_FOREACH(const std::vector<DMatch>& match, knn)
        {
            matches.push_back(FeatureMatch(match[0].queryIdx, match[0].trainIdx, match[0].distance));
        }
    }

    return matches;
}

void FeatureMatcher::RunSymmetryTest(FeatureMatches& forward, const FeatureMatches& backward)
{
    size_t n = forward.size();

    std::vector<size_t> dstIdx(n, INVALID_INDEX);
    std::vector<bool>   good(n, false);
   
    BOOST_FOREACH(FeatureMatch& match, forward)
    {
        dstIdx[match.srcIdx] = match.dstIdx;
    }

    BOOST_FOREACH(const FeatureMatch& match, backward)
    {
        size_t i = match.dstIdx;
        size_t j = match.srcIdx;

        good[i] = good[i] ? good[i] : (dstIdx[i] == j);
    } 

    BOOST_FOREACH(FeatureMatch& match, forward)
    {
        if (good[match.srcIdx] == false)
        {
            match.Reject(FeatureMatch::UNIQUENESS_FAILED);
        }
    }
}

seq2map::String FeatureMatcher::Report() const
{
    std::vector<String> summary;
    summary.push_back(m_descMatchingMetre.ToString());
    summary.push_back(m_ratioTestMetre   .ToString());
    summary.push_back(m_symmetryTestMetre.ToString());
    summary.push_back(m_filteringMetre   .ToString());

    return boost::algorithm::join(summary, " / ") + (m_useGpu ? " [GPU]" : "");
}

bool FundamentalMatrixFilter::Filter(ImageFeatureMap& map, Indices& inliers)
{
    // at least 8 point correspondences are required 
    //  to estimate the fundamental matrix
    if (inliers.size() < 8)
    {
        E_WARNING << "insufficient inliers of " << inliers.size() << ", eight required minimally";
        return false;
    }

    Points2D pts0, pts1;
    const KeyPoints& src = map.From().GetKeyPoints();
    const KeyPoints& dst = map.To().GetKeyPoints();
    size_t inliersSize = inliers.size();
    std::vector<uchar> mask(inliersSize, 0);

    pts0.reserve(inliersSize);
    pts1.reserve(inliersSize);

    BOOST_FOREACH(size_t idx, inliers)
    {
        size_t i = map[idx].srcIdx;
        size_t j = map[idx].dstIdx;

        pts0.push_back(src[i].pt);
        pts1.push_back(dst[j].pt);
    }

    int method = m_ransac ? CV_FM_RANSAC : CV_FM_LMEDS;
    m_fmat = findFundamentalMat(pts0, pts1, mask, method, m_epsilon, m_confidence);

    Indices::iterator itr = inliers.begin(); 
    for (size_t i = 0; itr != inliers.end(); i++)
    {
        bool outlier = mask[i] == 0;

        if (outlier)
        {
            map[*itr].Reject(FeatureMatch::GEOMETRIC_TEST_FAILED);
            inliers.erase(itr++);
        }
        else
        {
            itr++;
        }
    }

    return true;
}

void EssentialMatrixFilter::SetCameraMatrices(const cv::Mat& K0, const cv::Mat& K1)
{
    if (!checkCameraMatrix(K0) || !checkCameraMatrix(K1))
    {
        E_ERROR << "invalid camera matrix/matrices given";
        E_ERROR << mat2string(K0, "K0");
        E_ERROR << mat2string(K1, "K1");

        m_K0inv = cv::Mat::eye(3, 3, CV_64F);
        m_K1inv = cv::Mat::eye(3, 3, CV_64F);

        return;
    }

    m_K0inv = K0.inv();
    m_K1inv = K1.inv();

    m_K0inv.convertTo(m_K0inv, CV_64F);
    m_K1inv.convertTo(m_K1inv, CV_64F);
}

bool EssentialMatrixFilter::Filter(ImageFeatureMap& map, Indices& inliers)
{
    // at least 5 point correspondences are required 
    //  to estimate the essential matrix
    if (inliers.size() < 8)
    {
        E_WARNING << "insufficient inliers of " << inliers.size() << ", five required minimally";
        return false;
    }

    Points2D pts0, pts1;
    const KeyPoints& src = map.From().GetKeyPoints();
    const KeyPoints& dst = map.To().  GetKeyPoints();
    size_t inliersSize = inliers.size();
    std::vector<uchar> mask(inliersSize, 0);
    Mat I = cv::Mat::eye(3, 3, CV_64F);

    pts0.reserve(inliersSize);
    pts1.reserve(inliersSize);

    BOOST_FOREACH (size_t idx, inliers)
    {
        size_t i = map[idx].srcIdx;
        size_t j = map[idx].dstIdx;

        pts0.push_back(src[i].pt);
        pts1.push_back(dst[j].pt);
    }

    BackprojectPoints(pts0, m_K0inv);
    BackprojectPoints(pts1, m_K1inv);

    int method = m_ransac ? RANSAC : LMEDS;
    m_emat = findEssentialMat(pts0, pts1, I, method, m_confidence, m_epsilon, mask);

    if (m_emat.empty())
    {
        E_WARNING << "findEssentialMat returns empty matrix";
        return false;
    }

    Indices::iterator itr = inliers.begin(); 
    for (size_t i = 0; itr != inliers.end(); i++)
    {
        bool outlier = mask[i] == 0;

        if (outlier)
        {
            map[*itr].Reject(FeatureMatch::GEOMETRIC_TEST_FAILED);
            inliers.erase(itr++);
        }
        else
        {
            itr++;
        }
    }

    if (m_poseRecovery)
    {
        recoverPose(m_emat, pts0, pts1, I, m_rmat, m_tvec, Mat(mask));
    }

    return true;
}

void EssentialMatrixFilter::BackprojectPoints(Points2D& pts, const cv::Mat& Kinv)
{
    const double* Ki = Kinv.ptr<double>();
    
    //      / ki0 ki1 ki2 \
    // Ki = | ki3 ki4 ki5 |
    //      \ ki6 ki7 ki8 /

    BOOST_FOREACH (Point2D& pt, pts)
    {
        double x = pt.x * Ki[0] + pt.y * Ki[1] + Ki[2];
        double y = pt.x * Ki[3] + pt.y * Ki[4] + Ki[5];
        double z = pt.x * Ki[6] + pt.y * Ki[7] + Ki[8];

        pt.x = x / z;
        pt.y = y / z;
    }
}

bool SigmaFilter::Filter(ImageFeatureMap& map, Indices& inliers)
{
    if (inliers.size() < 2) return true;

    m_mean  = 0;
    m_stdev = 0;

    // calculate mean
    BOOST_FOREACH (size_t idx, inliers)
    {
        m_mean += map[idx].distance;
    }
    m_mean = m_mean / inliers.size();

    // calculate standard deviation
    BOOST_FOREACH (size_t idx, inliers)
    {
        float d = map[idx].distance - m_mean;
        m_stdev += d * d;
    }
    m_stdev = std::sqrt(m_stdev / inliers.size());

    // calculate k-sigma
    float ksigma = m_k * m_stdev;
    float cutoff = m_mean + ksigma;

    Indices::iterator itr = inliers.begin();
    while (itr != inliers.end())
    {
        bool outlier = map[*itr].distance > cutoff;

        if (outlier)
        {
            map[*itr].Reject(FeatureMatch::SIGMA_TEST_FAILED);
            inliers.erase(itr++);
        }
        else
        {
            itr++;
        }
    }

    return true;
}

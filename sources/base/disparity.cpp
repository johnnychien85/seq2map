#include <boost/algorithm/string/predicate.hpp>
#include <seq2map/disparity.hpp>

using namespace seq2map;
namespace po = boost::program_options;

bool StereoMatcher::Store(cv::FileStorage& fs) const
{
    fs << "matcherName" << GetMatcherName();
    WriteParams(fs);

    return true;
}

bool StereoMatcher::Restore(const cv::FileNode& fn)
{
    String matcherName;
    fn["matcherName"] >> matcherName;

    if (!boost::equals(matcherName, GetMatcherName()))
    {
        E_ERROR << "error restoring from file node";
        E_ERROR << "node presents a \"" << matcherName << "\" type while I am \"" << GetMatcherName() << "\"";

        return false;
    }

    return ReadParams(fn);
}

template<class T>
void CvStereoMatcher<T>::WriteParams(cv::FileStorage& fs) const
{
    fs << "minDisparity"   << m_minDisparity;
    fs << "numDisparities" << m_numDisparities;
    fs << "blockSize"      << m_blockSize;
    fs << "speckleWinSize" << m_speckleWinSize;
    fs << "speckleRange"   << m_speckleRange;
    fs << "disp12MaxDiff"  << m_disp12MaxDiff;
}

template<class T>
bool CvStereoMatcher<T>::ReadParams(const cv::FileNode& fn)
{
    fn["minDisparity"]   >> m_minDisparity;
    fn["numDisparities"] >> m_numDisparities;
    fn["blockSize"]      >> m_blockSize;
    fn["speckleWinSize"] >> m_speckleWinSize;
    fn["speckleRange"]   >> m_speckleRange;
    fn["disp12MaxDiff"]  >> m_disp12MaxDiff;

    return true;
}

template<class T>
void CvStereoMatcher<T>::ApplyParams()
{
    try
    {
        m_matcher->setMinDisparity     (m_minDisparity);
        m_matcher->setNumDisparities   (m_numDisparities);
        m_matcher->setBlockSize        (m_blockSize);
        m_matcher->setSpeckleWindowSize(m_speckleWinSize);
        m_matcher->setSpeckleRange     (m_speckleRange);
        m_matcher->setDisp12MaxDiff    (m_disp12MaxDiff);
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error applying parameters to internal OpenCV matcher object";
        E_ERROR << ex.what();
    }
}

template<class T>
Parameterised::Options CvStereoMatcher<T>::GetOptions(int flag)
{
    Options o("OpenCV general stereo matching options");
    o.add_options()
        ("dmin",          po::value<int>(&m_minDisparity  )->default_value(m_matcher->getMinDisparity()),      "Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.")
        ("dnum",          po::value<int>(&m_numDisparities)->default_value(m_matcher->getNumDisparities()),    "Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation of SGBM, this parameter must be divisible by 16.")
        ("block-size",    po::value<int>(&m_blockSize     )->default_value(m_matcher->getBlockSize()),         "The linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence.")
        ("speckle-size",  po::value<int>(&m_speckleWinSize)->default_value(m_matcher->getSpeckleWindowSize()), "Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.")
        ("speckle-range", po::value<int>(&m_speckleRange  )->default_value(m_matcher->getSpeckleRange()),      "Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.")
        ("d12-max-diff",  po::value<int>(&m_disp12MaxDiff )->default_value(m_matcher->getDisp12MaxDiff()),     "Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.");

    return o;
}

template<class T>
cv::Mat CvStereoMatcher<T>::Match(const cv::Mat& left, const cv::Mat& right)
{
    cv::Mat dp16U, dp32F;

    if (m_useGpuMat)
    {
        cv::cuda::GpuMat I1, I2, dp;
        
        I1.upload(left);
        I2.upload(right);

        m_matcher->compute(I1, I2, dp);
        dp.download(dp16U);
    }
    else
    {
        m_matcher->compute(left, right, dp16U);
    }

    dp16U.convertTo(dp32F, CV_32F, 1.0f/(double)cv::StereoMatcher::DISP_SCALE);

    return dp32F;
}

template<class T>
String BlockMatcher<T>::FilterType2String(int type)
{
    switch (type)
    {
    case cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE: return "NORMALISED"; break;
    case cv::StereoBM::PREFILTER_XSOBEL:              return "XSOBEL";     break;
    }

    E_WARNING << "unknown type " << type;
    return "UNKNOWN";
}

template<class T>
int BlockMatcher<T>::String2FilterType(String type)
{
    if      (type == "NORMALISED") return cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE;
    else if (type == "XSOBEL")     return cv::StereoBM::PREFILTER_XSOBEL;
    
    E_WARNING << "unknown type: \"" << type << "\"";

    return -1;
}

template<class T>
void BlockMatcher<T>::WriteParams(cv::FileStorage& fs) const
{
    CvStereoMatcher::WriteParams(fs);

    fs << "preFilterType"    << m_preFilterType;
    fs << "preFilterCap"     << m_preFilterCap;
    fs << "textureThreshold" << m_textureThreshold;
    fs << "uniquenessRatio"  << m_uniquenessRatio;
    fs << "smallerBlockSize" << m_smallerBlockSize;
}

template<class T>
bool BlockMatcher<T>::ReadParams(const cv::FileNode& fn)
{
    fn["preFilterType"]    >> m_preFilterType;
    fn["preFilterCap"]     >> m_preFilterCap;
    fn["textureThreshold"] >> m_textureThreshold;
    fn["uniquenessRatio"]  >> m_uniquenessRatio;
    fn["smallerBlockSize"] >> m_smallerBlockSize;

    return CvStereoMatcher::ReadParams(fn);
}

template<class T>
void BlockMatcher<T>::ApplyParams()
{
    CvStereoMatcher::ApplyParams();

    m_matcher->setPreFilterType   (String2FilterType(m_preFilterType));
    m_matcher->setPreFilterCap    (m_preFilterCap);
    m_matcher->setTextureThreshold(m_textureThreshold);
    m_matcher->setUniquenessRatio (m_uniquenessRatio);
    m_matcher->setSmallerBlockSize(m_smallerBlockSize);
}

template<class T>
Parameterised::Options BlockMatcher<T>::GetOptions(int flag)
{
    Options g = CvStereoMatcher::GetOptions(flag);
    Options o("OpenCV block matcher options");

    const String preFilterType = FilterType2String(m_matcher->getPreFilterType());

    o.add_options()
        ("pre-filter-type",    po::value<String>(&m_preFilterType   )->default_value(preFilterType),                    "NORMALISED or XSOBEL")
        ("pre-filter-cap",     po::value<int>   (&m_preFilterCap    )->default_value(m_matcher->getPreFilterCap()),     "...")
        ("texture-threshold",  po::value<int>   (&m_textureThreshold)->default_value(m_matcher->getTextureThreshold()), "...")
        ("uniqueness-ratio",   po::value<int>   (&m_uniquenessRatio )->default_value(m_matcher->getUniquenessRatio()),  "Margin in percentage by which the best (minimum) computed cost function value should \"win\" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.")
        ("smaller-block-size", po::value<int>   (&m_smallerBlockSize)->default_value(m_matcher->getSmallerBlockSize()), "...");

    return g.add(o);
}


String SemiGlobalBlockMatcher::Mode2String(int type)
{
    switch (type)
    {
    case cv::StereoSGBM::MODE_HH:        return "FULL";  break;
    case cv::StereoSGBM::MODE_SGBM:      return "SGBM";  break;
    case cv::StereoSGBM::MODE_SGBM_3WAY: return "SGBM3"; break;
    }

    E_WARNING << "unknown type " << type;
    return "UNKNOWN";
}

int SemiGlobalBlockMatcher::String2Mode(String type)
{
    if      (type == "FULL")   return cv::StereoSGBM::MODE_HH;
    else if (type == "SGBM")   return cv::StereoSGBM::MODE_SGBM;
    else if (type == "SGBM3")  return cv::StereoSGBM::MODE_SGBM_3WAY;

    E_WARNING << "unknown type: \"" << type << "\"";

    return -1;
}

void SemiGlobalBlockMatcher::WriteParams(cv::FileStorage& fs) const
{
    CvStereoMatcher::WriteParams(fs);

    fs << "mode"            << m_mode;
    fs << "preFilterCap"    << m_preFilterCap;
    fs << "uniquenessRatio" << m_uniquenessRatio;
}

bool SemiGlobalBlockMatcher::ReadParams(const cv::FileNode& fn)
{
    fn["mode"]            >> m_mode;
    fn["preFilterCap"]    >> m_preFilterCap;
    fn["uniquenessRatio"] >> m_uniquenessRatio;

    return CvStereoMatcher::ReadParams(fn);
}

void SemiGlobalBlockMatcher::ApplyParams()
{
    CvStereoMatcher::ApplyParams();

    if (m_p1 == 0) m_p1 = 8  * m_blockSize * m_blockSize;
    if (m_p2 == 0) m_p2 = 32 * m_blockSize * m_blockSize;


    m_matcher->setMode           (String2Mode(m_mode));
    m_matcher->setPreFilterCap   (m_preFilterCap);
    m_matcher->setUniquenessRatio(m_uniquenessRatio);
    m_matcher->setP1             (m_p1);
    m_matcher->setP2             (m_p2);
}

Parameterised::Options SemiGlobalBlockMatcher::GetOptions(int flag)
{
    Options g = CvStereoMatcher::GetOptions(flag);
    Options o("OpenCV semi-global block matcher options");

    o.add_options()
        ("mode",             po::value<String>(&m_mode           )->default_value(Mode2String(m_matcher->getMode())), "SGBM, SGBM3, or FULL")
        ("pre-filter-cap",   po::value<int>   (&m_preFilterCap   )->default_value(m_matcher->getPreFilterCap()     ), "Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.")
        ("uniqueness-ratio", po::value<int>   (&m_uniquenessRatio)->default_value(m_matcher->getUniquenessRatio()  ), "Margin in percentage by which the best (minimum) computed cost function value should \"win\" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.")
        ("p1",               po::value<int>   (&m_p1             )->default_value(                                0), "The first parameter controlling the disparity smoothness.")
        ("p2",               po::value<int>   (&m_p2             )->default_value(                                0), "The second parameter controlling the disparity smoothness.");

    return g.add(o);
}

void StereoMatcherFactory::Init()
{
    Register<CpuBlockMatcher>       ("BM");
    Register<SemiGlobalBlockMatcher>("SGBM");

    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        Register<GpuBlockMatcher>("BM_GPU");
    }
}

StereoMatcher::Ptr StereoMatcherFactory::Create(const cv::FileNode& fn)
{
    String matcherName;
    fn["matcherName"] >> matcherName;

    StereoMatcher::Ptr matcher = Factory::Create(matcherName);

    if (!matcher)
    {
        E_ERROR << "error creating stereo matcher \"" << matcherName << "\"";
        return StereoMatcher::Ptr();
    }

    if (!matcher->Restore(fn))
    {
        E_ERROR << "error restoring stereo matcher from file node";
        return StereoMatcher::Ptr();
    }

    return matcher;
}

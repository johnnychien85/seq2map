#include <seq2map/disparity.hpp>
#include <climits>

using namespace cv;
using namespace seq2map;

void StereoMatcherAdaptor::WriteParams(cv::FileStorage& f)
{
    f << "matcher" << m_matcherName;
    f << "numDisparities" << m_numDisparities;
    f << "denorm"  << GetNormalisationFactor();
    f << "scale"   << GetScale();
}

BlockMatcher::BlockMatcher(int numDisparities, int SADWindowSize)
    : StereoMatcherAdaptor("BM", numDisparities),
    m_BM(StereoBM::create(numDisparities, SADWindowSize))
{
    if (m_BM.empty())
    {
        E_FATAL << "OpenCV BM initialisation failed";
    }
}

BlockMatcher::~BlockMatcher()
{
    m_BM.release();
}

Mat	BlockMatcher::Match(const Mat& left, const Mat& right)
{
    Mat dpmap;
    if (!m_BM.empty()) m_BM->compute(left, right, dpmap);

    return dpmap;
}

void BlockMatcher::WriteParams(FileStorage& f)
{
    assert(!m_BM.empty());

    StereoMatcherAdaptor::WriteParams(f);
    f << "SADWindowSize" << m_BM->getBlockSize();
}

SemiGlobalMatcher::SemiGlobalMatcher(int numDisparities, int SADWindowSize,
    int P1, int P2, int disp12MaxDiff, int preFilterCap, int uniquenessRatio,
    int speckleWindowSize, int speckleRange, bool fullDP)
    : StereoMatcherAdaptor("SGM", numDisparities)
{
	m_SGBM = StereoSGBM::create(0, numDisparities, SADWindowSize,
		P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio,
		speckleWindowSize, speckleRange, fullDP ? StereoSGBM::MODE_HH : StereoSGBM::MODE_SGBM);

	if (m_SGBM.empty())
	{
		E_FATAL << "OpenCV SGBM object initialisation failed";
	}
}

SemiGlobalMatcher::~SemiGlobalMatcher()
{
	m_SGBM.release();
}

Mat SemiGlobalMatcher::Match(const Mat& left, const Mat& right)
{
    Mat dpmap;
    if (!m_SGBM.empty()) m_SGBM->compute(left, right, dpmap);

    return dpmap;
}

void SemiGlobalMatcher::WriteParams(FileStorage& f)
{
    assert(!m_SGBM.empty());

    StereoMatcherAdaptor::WriteParams(f);
	f << "minDisparity" << m_SGBM->getMinDisparity();
	f << "numDisparities" << m_SGBM->getNumDisparities();
	f << "SADWindowSize" << m_SGBM->getBlockSize();
	f << "preFilterCap" << m_SGBM->getPreFilterCap();
	f << "uniquenessRatio" << m_SGBM->getUniquenessRatio();
	f << "P1" << m_SGBM->getP1();
	f << "P2" << m_SGBM->getP2();
	f << "speckleWindowSize" << m_SGBM->getSpeckleWindowSize();
	f << "speckleRange" << m_SGBM->getSpeckleRange();
	f << "disp12MaxDiff" << m_SGBM->getDisp12MaxDiff();
	f << "fullDP" << (m_SGBM->getMode() == StereoSGBM::MODE_HH);
}

bool DisparityIO::Write(const Mat& dpmap, const Path& path)
{
    Mat dpmap16U;
    dpmap.convertTo(dpmap16U, CV_16U);

    return imwrite(path.string(), m_denorm * dpmap16U);
}

Mat DisparityIO::Read(const Path& path)
{
    Mat dpmap = imread(path.string(), IMREAD_GRAYSCALE);

    if (dpmap.empty() || dpmap.depth() != CV_16U)
    {
        E_ERROR << "either the image is not readable or it's depth is not CV_16U";
        return Mat();
    }

    // rescale disparity image
    dpmap /= m_denorm; // denormalisation
    dpmap.convertTo(dpmap, CV_32F);
    dpmap /= m_scale; // recover subpixel scale

    return dpmap;
}

seq2map::StereoMatcher::Ptr StereoMatcherFactory::Create(const cv::FileNode& fn)
{
    return seq2map::StereoMatcher::Ptr();
}

void StereoMatcherFactory::Init()
{

}

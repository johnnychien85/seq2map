#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <seq2map/features.hpp>

using namespace cv;
using namespace seq2map;

namespace po = boost::program_options;

const seq2map::String HetergeneousDetextractor::s_detectorFileNodeName  = "detection";
const seq2map::String HetergeneousDetextractor::s_extractorFileNodeName = "extraction";
const seq2map::String ImageFeatureSet::s_fileMagicNumber = "IMKPTDSC"; // which means IMage KeyPoinTs & DeSCriptors
const seq2map::String ImageFeatureSet::s_fileHeaderSep = " ";

FeatureDetectorFactory::FeatureDetectorFactory()
{
    Factory::Register<FASTFeatureDetector>     ( "FAST"  );
    Factory::Register<GFTTFeatureDetector>     ( "GFTT"  );
    Factory::Register<ORBFeatureDetextractor>  ( "ORB"   );
//  Factory::Register<BRISKFeatureDetextractor>( "BRISK" );
#ifdef HAVE_XFEATURES2D // non-free feature detectors .....
    Factory::Register<SIFTFeatureDetextractor> ( "SIFT"  );
//  Factory::Register<SURFFeatureDetextractor> ( "SURF"  );
#endif // HAVE_XFEATURES2D ................................
}

FeatureExtractorFactory::FeatureExtractorFactory()
{
    Factory::Register<ORBFeatureDetextractor>  ( "ORB"   );
//  Factory::Register<BRISKFeatureDetextractor>( "BRISK" );
#ifdef HAVE_XFEATURES2D // non-free descriptor extractors..
    Factory::Register<SIFTFeatureDetextractor> ( "SIFT"  );
//  Factory::Register<SURFFeatureDetextractor> ( "SURF"  );
//  Factory::Register<BRIEFFeatureExtractor>   ( "BRIEF" );
#endif // HAVE_XFEATURES2D ................................
}

FeatureDetextractorFactory::FeatureDetextractorFactory()
{
    Factory::Register<ORBFeatureDetextractor>  ( "ORB"   );
//	Factory::Register<BRISKFeatureDetextractor>( "BRISK" );
//  Factory::Register<KAZEFeatureDetextractor> ( "KAZE"  );
//  Factory::Register<AKAZEFeatureDetextractor>( "AKAZE" );
#ifdef HAVE_XFEATURES2D // non-free feature detextractors..
    Factory::Register<SIFTFeatureDetextractor> ( "SIFT"  );
//  Factory::Register<SURFFeatureDetextractor> ( "SURF"  );
#endif // HAVE_XFEATURES2D ................................
}

ImageFeature ImageFeatureSet::GetFeature(const size_t idx)
{
    assert(idx < m_keypoints.size());
    return ImageFeature(m_keypoints[idx], m_descriptors.row((int)idx));
}

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

    E_WARNING << "unknown norm type: " << type;

    return "UNKNOWN";
}

seq2map::String ImageFeatureSet::MatType2String(int type)
{
    switch (type)
    {
    case CV_8U:  return "8U";  break;
    case CV_8S:  return "8S";  break;
    case CV_16U: return "16U"; break;
    case CV_16S: return "16S"; break;
    case CV_32S: return "32S"; break;
    case CV_32F: return "32F"; break;
    case CV_64F: return "64F"; break;
    case CV_USRTYPE1: return "USR"; break;
    }
    E_WARNING << "unknown matrix type: " << type;
    return MatType2String(CV_USRTYPE1);
}

bool ImageFeatureSet::Write(const Path& path) const
{
    std::ofstream of(path.string(), std::ios::out | std::ios::binary);
    
    if (!of.is_open())
    {
        E_ERROR << "error opening output stream";
        return false;
    }

    // write the magic number first
    of << s_fileMagicNumber;

    // the header
    of << "CV3"                                 << s_fileHeaderSep;
    of << NormType2String(m_normType)           << s_fileHeaderSep;
    of << MatType2String (m_descriptors.type()) << s_fileHeaderSep;

    of.write((char*) &m_descriptors.rows, sizeof m_descriptors.rows);
    of.write((char*) &m_descriptors.cols, sizeof m_descriptors.cols);

    // the key points section
    BOOST_FOREACH(const KeyPoint& kp, m_keypoints)
    {
        of.write((char*)&kp.pt.x,     sizeof kp.pt.x);
        of.write((char*)&kp.pt.y,     sizeof kp.pt.y);
        of.write((char*)&kp.response, sizeof kp.response);
        of.write((char*)&kp.octave,   sizeof kp.octave);
        of.write((char*)&kp.angle,    sizeof kp.angle);
        of.write((char*)&kp.size,     sizeof kp.size);
    }

    // feature vectors
    of.write((char*)m_descriptors.data, m_descriptors.elemSize() * m_descriptors.total());

    // end of file
    of.close();

    return true;
}

FeatureDetextractorPtr FeatureDetextractorFactory::Create(const seq2map::String& detectorName, const seq2map::String& extractorName)
{
	FeatureDetextractorPtr dxtor;

	if (detectorName == extractorName)
	{
        const String& dxtorName = detectorName; // = extractorName
        dxtor = Factory::Create(dxtorName);

        if (!dxtor) E_ERROR << "feature detector-and-extractor unknown: " << dxtorName;
	}
	else
	{
        FeatureDetectorPtr  detector = m_detectorFactory.Create(detectorName);
        FeatureExtractorPtr xtractor = m_extractorFactory.Create(extractorName);

        if (!detector) E_ERROR << "feature detector unknown: "     << detectorName;
        if (!xtractor) E_ERROR << "descriptor extractor unknown: " << extractorName;

        if (detector && xtractor)
        {
            dxtor = FeatureDetextractorPtr(new HetergeneousDetextractor(detector, xtractor));
        }
	}

	return dxtor;
}

void HetergeneousDetextractor::WriteParams(cv::FileStorage& fs) const
{
    fs << s_detectorFileNodeName << "{:";
    m_detector->WriteParams(fs);
    fs << "}";

    fs << s_extractorFileNodeName << "{:";
    m_extractor->WriteParams(fs);
    fs << "}";
}

bool HetergeneousDetextractor::ReadParams(const cv::FileNode& fn)
{
    return m_detector ->ReadParams(fn[s_detectorFileNodeName ]) &&
           m_extractor->ReadParams(fn[s_extractorFileNodeName]);
}

void HetergeneousDetextractor::ApplyParams()
{
    m_detector ->ApplyParams();
    m_extractor->ApplyParams();
}

Parameterised::Options HetergeneousDetextractor::GetOptions(int flag)
{
    if (flag & FeatureOptionType::DETECTION_OPTIONS)  return m_detector ->GetOptions(flag);
    if (flag & FeatureOptionType::EXTRACTION_OPTIONS) return m_extractor->GetOptions(flag);

    return Options();
}

ImageFeatureSet HetergeneousDetextractor::DetectAndExtractFeatures(const Mat& im) const
{
    assert(m_detector && m_extractor);
    return m_extractor->ExtractFeatures(im, m_detector->DetectFeatures(im));
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
    SetCvDetectorPtr(cvDxtor);
    SetCvExtractorPtr(cvDxtor);
    SetCvDetextractorPtr(cvDxtor);
}

/**
 * OpenCV Feature Detection Wrapper of
 *  "Good Feature to Track (GFTT)"
 */

void GFTTFeatureDetector::WriteParams(cv::FileStorage& fs) const
{
    fs << "maxFeatures"  << m_gftt->getMaxFeatures();
    fs << "qualityLevel" << m_gftt->getQualityLevel();
    fs << "minDistance"  << m_gftt->getMinDistance();
    fs << "blockSize"    << m_gftt->getBlockSize();
    fs << "harrisCorner" << m_gftt->getHarrisDetector();
    fs << "harrisK"      << m_gftt->getK();
}

bool GFTTFeatureDetector::ReadParams(const cv::FileNode& fn)
{
    fn["maxFeatures"]  >> m_maxFeatures;
    fn["qualityLevel"] >> m_qualityLevel;
    fn["minDistance"]  >> m_minDistance;
    fn["blockSize"]    >> m_blockSize;
    fn["harrisCorner"] >> m_harrisCorner;
    fn["harrisK"]      >> m_harrisK;

    return true;
}

void GFTTFeatureDetector::ApplyParams()
{
    m_gftt->setMaxFeatures   (m_maxFeatures );
    m_gftt->setQualityLevel  (m_qualityLevel);
    m_gftt->setMinDistance   (m_minDistance );
    m_gftt->setBlockSize     (m_blockSize   );
    m_gftt->setHarrisDetector(m_harrisCorner);
    m_gftt->setK             (m_harrisK     );
}

Parameterised::Options GFTTFeatureDetector::GetOptions(int flag)
{
    Options o("GFTT (Good Features To Track) Feature Detection Options");
    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        o.add_options()
            ("max-features",  po::value<int>   (&m_maxFeatures )->default_value(m_gftt->getMaxFeatures()   ), "Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.")
            ("quality-level", po::value<double>(&m_qualityLevel)->default_value(m_gftt->getQualityLevel()  ), "Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response. The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.")
            ("min-distance",  po::value<double>(&m_minDistance )->default_value(m_gftt->getMinDistance()   ), "Minimum possible Euclidean distance between the returned corners.")
            ("block-size",    po::value<int>   (&m_blockSize   )->default_value(m_gftt->getBlockSize()     ), "Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.")
            ("harris-corner", po::value<bool>  (&m_harrisCorner)->default_value(m_gftt->getHarrisDetector()), "Parameter indicating whether to use a Harris detector.")
            ("harris-k",      po::value<double>(&m_harrisK     )->default_value(m_gftt->getK()             ), "Free parameter of the Harris detector.");
    }
    return o;
}

/**
* OpenCV Feature Detection Wrapper of
*  "Features from Accelerated Segment Test (FAST)"
*/

void FASTFeatureDetector::WriteParams(cv::FileStorage& fs) const
{
    fs << "threshold" << m_fast->getThreshold();
    fs << "nonmaxSup" << m_fast->getNonmaxSuppression();
    fs << "neighbour" << m_fast->getType();
}

bool FASTFeatureDetector::ReadParams(const cv::FileNode& fn)
{
    fn["threshold"] >> m_threshold;
    fn["nonmaxSup"] >> m_nonmaxSup;
    fn["neighbour"] >> m_neighbour;

    return true;
}

void FASTFeatureDetector::ApplyParams()
{
    int type = NeighbourCode2Type(m_neighbour);

    m_fast->setThreshold        (m_threshold);
    m_fast->setNonmaxSuppression(m_nonmaxSup);
    m_fast->setType             (type       );
}

Parameterised::Options FASTFeatureDetector::GetOptions(int flag)
{
    int  thresh = m_fast->getThreshold();
    bool nonmax = m_fast->getNonmaxSuppression();
    int  neighb = Type2NeighbourCode(m_fast->getType());

    Options o("FAST (Features from Accelerated Segment Test) Feature Detection Options");

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        o.add_options()
            ("threshold",  po::value<int> (&m_threshold)->default_value(thresh), "Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.")
            ("nonmax-sup", po::value<bool>(&m_nonmaxSup)->default_value(nonmax), "If true, non-maximum suppression is applied to detected corners (keypoints).")
            ("neighbour",  po::value<int> (&m_neighbour)->default_value(neighb), "The neighnourhood code, must be \"916\", \"712\" or \"58\". The code corresponds to the three neighbourhood type defined in the paper, namely TYPE_9_16, TYPE_7_12 and TYPE_5_8.");
    }

    return o;
}

int FASTFeatureDetector::NeighbourCode2Type(int neighbour)
{
    switch (neighbour)
    {
    case 916: return FastFeatureDetector::TYPE_9_16; break;
    case 712: return FastFeatureDetector::TYPE_7_12; break;
    case 58:  return FastFeatureDetector::TYPE_5_8;  break;
    }

    E_ERROR << "unknown neighbourhood type: " << neighbour;

    return FastFeatureDetector::TYPE_9_16;
}

int FASTFeatureDetector::Type2NeighbourCode(int type)
{
    switch (type)
    {
    case FastFeatureDetector::TYPE_9_16: return 916; break;
    case FastFeatureDetector::TYPE_7_12: return 712; break;
    case FastFeatureDetector::TYPE_5_8:  return 58;  break;
    }

    E_ERROR << "unknown FastFeatureDetector type: " << type;

    return Type2NeighbourCode(FastFeatureDetector::TYPE_9_16);
}

 /**
 * OpenCV Feature Detection and Extraction Wrapper of
 *  "Oriented BRIEF (ORB)"
 */

void ORBFeatureDetextractor::WriteParams(cv::FileStorage& fs) const
{
    fs << "maxFeatures"   << m_cvDxtor->getMaxFeatures();
    fs << "scaleFactor"   << m_cvDxtor->getScaleFactor();
    fs << "levels"        << m_cvDxtor->getNLevels();
    fs << "edgeThreshold" << m_cvDxtor->getEdgeThreshold();
    fs << "wtaK"          << m_cvDxtor->getWTA_K();
    fs << "scoreType"     << ScoreType2String(m_cvDxtor->getScoreType());
    fs << "patchSize"     << m_cvDxtor->getPatchSize();
    fs << "fastThreshold" << m_cvDxtor->getFastThreshold();
}

bool ORBFeatureDetextractor::ReadParams(const cv::FileNode& fn)
{
    fn["maxFeatures"]   >> m_maxFeatures;
    fn["scaleFactor"]   >> m_scaleFactor;
    fn["levels"]        >> m_levels;
    fn["edgeThreshold"] >> m_edgeThreshold;
    fn["wtaK"]          >> m_wtaK;
    fn["scoreType"]     >> m_scoreType;
    fn["patchSize"]     >> m_patchSize;
    fn["fastThreshold"] >> m_fastThreshold;

    return true;
}

void ORBFeatureDetextractor::ApplyParams()
{
    m_cvDxtor->setMaxFeatures  (m_maxFeatures);
    m_cvDxtor->setScaleFactor  (m_scaleFactor);
    m_cvDxtor->setNLevels      (m_levels);
    m_cvDxtor->setEdgeThreshold(m_edgeThreshold);
    m_cvDxtor->setWTA_K        (m_wtaK);
    m_cvDxtor->setScoreType    (String2ScoreType(m_scoreType));
    m_cvDxtor->setPatchSize    (m_patchSize);
    m_cvDxtor->setFastThreshold(m_fastThreshold);
}

Parameterised::Options ORBFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        String scoreType = ScoreType2String(m_cvDxtor->getScoreType());
        Options o("ORB (Oriented BRIEF) Feature Detection Options");
        o.add_options()
            ("max-features",   po::value<int>   (&m_maxFeatures  )->default_value(m_cvDxtor->getMaxFeatures()),   "The maximum number of features to retain.")
            ("scale-factor",   po::value<double>(&m_scaleFactor  )->default_value(m_cvDxtor->getScaleFactor()),   "Pyramid decimation ratio, greater than 1. scaleFactor=2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.")
            ("levels",         po::value<int>   (&m_levels       )->default_value(m_cvDxtor->getNLevels()),       "The number of pyramid levels.")
            ("edge-threshold", po::value<int>   (&m_edgeThreshold)->default_value(m_cvDxtor->getEdgeThreshold()), "This is size of the border where the features are not detected.")
            ("wta-k",          po::value<int>   (&m_wtaK         )->default_value(m_cvDxtor->getWTA_K()),         "The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).")
            ("score",          po::value<String>(&m_scoreType    )->default_value(scoreType),                     "The default \"HARRIS\" means that Harris algorithm is used to rank features; \"FAST\" is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.")
            ("fast-threshold", po::value<int>   (&m_fastThreshold)->default_value(m_cvDxtor->getFastThreshold()), "The threshold used by the FAST algorithm when the score is set to \"FAST\"");
        a.add(o);
    }

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        Options o("ORB (Oriented BRIEF) Feature Extraction Option");
        o.add_options()
            ("patch-size",    po::value<int>    (&m_patchSize   )->default_value(m_cvDxtor->getPatchSize()),      "Size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.");
        a.add(o);
    }

    return a;
}

int ORBFeatureDetextractor::String2ScoreType(const seq2map::String& scoreName)
{
    if      (boost::iequals(scoreName, "HARRIS")) return ORB::HARRIS_SCORE;
    else if (boost::iequals(scoreName, "FAST"))   return ORB::FAST_SCORE;

    E_ERROR << "unknown score type string: " << scoreName;

    return ORB::HARRIS_SCORE;
}

seq2map::String ORBFeatureDetextractor::ScoreType2String(int type)
{
    switch (type)
    {
    case ORB::HARRIS_SCORE: return "HARRIS"; break;
    case ORB::FAST_SCORE:   return "FAST";   break;
    }

    E_ERROR << "unknown score type: " << type;

    return ScoreType2String(ORB::HARRIS_SCORE);
}


/**
* OpenCV Feature Detection and Extraction Wrapper of
*  "SIFT (Scale-Invariant Feature Transform)"
*/

void SIFTFeatureDetextractor::WriteParams(cv::FileStorage& fs) const
{
    fs << "maxFeatures"       << m_maxFeatures;
    fs << "levels"            << m_octaveLayers;
    fs << "contrastThreshold" << m_contrastThreshold;
    fs << "edgeThreshold"     << m_edgeThreshold;
    fs << "sigma"             << m_sigma;
}

bool SIFTFeatureDetextractor::ReadParams(const cv::FileNode& fn)
{
    fn["maxFeatures"]       >> m_maxFeatures;
    fn["levels"]            >> m_octaveLayers;
    fn["contrastThreshold"] >> m_contrastThreshold;
    fn["edgeThreshold"]     >> m_edgeThreshold;
    fn["sigma"]             >> m_sigma;
    
    return true;
}

void SIFTFeatureDetextractor::ApplyParams()
{
    SetCvSuperDetextractorPtr(cv::xfeatures2d::SIFT::create(
        m_maxFeatures,
        m_octaveLayers,
        m_contrastThreshold,
        m_edgeThreshold,
        m_sigma
    ));
}

Parameterised::Options SIFTFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        Options o("SIFT (Scale-Invariant Feature Transform) Feature Detection Options");
        o.add_options()
            ("max-features",       po::value<int>   (&m_maxFeatures      )->default_value(m_maxFeatures      ), "The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)")
            ("octave-layers",      po::value<int>   (&m_octaveLayers     )->default_value(m_octaveLayers     ), "The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.")
            ("contrast-threshold", po::value<double>(&m_contrastThreshold)->default_value(m_contrastThreshold), "The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.")
            ("edge-threshold",     po::value<double>(&m_edgeThreshold    )->default_value(m_edgeThreshold    ), "The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).")
            ("sigma",              po::value<double>(&m_sigma            )->default_value(m_sigma            ), "The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.");
        a.add(o);
    }

    return a;
}

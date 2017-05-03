#include <boost/algorithm/string/predicate.hpp>
#include <seq2map/features_opencv.hpp>

using namespace cv;
using namespace seq2map;

namespace po = boost::program_options;

template<class T>
void CvSuperDetextractorAdaptor<T>::SetCvSuperDetextractorPtr(CvDextractorPtr cvDxtor)
{
    SetCvDetectorPtr    (cvDxtor);
    SetCvExtractorPtr   (cvDxtor);
    SetCvDetextractorPtr(cvDxtor);
}

//
// Begin of OpenCV Feature-Specific Implementations
//

/******************************************************************************
* Class:      GFTTFeatureDetector
* Features:   Good Feature to Track, GFTT
* Detection:  Yes
* Extraction: No
* Reference:  
*****************************************************************************/

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
    m_gftt->setMaxFeatures   (m_maxFeatures);
    m_gftt->setQualityLevel  (m_qualityLevel);
    m_gftt->setMinDistance   (m_minDistance);
    m_gftt->setBlockSize     (m_blockSize);
    m_gftt->setHarrisDetector(m_harrisCorner);
    m_gftt->setK             (m_harrisK);
}

Parameterised::Options GFTTFeatureDetector::GetOptions(int flag)
{
    Options o("GFTT (Good Features To Track) detection options");

    if (flag & DETECTION_OPTIONS)
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

/*****************************************************************************
* Class:      FASTFeatureDetector
* Features:   Features from Accelerated Segment Test, FAST
* Detection:  Yes
* Extraction: No
* Reference:  
*****************************************************************************/

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

    m_fast->setThreshold(m_threshold);
    m_fast->setNonmaxSuppression(m_nonmaxSup);
    m_fast->setType(type);
}

Parameterised::Options FASTFeatureDetector::GetOptions(int flag)
{
    int  thresh = m_fast->getThreshold();
    bool nonmax = m_fast->getNonmaxSuppression();
    int  neighb = Type2NeighbourCode(m_fast->getType());

    Options o("FAST (Features from Accelerated Segment Test) detection options");

    if (flag & DETECTION_OPTIONS)
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

/******************************************************************************
* Class:      AGASTFeatureDetector
* Features:   Adaptive and Generic Corner Detection Based on the Accelerated
*             Segment Test, AGAST
* Detection:  Yes
* Extraction: No
* Reference:  
*****************************************************************************/

void AGASTFeatureDetector::WriteParams(cv::FileStorage & fs) const
{
    fs << "threshold" << m_agast->getThreshold();
    fs << "nonmaxSup" << m_agast->getNonmaxSuppression();
    fs << "neighbour" << NeighbourType2String(m_agast->getType());
}

bool AGASTFeatureDetector::ReadParams(const cv::FileNode & fn)
{
    fn["threshold"] >> m_threshold;
    fn["nonmaxSup"] >> m_nonmaxSup;
    fn["neighbour"] >> m_neighbour;

    return true;
}

void AGASTFeatureDetector::ApplyParams()
{
    m_agast->setThreshold        (m_threshold);
    m_agast->setNonmaxSuppression(m_nonmaxSup);
    m_agast->setType             (String2NeighbourType(m_neighbour));
}

Parameterised::Options AGASTFeatureDetector::GetOptions(int flag)
{
    int  thresh = m_agast->getThreshold();
    bool nonmax = m_agast->getNonmaxSuppression();
    seq2map::String neighb = NeighbourType2String(m_agast->getType());

    Options o("AGAST (Adaptive and Generic Corner Detection Based on the Accelerated Segment Test) detection options");

    if (flag & DETECTION_OPTIONS)
    {
        o.add_options()
            ("threshold",  po::value<int>   (&m_threshold)->default_value(thresh), "Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.")
            ("nonmax-sup", po::value<bool>  (&m_nonmaxSup)->default_value(nonmax), "If true, non-maximum suppression is applied to detected corners (keypoints).")
            ("neighbour",  po::value<String>(&m_neighbour)->default_value(neighb), "The neighnourhood type, must be \"58\", \"712d\", \"712s\" or \"916\".");
    }

    return o;
}

int AGASTFeatureDetector::String2NeighbourType(seq2map::String neighbour)
{
    if      (boost::iequals(neighbour, "58"  )) return AgastFeatureDetector::AGAST_5_8;
    else if (boost::iequals(neighbour, "712d")) return AgastFeatureDetector::AGAST_7_12d;
    else if (boost::iequals(neighbour, "712s")) return AgastFeatureDetector::AGAST_7_12s;
    else if (boost::iequals(neighbour, "916" )) return AgastFeatureDetector::OAST_9_16;

    E_ERROR << "unknown neighbourhood type: " << neighbour;

    return AgastFeatureDetector::OAST_9_16;
}

seq2map::String AGASTFeatureDetector::NeighbourType2String(int type)
{
    switch (type)
    {
    case AgastFeatureDetector::AGAST_5_8:   return "58";   break;
    case AgastFeatureDetector::AGAST_7_12d: return "712d"; break;
    case AgastFeatureDetector::AGAST_7_12s: return "712s"; break;
    case AgastFeatureDetector::OAST_9_16:   return "916";  break;
    }

    E_ERROR << "unknown AgastFeatureDetector type: " << type;

    return NeighbourType2String(AgastFeatureDetector::OAST_9_16);
}

/******************************************************************************
* Class:      ORBFeatureDetextractor
* Features:   Oriented BRIEF, ORB
* Detection:  Yes
* Extraction: Yes
* Reference:  
*****************************************************************************/

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

    if (flag & DETECTION_OPTIONS)
    {
        String scoreType = ScoreType2String(m_cvDxtor->getScoreType());
        Options o("ORB (Oriented BRIEF) detection options");
        o.add_options()
            ("max-features",   po::value<int>   (&m_maxFeatures)  ->default_value(m_cvDxtor->getMaxFeatures()),   "The maximum number of features to retain.")
            ("scale-factor",   po::value<double>(&m_scaleFactor)  ->default_value(m_cvDxtor->getScaleFactor()),   "Pyramid decimation ratio, greater than 1. scaleFactor=2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.")
            ("levels",         po::value<int>   (&m_levels)       ->default_value(m_cvDxtor->getNLevels()),       "The number of pyramid levels.")
            ("edge-threshold", po::value<int>   (&m_edgeThreshold)->default_value(m_cvDxtor->getEdgeThreshold()), "This is size of the border where the features are not detected.")
            ("wta-k",          po::value<int>   (&m_wtaK)         ->default_value(m_cvDxtor->getWTA_K()),         "The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).")
            ("score",          po::value<String>(&m_scoreType)    ->default_value(scoreType),                     "The default \"HARRIS\" means that Harris algorithm is used to rank features; \"FAST\" is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.")
            ("fast-threshold", po::value<int>   (&m_fastThreshold)->default_value(m_cvDxtor->getFastThreshold()), "The threshold used by the FAST algorithm when the score is set to \"FAST\"");
        a.add(o);
    }

    if (flag & EXTRACTION_OPTIONS)
    {
        Options o("ORB (Oriented BRIEF) feature extraction option");
        o.add_options()
            ("patch-size",     po::value<int>(&m_patchSize)       ->default_value(m_cvDxtor->getPatchSize()),     "Size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.");
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

/******************************************************************************
* Class:      BRISKFeatureDetextractor
* Features:   Binary Robust Invariant Scalable Keypoints, BRISK
* Detection:  Yes
* Extraction: Yes
* Reference:  
*****************************************************************************/

void BRISKFeatureDetextractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "threshold"    << m_threshold;
    fs << "levels"       << m_levels;
    fs << "patternScale" << m_patternScale;
}

bool BRISKFeatureDetextractor::ReadParams(const cv::FileNode & fn)
{
    fn["threshold"]    >> m_threshold;
    fn["levels"]       >> m_levels;
    fn["patternScale"] >> m_patternScale;

    return true;
}

void BRISKFeatureDetextractor::ApplyParams()
{
    SetCvSuperDetextractorPtr(cv::BRISK::create(
        m_threshold,
        m_levels,
        m_patternScale
        ));
}

Parameterised::Options BRISKFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & DETECTION_OPTIONS)
    {
        Options o("BRISK (Binary Robust Invariant Scalable Keypoints) detection options");
        o.add_options()
            ("threshold",     po::value<int>  (&m_threshold)   ->default_value(m_threshold),    "AGAST detection threshold score.")
            ("levels",        po::value<int>  (&m_levels)      ->default_value(m_levels),       "Number of detection octaves; use 0 to do single scale.")
            ("pattern-scale", po::value<float>(&m_patternScale)->default_value(m_patternScale), "The scale to the applied to the pattern used for sampling the neighbourhood of a keypoint.");
        a.add(o);
    }

    return a;
}

/******************************************************************************
* Class:      KAZEFeatureDetextractor
* Features:   KAZE ("Wind" in Japanese)
* Detection:  Yes
* Extraction: Yes
* Reference:  
*****************************************************************************/

void KAZEFeatureDetextractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "extended"         << m_cvDxtor->getExtended();
    fs << "upright"          << m_cvDxtor->getUpright();
    fs << "threshold"        << m_cvDxtor->getThreshold();
    fs << "levels"           << m_cvDxtor->getNOctaves();
    fs << "octaveLayers"     << m_cvDxtor->getNOctaveLayers();
    fs << "diffusivityType"  << DiffuseType2String(m_cvDxtor->getDiffusivity());
}

bool KAZEFeatureDetextractor::ReadParams(const cv::FileNode & fn)
{
    fn["extended"]          >> 	m_extended;
    fn["upright"]           >> 	m_upright;
    fn["threshold"]         >> 	m_threshold;
    fn["levels"]            >> 	m_levels;
    fn["octaveLayers"]      >> 	m_octaveLayers;
    fn["diffusivityType"]   >>  m_diffusivityType;

    return true;
}

void KAZEFeatureDetextractor::ApplyParams()
{
    int diffuse = String2DiffuseType(m_diffusivityType);

    m_cvDxtor->setExtended     (m_extended);
    m_cvDxtor->setUpright      (m_upright);
    m_cvDxtor->setThreshold    (m_threshold);
    m_cvDxtor->setNOctaves     (m_levels);
    m_cvDxtor->setNOctaveLayers(m_octaveLayers);
    m_cvDxtor->setDiffusivity  (diffuse);
}

Parameterised::Options KAZEFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & DETECTION_OPTIONS)
    {
        String diffuse = DiffuseType2String(m_cvDxtor->getDiffusivity());

        Options o("KAZE detection options");
        o.add_options()
            ("threshold",     po::value<double>(&m_threshold)      ->default_value(m_cvDxtor->getThreshold()),     "Detector response threshold to accept point.")
            ("levels",        po::value<int>   (&m_levels)         ->default_value(m_cvDxtor->getNOctaves()),      "Maximum octave evolution of the image.")
            ("octave-layers", po::value<int>   (&m_octaveLayers)   ->default_value(m_cvDxtor->getNOctaveLayers()), "Default number of sublevels per scale level.")
            ("diffusivity",   po::value<String>(&m_diffusivityType)->default_value(diffuse),                       "Diffusivity type, one of \"PMG1\", \"PMG2\", \"WEICKERT\" and \"CHARBONNIER\".");
        a.add(o);
    }

    if (flag & EXTRACTION_OPTIONS)
    {
        Options o("KAZE extraction options");
        o.add_options()
            ("extended",       po::value<bool>   (&m_extended)        ->default_value(m_cvDxtor->getExtended()),      "Set to enable extraction of extended (128-byte) descriptor.")
            ("upright",        po::value<bool>   (&m_upright)         ->default_value(m_cvDxtor->getUpright()),       "Set to enable use of upright descriptors (non rotation-invariant).");
        a.add(o);
    }

    return a;
}

int KAZEFeatureDetextractor::String2DiffuseType(const seq2map::String & diffuse)
{
    if      (boost::iequals(diffuse, "PMG1"))        return KAZE::DIFF_PM_G1;
    else if (boost::iequals(diffuse, "PMG2"))        return KAZE::DIFF_PM_G2;
    else if (boost::iequals(diffuse, "WEICKERT"))    return KAZE::DIFF_WEICKERT;
    else if (boost::iequals(diffuse, "CHARBONNIER")) return KAZE::DIFF_CHARBONNIER;

    E_ERROR << "unknown diffusivity string: " << diffuse;

    return KAZE::DIFF_PM_G2;
}

seq2map::String KAZEFeatureDetextractor::DiffuseType2String(int type)
{
    switch (type)
    {
    case KAZE::DIFF_PM_G2      : return "PMG2";        break;
    case KAZE::DIFF_PM_G1      : return "PMG1";        break;
    case KAZE::DIFF_WEICKERT   : return "WEICKERT";    break;
    case KAZE::DIFF_CHARBONNIER: return "CHARBONNIER"; break;
    }

    E_ERROR << "unknown diffusivity type: " << type;

    return DiffuseType2String(KAZE::DIFF_PM_G2);
}

/******************************************************************************
* Class:      AKAZEFeatureDetextractor
* Features:   Accelerated KAZE, AKAZE
* Detection:  Yes
* Extraction: Yes
* Reference:  
*****************************************************************************/

void AKAZEFeatureDetextractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "descriptorType"     << DescriptorType2String(m_cvDxtor->getDescriptorType());
    fs << "descriptorBits"     << m_cvDxtor->getDescriptorSize();
    fs << "descriptorChannels" << m_cvDxtor->getDescriptorChannels();
    fs << "threshold"          << m_cvDxtor->getThreshold();
    fs << "levels"             << m_cvDxtor->getNOctaves();
    fs << "octaveLayers"       << m_cvDxtor->getNOctaveLayers();
    fs << "diffusivityType"    << KAZEFeatureDetextractor::DiffuseType2String(m_cvDxtor->getDiffusivity());
}

bool AKAZEFeatureDetextractor::ReadParams(const cv::FileNode & fn)
{
    fn["descriptorType"]     >> m_descriptorType;
    fn["descriptorBits"]     >> m_descriptorBits;
    fn["descriptorChannels"] >> m_descriptorChannels;
    fn["threshold"]          >> m_threshold;
    fn["levels"]             >> m_levels;
    fn["octaveLayers"]       >> m_octaveLayers;
    fn["diffusivityType"]    >> m_diffusivityType;

    return true;
}

void AKAZEFeatureDetextractor::ApplyParams()
{
    m_cvDxtor->setDescriptorType    (String2DescriptorType(m_descriptorType));
    m_cvDxtor->setDescriptorSize    (m_descriptorBits);
    m_cvDxtor->setDescriptorChannels(m_descriptorChannels);
    m_cvDxtor->setThreshold         (m_threshold);
    m_cvDxtor->setNOctaves          (m_levels);
    m_cvDxtor->setNOctaveLayers     (m_octaveLayers);
    m_cvDxtor->setDiffusivity       (KAZEFeatureDetextractor::String2DiffuseType(m_diffusivityType));
}

Parameterised::Options AKAZEFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & DETECTION_OPTIONS)
    {
        String diffuse = KAZEFeatureDetextractor::DiffuseType2String(m_cvDxtor->getDiffusivity());

        Options o("AKAZE (Accelerated KAZE) detection options");
        o.add_options()
            ("threshold",     po::value<double>(&m_threshold)      ->default_value(m_cvDxtor->getThreshold()),     "Detector response threshold to accept point.")
            ("levels",        po::value<int>   (&m_levels)         ->default_value(m_cvDxtor->getNOctaves()),      "Maximum octave evolution of the image.")
            ("octave-layers", po::value<int>   (&m_octaveLayers)   ->default_value(m_cvDxtor->getNOctaveLayers()), "Default number of sublevels per scale level.")
            ("diffusivity",   po::value<String>(&m_diffusivityType)->default_value(diffuse),                       "Diffusivity type, one of \"PMG1\", \"PMG2\", \"WEICKERT\" and \"CHARBONNIER\".");
        a.add(o);
    }

    if (flag & EXTRACTION_OPTIONS)
    {
        String desc = DescriptorType2String(m_cvDxtor->getDescriptorType());
        Options o("AKAZE (Accelerated KAZE) extraction options");
        o.add_options()
            ("descriptor-type",     po::value<String>(&m_descriptorType)    ->default_value(desc),                               "Type of the descriptor: \"KAZE\", \"UPRIGHT-KAZE\", \"MLDB\" or \"UPRIGHT-MLDB\".")
            ("descriptor-bits",     po::value<int>   (&m_descriptorBits)    ->default_value(m_cvDxtor->getDescriptorSize()),     "Size of the descriptor in bits; Set 0 for full size.")
            ("descriptor-channels", po::value<int>   (&m_descriptorChannels)->default_value(m_cvDxtor->getDescriptorChannels()), "Number of channels in the descriptor, 1, 2, or 3.");
        a.add(o);
    }

    return a;
}

int AKAZEFeatureDetextractor::String2DescriptorType(const seq2map::String & descriptor)
{
    if      (boost::iequals(descriptor, "KAZE"))         return AKAZE::DESCRIPTOR_KAZE;
    else if (boost::iequals(descriptor, "UPRIGHT-KAZE")) return AKAZE::DESCRIPTOR_KAZE_UPRIGHT;
    else if (boost::iequals(descriptor, "MLDB"))         return AKAZE::DESCRIPTOR_MLDB;
    else if (boost::iequals(descriptor, "UPRIGHT-MLDB")) return AKAZE::DESCRIPTOR_MLDB_UPRIGHT;

    E_ERROR << "unknown descriptor type string: " << descriptor;

    return AKAZE::DESCRIPTOR_KAZE;
}

seq2map::String AKAZEFeatureDetextractor::DescriptorType2String(int type)
{
    switch (type)
    {
    case AKAZE::DESCRIPTOR_KAZE:         return "KAZE";         break;
    case AKAZE::DESCRIPTOR_KAZE_UPRIGHT: return "UPRIGHT-KAZE"; break;
    case AKAZE::DESCRIPTOR_MLDB:         return "MLDB";         break;
    case AKAZE::DESCRIPTOR_MLDB_UPRIGHT: return "UPRIGHT-MLDB"; break;
    }

    E_ERROR << "unknown descriptor type: " << type;

    return DescriptorType2String(AKAZE::DESCRIPTOR_KAZE);
}
#ifdef WITH_XFEATURES2D
//
// Begin of OpenCV Extended Feature-Specific Implementations
//

/******************************************************************************
* Class:      StarFeatureDetector
* Features:   Center Surrounded Extrema, CenSurE
* Detection:  Yes
* Extraction: No
* Reference:  
*****************************************************************************/

void StarFeatureDetector::WriteParams(cv::FileStorage & fs) const
{
    fs << "maxSize"                << m_maxSize;
    fs << "responseThreshold"      << m_responseThreshold;
    fs << "lineThresholdProjected" << m_lineThresholdProjected;
    fs << "lineThresholdBinarized" << m_lineThresholdBinarized;
    fs << "suppressNonmaxSize"     << m_suppressNonmaxSize;
}

bool seq2map::StarFeatureDetector::ReadParams(const cv::FileNode & fn)
{
    fn["maxSize"]                >> m_maxSize;
    fn["responseThreshold"]      >> m_responseThreshold;
    fn["lineThresholdProjected"] >> m_lineThresholdProjected;
    fn["lineThresholdBinarized"] >> m_lineThresholdBinarized;
    fn["suppressNonmaxSize"]     >> m_suppressNonmaxSize; 

    return true;
}

void StarFeatureDetector::ApplyParams()
{
    SetCvDetectorPtr(cv::xfeatures2d::StarDetector::create(
        m_maxSize,
        m_responseThreshold,
        m_lineThresholdProjected,
        m_lineThresholdBinarized,
        m_suppressNonmaxSize
        ));
}

Parameterised::Options StarFeatureDetector::GetOptions(int flag)
{
    Options a;

    if (flag & DETECTION_OPTIONS)
    {
        Options o("Star Feature (based on CenSurE, or Center Surrounded Extrema) detection options");
        o.add_options()
            ("max-size",                 po::value<int>(&m_maxSize)               ->default_value(m_maxSize),                "The maximum number of filters to be applied."               )
            ("response-threshold",       po::value<int>(&m_responseThreshold)     ->default_value(m_responseThreshold),      "The threshold to eliminate weak corners."                   )
            ("line-threshold-projected", po::value<int>(&m_lineThresholdProjected)->default_value(m_lineThresholdProjected), "Maximum ratio between HARRIS responses, for the first test.")
            ("line-threshold-binarized", po::value<int>(&m_lineThresholdBinarized)->default_value(m_lineThresholdBinarized), "Maximum ratio between HARRIS sizes, for the second test."   )
            ("suppress-nonmax-size",     po::value<int>(&m_suppressNonmaxSize)    ->default_value(m_suppressNonmaxSize),     "Window size to apply the non-maximal suppression."          );
        a.add(o);
    }

    return a; 
}

/******************************************************************************
* Class:      MSDFeatureDetector
* Features:   Maximal Self-Dissimilarity, MSD
* Detection:  Yes
* Extraction: No
* Reference:  
*****************************************************************************/

void MSDFeatureDetector::WriteParams(cv::FileStorage & fs) const
{
    fs << "patchRadius"    <<  m_patchRadius;
    fs << "searchRadius"   <<  m_searchRadius;
    fs << "nmsRadius"      <<  m_nmsRadius;
    fs << "nmsScaleRadius" <<  m_nmsScaleRadius;
    fs << "saliency"       <<  m_saliency;
    fs << "neighbours"     <<  m_neighbours;
    fs << "scaleFactor"    <<  m_scaleFactor;
    fs << "scales"         <<  m_scales;
    fs << "oriented"       <<  m_oriented;
}

bool MSDFeatureDetector::ReadParams(const cv::FileNode & fn)
{
    fn["patchRadius"]    >> m_patchRadius;
    fn["searchRadius"]   >> m_searchRadius;
    fn["nmsRadius"]      >> m_nmsRadius;
    fn["nmsScaleRadius"] >> m_nmsScaleRadius;
    fn["saliency"]       >> m_saliency;
    fn["neighbours"]     >> m_neighbours;
    fn["scaleFactor"]    >> m_scaleFactor;
    fn["scales "]        >> m_scales;
    fn["oriented"]       >> m_oriented;

    return true;
}

void MSDFeatureDetector::ApplyParams()
{
    SetCvDetectorPtr(cv::xfeatures2d::MSDDetector::create(
        m_patchRadius,
        m_searchRadius,
        m_nmsRadius,
        m_nmsScaleRadius,
        m_saliency,
        m_neighbours,
        m_scaleFactor,
        m_scales,
        m_oriented
        ));
}

Parameterised::Options MSDFeatureDetector::GetOptions(int flag)
{
    Options a;

    if (flag & DETECTION_OPTIONS)
    {
        Options o("MSD (Maximal Self-Dissimilarity) detection options");
        o.add_options()
            ("patch-radius",     po::value<int>  (&m_patchRadius)   ->default_value(m_patchRadius),    "")
            ("search-dadius",    po::value<int>  (&m_searchRadius)  ->default_value(m_searchRadius),   "")
            ("nms-radius",       po::value<int>  (&m_nmsRadius)     ->default_value(m_nmsRadius),      "")
            ("nms-scale-radius", po::value<int>  (&m_nmsScaleRadius)->default_value(m_nmsScaleRadius), "")
            ("saliency",         po::value<float>(&m_saliency)      ->default_value(m_saliency),       "")
            ("neighbours",       po::value<int>  (&m_neighbours)    ->default_value(m_neighbours),     "")
            ("scale-factor",     po::value<float>(&m_scaleFactor)   ->default_value(m_scaleFactor),    "")
            ("scales",           po::value<int>  (&m_scales)        ->default_value(m_scales),         "")
            ("oriented",         po::value<bool> (&m_oriented)      ->default_value(m_oriented),       "");
        a.add(o);
    }

    return a; 
}

/******************************************************************************
* Class:      SIFTFeatureDetextractor
* Features:   Scale-Invariant Feature Transform, SIFT
* Detection:  Yes
* Extraction: Yes
* Reference:  
*****************************************************************************/

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
    fn["maxFeatures"]        >> m_maxFeatures;
    fn["levels"]             >> m_octaveLayers;
    fn["contrastThreshold"]  >> m_contrastThreshold;
    fn["edgeThreshold"]      >> m_edgeThreshold;
    fn["sigma"]              >> m_sigma;

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

    if (flag & DETECTION_OPTIONS)
    {
        Options o("SIFT (Scale-Invariant Feature Transform) detection options");
        o.add_options()
            ("max-features",       po::value<int>   (&m_maxFeatures)      ->default_value(m_maxFeatures),       "The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)")
            ("octave-layers",      po::value<int>   (&m_octaveLayers)     ->default_value(m_octaveLayers),      "The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.")
            ("contrast-threshold", po::value<double>(&m_contrastThreshold)->default_value(m_contrastThreshold), "The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.")
            ("edge-threshold",     po::value<double>(&m_edgeThreshold)    ->default_value(m_edgeThreshold),     "The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).")
            ("sigma",              po::value<double>(&m_sigma)            ->default_value(m_sigma),             "The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.");
        a.add(o);
    }

    return a;
}

/******************************************************************************
* Class:      SURFFeatureDetextractor
* Features:   Speeded-Up Robust Features, SURF
* Detection:  Yes
* Extraction: Yes
* Reference:  
*****************************************************************************/

void SURFFeatureDetextractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "hessianThreshold" << m_cvDxtor->getHessianThreshold();
    fs << "levels"           << m_cvDxtor->getNOctaves();
    fs << "octaveLayers"     << m_cvDxtor->getNOctaveLayers();
    fs << "extended"         << m_cvDxtor->getExtended();
    fs << "upright"          << m_cvDxtor->getUpright();
}

bool SURFFeatureDetextractor::ReadParams(const cv::FileNode & fn)
{
    fn["hessianThreshold"] >> m_hessianThreshold;
    fn["levels"]           >> m_levels;
    fn["octaveLayers"]     >> m_octaveLayers;
    fn["extended"]         >> m_extended;
    fn["upright"]          >> m_upright;

    return true;
}

void SURFFeatureDetextractor::ApplyParams()
{
    m_cvDxtor->setHessianThreshold(m_hessianThreshold);
    m_cvDxtor->setNOctaves        (m_levels          );
    m_cvDxtor->setNOctaveLayers   (m_octaveLayers    );
    m_cvDxtor->setExtended        (m_extended        );
    m_cvDxtor->setUpright         (m_upright         );
}

Parameterised::Options SURFFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & DETECTION_OPTIONS)
    {
        Options o("SURF (Speeded-Up Robust Features) detection options");
        o.add_options()
            ("hessian-threshold", po::value<double>(&m_hessianThreshold)->default_value(m_cvDxtor->getHessianThreshold()), "Threshold for hessian keypoint detector used in SURF.")
            ("levels",            po::value<int>   (&m_levels          )->default_value(m_cvDxtor->getNOctaves()        ), "Number of pyramid octaves the keypoint detector will use.")
            ("octave-layers",     po::value<int>   (&m_octaveLayers    )->default_value(m_cvDxtor->getNOctaveLayers()   ), "Number of octave layers within each octave.")
            ("upright",           po::value<bool>  (&m_upright         )->default_value(m_cvDxtor->getUpright()         ), "Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).");
        a.add(o);
    }

    if (flag & EXTRACTION_OPTIONS)
    {
        Options o("SURF (Speeded-Up Robust Features) extraction option");
        o.add_options()
            ("extended",          po::value<bool>  (&m_extended        )->default_value(m_cvDxtor->getExtended()        ), "Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).");
        a.add(o);
    }

    return a;
}


/******************************************************************************
* Class:      BRIEFFeatureExtractor
* Features:   Binary Robust Independent Elementary Features, BRIEF
* Detection:  No
* Extraction: Yes
* Reference:  
*****************************************************************************/

void BRIEFFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "descriptorBytes" << m_descriptorBytes;
    fs << "oriented"        << m_oriented;
}

bool BRIEFFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["descriptorBytes"] >> m_descriptorBytes;
    fn["oriented"]        >> m_oriented;

    return true;
}

void BRIEFFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::BriefDescriptorExtractor::create(
        m_descriptorBytes,
        m_oriented
        ));
}

Parameterised::Options BRIEFFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & EXTRACTION_OPTIONS)
    {
        Options o("BRIEF (Binary Robust Independent Elementary Features) extraction options");
        o.add_options()
            ("bytes",    po::value<int> (&m_descriptorBytes)->default_value(m_descriptorBytes), "Legth of the descriptor in bytes, valid values are: 16, 32 or 64.")
            ("oriented", po::value<bool>(&m_oriented)       ->default_value(m_oriented       ), "Sample patterns using keypoints orientation, disabled by default.");
        a.add(o);
    }

    return a;
}

/******************************************************************************
* Class:      DAISYFeatureExtractor
* Features:   DAISY: A Fast Local Descriptor for Dense Matching
* Detection:  No
* Extraction: Yes
* Reference:  
*****************************************************************************/

void DAISYFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "radius"        << m_radius;
    fs << "iradius"       << m_iradius;
    fs << "iangle"        << m_iangle;
    fs << "ihist"         << m_ihist;
    fs << "normalisation" << m_normType;
    fs << "interp"        << m_interp;
    fs << "oriented"      << m_oriented;
}

bool DAISYFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["radius"]        >> m_radius;
    fn["iradius"]       >> m_iradius;
    fn["iangle"]        >> m_iangle;
    fn["ihist" ]        >> m_ihist;
    fn["normalisation"] >> m_normType;
    fn["interp"]        >> m_interp;
    fn["oriented"]      >> m_oriented;

    return true;
}

void DAISYFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::DAISY::create(
        m_radius,
        m_iradius,
        m_iangle,
        m_ihist,
        String2NormType(m_normType),
        Mat(),
        m_interp,
        m_oriented
        ));
}

Parameterised::Options DAISYFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & EXTRACTION_OPTIONS)
    {
        Options o("DAISY extraction options");
        o.add_options()
            ("radius",        po::value<float> (&m_radius)  ->default_value(m_radius),   "Radius of the descriptor at the initial scale.")
            ("iradius",       po::value<int>   (&m_iradius) ->default_value(m_iradius),  "Amount of radial range division quantity.")
            ("iangle",        po::value<int>   (&m_iangle)  ->default_value(m_iangle),   "Amount of angular range division quantity.")
            ("ihist",         po::value<int>   (&m_ihist)   ->default_value(m_ihist),    "Amount of gradient orientations range division quantity.")
            ("normalisation", po::value<String>(&m_normType)->default_value(m_normType), "Choose descriptors normalization type, where \"NONE\" will not do any normalization, \"PARTIAL\" means that histograms are normalized independently for L2 norm equal to 1.0, \"FULL\" means that descriptors are normalized for L2 norm equal to 1.0, and \"SIFT\" means that descriptors are normalized for L2 norm equal to 1.0 but no individual one is bigger than 0.154 as in SIFT")
            ("interp",        po::value<bool>  (&m_interp)  ->default_value(m_interp),   "Switch to disable interpolation for speed improvement at minor quality loss.")
            ("oriented",      po::value<bool>  (&m_oriented)->default_value(m_oriented), "Sample patterns using keypoints orientation.");
        a.add(o);
    }

    return a; 
}

int DAISYFeatureExtractor::String2NormType(seq2map::String normType)
{
    if      (boost::iequals(normType, "NONE"))    return cv::xfeatures2d::DAISY::NRM_NONE;
    else if (boost::iequals(normType, "PARTIAL")) return cv::xfeatures2d::DAISY::NRM_PARTIAL;
    else if (boost::iequals(normType, "FULL"))    return cv::xfeatures2d::DAISY::NRM_FULL;
    else if (boost::iequals(normType, "SIFT"))    return cv::xfeatures2d::DAISY::NRM_SIFT;

    E_ERROR << "unknown normalisation type string: " << normType;

    return cv::xfeatures2d::DAISY::NRM_NONE;
}

seq2map::String DAISYFeatureExtractor::NormType2String(int type)
{
    switch (type)
    {
    case cv::xfeatures2d::DAISY::NRM_NONE:    return "NONE";    break;
    case cv::xfeatures2d::DAISY::NRM_PARTIAL: return "PARTIAL"; break;
    case cv::xfeatures2d::DAISY::NRM_FULL:    return "FULL";    break;
    case cv::xfeatures2d::DAISY::NRM_SIFT:    return "SIFT";    break;
    }

    E_ERROR << "unknown normalisation type: " << type;

    return NormType2String(cv::xfeatures2d::DAISY::NRM_NONE);
}

/******************************************************************************
* Class:      FREAKFeatureExtractor
* Features:   Fast REtinA Keypoint, FREAK
* Detection:  No
* Extraction: Yes
* Reference:  
*****************************************************************************/

void FREAKFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "levels"               << m_levels;
    fs << "patternScale"         << m_patternScale;
    fs << "normaliseScale"       << m_normaliseScale;
    fs << "normaliseOrientation" << m_normaliseOrientation;
}

bool FREAKFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["levels"]               >> m_levels;
    fn["patternScale"]         >> m_patternScale;
    fn["normaliseScale"]       >> m_normaliseScale;
    fn["normaliseOrientation"] >> m_normaliseOrientation;

    return true;
}

void FREAKFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::FREAK::create(
        m_normaliseOrientation,
        m_normaliseScale,
        m_patternScale,
        m_levels
        ));
}

Parameterised::Options FREAKFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & EXTRACTION_OPTIONS)
    {
        Options o("FREAK (Fast REtinA Keypoint) extraction options");
        o.add_options()
            ("levels",                po::value<int>  (&m_levels)              ->default_value(m_levels),               "Number of octaves covered by the detected keypoints.")
            ("pattern-scale",         po::value<float>(&m_patternScale)        ->default_value(m_patternScale),         "Scaling of the description pattern.")
            ("normalise-scale",       po::value<bool> (&m_normaliseScale)      ->default_value(m_normaliseScale),       "Enable scale normalization.")
            ("normalise-orientation", po::value<bool> (&m_normaliseOrientation)->default_value(m_normaliseOrientation), "Enable orientation normalization.");
        a.add(o);
    }

    return a;
}

/******************************************************************************
* Class:      LATCHFeatureExtractor
* Features:   Learned Arrangements of Three Patch, LATCH
* Detection:  No
* Extraction: Yes
* Reference:  
*****************************************************************************/

void LATCHFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "bytess"            << m_bytes;
    fs << "rotationInvariance"<< m_rotationInvariance;
    fs << "halfSSDSize "      << m_halfSSDSize;
}

bool seq2map::LATCHFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["bytess"]             >> m_bytes;
    fn["rotationInvariance"] >> m_rotationInvariance;
    fn["halfSSDSize"]        >> m_halfSSDSize;

    return true;
}

void LATCHFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::LATCH::create(
        m_bytes,
        m_rotationInvariance,
        m_halfSSDSize
        ));
}

Parameterised::Options LATCHFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & EXTRACTION_OPTIONS)
    {
        Options o("LATCH extraction options");
        o.add_options()
            ("bytes",               po::value<int> (&m_bytes)             ->default_value(m_bytes),             "The size of the descriptor, can be 64, 32, 16, 8, 4, 2 or 1.")
            ("rotation-invariance", po::value<bool>(&m_rotationInvariance)->default_value(m_rotationInvariance),"Whether or not the descriptor should compansate for orientation changes.")
            ("half-ssd-size",       po::value<int> (&m_halfSSDSize)       ->default_value(m_halfSSDSize),       "The size of half of the mini-patches size. For example, if we would like to compare triplets of patches of size 7x7 then the value should be (7-1)/2 = 3.");
        a.add(o);
    }

    return a; 
}

/******************************************************************************
* Class:      LUCIDFeatureExtractor
* Features:   Locally Uniform Comparison Image Descriptor, LUCID
* Detection:  No
* Extraction: Yes
* Reference:  
*****************************************************************************/

void LUCIDFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "lucidKernel" << m_lucidKernel;
    fs << "blurKernel"  << m_blurKernel;
}

bool seq2map::LUCIDFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["lucidKernel"] >> m_lucidKernel;
    fn["blurKernel"]  >> m_blurKernel;

    return true;
}

void seq2map::LUCIDFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::LUCID::create(
        m_lucidKernel,
        m_blurKernel
        ));
}

Parameterised::Options LUCIDFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & DETECTION_OPTIONS)
    {
        Options o("LUCID (Locally Uniform Comparison Image Descriptor) extraction options");
        o.add_options()
            ("lucid-kernel", po::value<int> (&m_lucidKernel)->default_value(m_lucidKernel), "Kernel for descriptor construction.")
            ("blur-kernel",  po::value<int> (&m_blurKernel) ->default_value(m_blurKernel),  "Kernel for blurring image prior to descriptor construction.");
        a.add(o);
    }

    return a; 
}
#endif // WITH_XFEATURES2D


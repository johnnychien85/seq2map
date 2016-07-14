#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <seq2map/features.hpp>

using namespace cv;
using namespace seq2map;

namespace po = boost::program_options;

const seq2map::String HetergeneousDetextractor::s_detectorFileNodeName = "detection";
const seq2map::String HetergeneousDetextractor::s_extractorFileNodeName = "extraction";
const seq2map::String ImageFeatureSet::s_fileMagicNumber = "IMKPTDSC"; // which means IMage KeyPoinTs & DeSCriptors
const seq2map::String ImageFeatureSet::s_fileHeaderSep = " ";

FeatureDetectorFactory::FeatureDetectorFactory()
{
	Factory::Register<FASTFeatureDetector>("FAST");
	Factory::Register<GFTTFeatureDetector>("GFTT");
    Factory::Register<AGASTFeatureDetector>("AGAST");
	Factory::Register<ORBFeatureDetextractor>("ORB");
	Factory::Register<BRISKFeatureDetextractor>("BRISK");
    Factory::Register<KAZEFeatureDetextractor> ( "KAZE"  );
    Factory::Register<AKAZEFeatureDetextractor>( "AKAZE" );
#ifdef HAVE_XFEATURES2D // non-free feature detectors .....
	Factory::Register<SIFTFeatureDetextractor>("SIFT");
	Factory::Register<SURFFeatureDetextractor> ("SURF");
    Factory::Register<STARFeatureDetector>("STAR");
    Factory::Register<MSDDFeatureDetector>("MSDD");
#endif // HAVE_XFEATURES2D ................................
}

FeatureExtractorFactory::FeatureExtractorFactory()
{
	Factory::Register<ORBFeatureDetextractor>("ORB");
	Factory::Register<BRISKFeatureDetextractor>( "BRISK" );
    Factory::Register<KAZEFeatureDetextractor> ( "KAZE"  );
    Factory::Register<AKAZEFeatureDetextractor>( "AKAZE" );
#ifdef HAVE_XFEATURES2D // non-free descriptor extractors..
	Factory::Register<SIFTFeatureDetextractor>("SIFT");
	Factory::Register<SURFFeatureDetextractor>( "SURF"  );
	Factory::Register<BRIEFFeatureExtractor>  ( "BRIEF" );
    Factory::Register<FREAKFeatureExtractor>  ( "FREAK" );
    Factory::Register<LUCIDFeatureExtractor>  ("LUCID");
    Factory::Register<DAISYFeatureExtractor>  ("DAISY");
    Factory::Register<LATCHFeatureExtractor>  ("LATCH");
#endif // HAVE_XFEATURES2D ................................
}

FeatureDetextractorFactory::FeatureDetextractorFactory()
{
	Factory::Register<ORBFeatureDetextractor>("ORB");
	Factory::Register<BRISKFeatureDetextractor>( "BRISK" );
	Factory::Register<KAZEFeatureDetextractor> ( "KAZE"  );
	Factory::Register<AKAZEFeatureDetextractor>( "AKAZE" );
#ifdef HAVE_XFEATURES2D // non-free feature detextractors..
	Factory::Register<SIFTFeatureDetextractor>("SIFT");
	Factory::Register<SURFFeatureDetextractor> ( "SURF"  );
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
	of << "CV3" << s_fileHeaderSep;
	of << NormType2String(m_normType) << s_fileHeaderSep;
	of << MatType2String(m_descriptors.type()) << s_fileHeaderSep;

	of.write((char*)&m_descriptors.rows, sizeof m_descriptors.rows);
	of.write((char*)&m_descriptors.cols, sizeof m_descriptors.cols);

	// the key points section
	BOOST_FOREACH(const KeyPoint& kp, m_keypoints)
	{
		of.write((char*)&kp.pt.x, sizeof kp.pt.x);
		of.write((char*)&kp.pt.y, sizeof kp.pt.y);
		of.write((char*)&kp.response, sizeof kp.response);
		of.write((char*)&kp.octave, sizeof kp.octave);
		of.write((char*)&kp.angle, sizeof kp.angle);
		of.write((char*)&kp.size, sizeof kp.size);
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

		if (!detector) E_ERROR << "feature detector unknown: " << detectorName;
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
	return m_detector->ReadParams(fn[s_detectorFileNodeName]) &&
		m_extractor->ReadParams(fn[s_extractorFileNodeName]);
}

void HetergeneousDetextractor::ApplyParams()
{
	m_detector->ApplyParams();
	m_extractor->ApplyParams();
}

Parameterised::Options HetergeneousDetextractor::GetOptions(int flag)
{
	if (flag & FeatureOptionType::DETECTION_OPTIONS)  return m_detector->GetOptions(flag);
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
	fs << "maxFeatures" << m_gftt->getMaxFeatures();
	fs << "qualityLevel" << m_gftt->getQualityLevel();
	fs << "minDistance" << m_gftt->getMinDistance();
	fs << "blockSize" << m_gftt->getBlockSize();
	fs << "harrisCorner" << m_gftt->getHarrisDetector();
	fs << "harrisK" << m_gftt->getK();
}

bool GFTTFeatureDetector::ReadParams(const cv::FileNode& fn)
{
	fn["maxFeatures"] >> m_maxFeatures;
	fn["qualityLevel"] >> m_qualityLevel;
	fn["minDistance"] >> m_minDistance;
	fn["blockSize"] >> m_blockSize;
	fn["harrisCorner"] >> m_harrisCorner;
	fn["harrisK"] >> m_harrisK;

	return true;
}

void GFTTFeatureDetector::ApplyParams()
{
	m_gftt->setMaxFeatures(m_maxFeatures);
	m_gftt->setQualityLevel(m_qualityLevel);
	m_gftt->setMinDistance(m_minDistance);
	m_gftt->setBlockSize(m_blockSize);
	m_gftt->setHarrisDetector(m_harrisCorner);
	m_gftt->setK(m_harrisK);
}

Parameterised::Options GFTTFeatureDetector::GetOptions(int flag)
{
	Options o("GFTT (Good Features To Track) Feature Detection Options");
	if (flag & FeatureOptionType::DETECTION_OPTIONS)
	{
		o.add_options()
			("max-features",  po::value<int>   (&m_maxFeatures) ->default_value(m_gftt->getMaxFeatures()),    "Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.")
			("quality-level", po::value<double>(&m_qualityLevel)->default_value(m_gftt->getQualityLevel()),   "Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response. The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.")
			("min-distance",  po::value<double>(&m_minDistance) ->default_value(m_gftt->getMinDistance()),    "Minimum possible Euclidean distance between the returned corners.")
			("block-size",    po::value<int>   (&m_blockSize)   ->default_value(m_gftt->getBlockSize()),      "Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.")
			("harris-corner", po::value<bool>  (&m_harrisCorner)->default_value(m_gftt->getHarrisDetector()), "Parameter indicating whether to use a Harris detector.")
			("harris-k",      po::value<double>(&m_harrisK)     ->default_value(m_gftt->getK()),              "Free parameter of the Harris detector.");
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

	m_fast->setThreshold(m_threshold);
	m_fast->setNonmaxSuppression(m_nonmaxSup);
	m_fast->setType(type);
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
	m_cvDxtor->setMaxFeatures(m_maxFeatures);
	m_cvDxtor->setScaleFactor(m_scaleFactor);
	m_cvDxtor->setNLevels(m_levels);
	m_cvDxtor->setEdgeThreshold(m_edgeThreshold);
	m_cvDxtor->setWTA_K(m_wtaK);
	m_cvDxtor->setScoreType(String2ScoreType(m_scoreType));
	m_cvDxtor->setPatchSize(m_patchSize);
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
			("max-features",   po::value<int>   (&m_maxFeatures)  ->default_value(m_cvDxtor->getMaxFeatures()),   "The maximum number of features to retain.")
			("scale-factor",   po::value<double>(&m_scaleFactor)  ->default_value(m_cvDxtor->getScaleFactor()),   "Pyramid decimation ratio, greater than 1. scaleFactor=2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.")
			("levels",         po::value<int>   (&m_levels)       ->default_value(m_cvDxtor->getNLevels()),       "The number of pyramid levels.")
			("edge-threshold", po::value<int>   (&m_edgeThreshold)->default_value(m_cvDxtor->getEdgeThreshold()), "This is size of the border where the features are not detected.")
			("wta-k",          po::value<int>   (&m_wtaK)         ->default_value(m_cvDxtor->getWTA_K()),         "The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).")
			("score",          po::value<String>(&m_scoreType)    ->default_value(scoreType),                     "The default \"HARRIS\" means that Harris algorithm is used to rank features; \"FAST\" is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.")
			("fast-threshold", po::value<int>   (&m_fastThreshold)->default_value(m_cvDxtor->getFastThreshold()), "The threshold used by the FAST algorithm when the score is set to \"FAST\"");
		a.add(o);
	}

	if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
	{
		Options o("ORB (Oriented BRIEF) Feature Extraction Option");
		o.add_options()
			("patch-size", po::value<int>(&m_patchSize)->default_value(m_cvDxtor->getPatchSize()), "Size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.");
		a.add(o);
	}

	return a;
}

int ORBFeatureDetextractor::String2ScoreType(const seq2map::String& scoreName)
{
	if (boost::iequals(scoreName, "HARRIS")) return ORB::HARRIS_SCORE;
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

	if (flag & FeatureOptionType::DETECTION_OPTIONS)
	{
		Options o("SIFT (Scale-Invariant Feature Transform) Feature Detection Options");
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

	if (flag & FeatureOptionType::DETECTION_OPTIONS)
	{
        Options o("SURF (Speeded up robust features) Feature Detection Options");
		o.add_options()
			("hessian-threshold", po::value<double>(&m_hessianThreshold)->default_value(m_cvDxtor->getHessianThreshold()), "Threshold for hessian keypoint detector used in SURF.")
			("levels",            po::value<int>   (&m_levels          )->default_value(m_cvDxtor->getNOctaves()        ), "Number of pyramid octaves the keypoint detector will use.")
			("octave-layers",     po::value<int>   (&m_octaveLayers    )->default_value(m_cvDxtor->getNOctaveLayers()   ), "Number of octave layers within each octave.")
			("upright",           po::value<bool>  (&m_upright         )->default_value(m_cvDxtor->getUpright()         ), "Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).");
        a.add(o);
    }

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        Options o("SURF (Speeded up robust features) Feature Extraction Option");
        o.add_options()
            ("extended",          po::value<bool>  (&m_extended        )->default_value(m_cvDxtor->getExtended()        ), "Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).");
        a.add(o);
    }

	return a;
}

void AKAZEFeatureDetextractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "descriptorType"       << ScoreType2String(m_cvDxtor->getDescriptorType());
    fs << "descriptorSize"       <<  m_cvDxtor->getDescriptorSize();
    fs << "m_descriptorChannels" <<  m_cvDxtor->getDescriptorChannels();
    fs << "m_threshold"          <<  m_cvDxtor->getThreshold();
    fs << "m_levels"             <<  m_cvDxtor->getNOctaves();
    fs << "m_octaveLayers"       <<  m_cvDxtor->getNOctaveLayers();
    fs << "m_diffusivityType"    << KazeScoreType2String(m_cvDxtor->getDiffusivity());
}

bool AKAZEFeatureDetextractor::ReadParams(const cv::FileNode & fn)
{
    fn["descriptorType"] >>      m_descriptorType;
    fn["descriptorType"] >>      m_descriptorType;
    fn["descriptorSize"] >>      m_descriptorSize;
    fn["descriptorChannels"] >>  m_descriptorChannels;
    fn["threshold"] >>           m_threshold;
    fn["levels"] >>              m_levels;
    fn["octaveLayers"] >>        m_octaveLayers;
    fn["diffusivityType "] >>    m_diffusivityType;
    return true;
}

void AKAZEFeatureDetextractor::ApplyParams()
{
    m_cvDxtor->setDescriptorType(String2ScoreType(m_descriptorType));
    m_cvDxtor->setDescriptorSize( m_descriptorSize);
    m_cvDxtor->setDescriptorChannels(m_descriptorChannels);
    m_cvDxtor->setThreshold(m_threshold);
    m_cvDxtor->setNOctaves(m_levels);
    m_cvDxtor->setNOctaveLayers(m_octaveLayers);
    m_cvDxtor->setDiffusivity(String2KazeScoreType(m_diffusivityType));
}

Parameterised::Options AKAZEFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        String diffusivityType = KazeScoreType2String(m_cvDxtor->getDiffusivity());
        Options o("AKAZE () Feature Detection Options");
        o.add_options()
            ("threshold",      po::value<float> (&m_threshold)      ->default_value( m_cvDxtor->getThreshold()), "Detector response threshold to accept point.")
            ("levels",         po::value<int>   (&m_levels)         ->default_value( m_cvDxtor->getThreshold()), "Maximum octave evolution of the image.")
            ("octave-layers",  po::value<int>   (&m_octaveLayers)   ->default_value( m_cvDxtor->getThreshold()), "Default number of sublevels per scale level.")
            ("diffusivity-type",po::value<String>(&m_diffusivityType)->default_value(diffusivityType),        "Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER.");
          a.add(o);
    }

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        String  descriptorType = ScoreType2String( m_cvDxtor->getDescriptorType());
        Options o("AKAZE ( ) Feature Extraction Option");
        o.add_options()
            ("descriptor-type",     po::value<String>(&m_descriptorType)    ->default_value(descriptorType),                   "Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.")
            ("descriptor-size",     po::value<int>   (&m_descriptorSize)    ->default_value( m_cvDxtor->getDescriptorSize()),     "Size of the descriptor in bits. 0 -> Full size.")
            ("descriptor-channels", po::value<int>   (&m_descriptorChannels)->default_value( m_cvDxtor->getDescriptorChannels()), "	Number of channels in the descriptor (1, 2, 3).");
        a.add(o);
    }

    return a;
}

int AKAZEFeatureDetextractor::String2ScoreType(const seq2map::String & scoreName)
{
    if (boost::iequals(scoreName, "KAZE"))            return AKAZE::DESCRIPTOR_KAZE;
    else if (boost::iequals(scoreName, "KUPRIGHT"))   return AKAZE::DESCRIPTOR_KAZE_UPRIGHT;
    else if (boost::iequals(scoreName, "MLDB"))       return AKAZE::DESCRIPTOR_MLDB;
    else if (boost::iequals(scoreName, "MUPRIGHT"))   return AKAZE::DESCRIPTOR_MLDB_UPRIGHT;

    E_ERROR << "unknown score type string: " << scoreName;

    return AKAZE::DESCRIPTOR_KAZE;
}

seq2map::String AKAZEFeatureDetextractor::ScoreType2String(int type)
{
    switch (type)
    {
    case AKAZE::DESCRIPTOR_KAZE:         return "KAZE";     break;
    case AKAZE::DESCRIPTOR_KAZE_UPRIGHT: return "KUPRIGHT"; break;
    case AKAZE::DESCRIPTOR_MLDB:         return "MLDB";     break;
    case AKAZE::DESCRIPTOR_MLDB_UPRIGHT: return "MUPRIGHT"; break;
    }

    E_ERROR << "unknown score type: " << type;

    return ScoreType2String(AKAZE::DESCRIPTOR_KAZE);
}

int AKAZEFeatureDetextractor::String2KazeScoreType(const seq2map::String & scoreName)
{
    if (boost::iequals(scoreName, "G2"))                 return KAZE::DIFF_PM_G2;
    else if (boost::iequals(scoreName, "G1"))            return KAZE::DIFF_PM_G1;
    else if (boost::iequals(scoreName, "WEICKERT"))      return KAZE::DIFF_WEICKERT;
    else if (boost::iequals(scoreName, "CHARBONNIER"))   return KAZE::DIFF_CHARBONNIER;
    
    E_ERROR << "unknown score type string: " << scoreName;

    return KAZE::DIFF_PM_G2;
}

seq2map::String AKAZEFeatureDetextractor::KazeScoreType2String(int type)
{
    switch (type)
    {
    case KAZE::DIFF_PM_G2         :return "G2";            break;
    case KAZE::DIFF_PM_G1         :return "G1";            break;
    case KAZE::DIFF_WEICKERT      :return "WEICKERT";      break;
    case KAZE::DIFF_CHARBONNIER   :return "CHARBONNIER";   break;
    }

    E_ERROR << "unknown score type: " << type;

    return KazeScoreType2String(KAZE::DIFF_PM_G2);
}

void KAZEFeatureDetextractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "extended"         <<  m_cvDxtor->getExtended();
    fs << "upright"          <<  m_cvDxtor->getUpright();
    fs << "threshold"        <<  m_cvDxtor->getThreshold();
    fs << "levels"           <<  m_cvDxtor->getNOctaves();
    fs << "octaveLayers"     <<  m_cvDxtor->getNOctaveLayers();
    fs << "diffusivityType"  << ScoreType2String( m_cvDxtor->getDiffusivity());
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
    m_cvDxtor->setExtended(m_extended);
    m_cvDxtor->setUpright(m_upright);
    m_cvDxtor->setThreshold(m_threshold);
    m_cvDxtor->setNOctaves(m_levels);
    m_cvDxtor->setNOctaveLayers(m_octaveLayers);
    m_cvDxtor->setDiffusivity(String2ScoreType(m_diffusivityType));
}

Parameterised::Options KAZEFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        String diffusivityType = ScoreType2String( m_cvDxtor->getDiffusivity());
        Options o("AKAZE () Feature Detection Options");
        o.add_options()
            ("threshold",       po::value<float> (&m_threshold)      ->default_value( m_cvDxtor->getThreshold()),     "Detector response threshold to accept point.")
            ("levels",          po::value<int>   (&m_levels)         ->default_value( m_cvDxtor->getNOctaves()),      "Maximum octave evolution of the image.")
            ("octave-layers",   po::value<int>   (&m_octaveLayers)   ->default_value( m_cvDxtor->getNOctaveLayers()), "Default number of sublevels per scale level.")
            ("diffusivity-type",po::value<String>(&m_diffusivityType)->default_value(diffusivityType),            "Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER.");
        a.add(o);
    }

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        Options o("AKAZE ( ) Feature Extraction Option");
        o.add_options()
            ("extended",   po::value<bool> (&m_extended)->default_value( m_cvDxtor->getExtended()), "Set to enable extraction of extended (128-byte) descriptor.")
            ("upright",    po::value<bool> (&m_upright) ->default_value( m_cvDxtor->getUpright()),  "Set to enable use of upright descriptors (non rotation-invariant).");
        a.add(o);
    }

    return a;
}

int KAZEFeatureDetextractor::String2ScoreType(const seq2map::String & scoreName)
{
    if (boost::iequals(scoreName, "G2"))                 return KAZE::DIFF_PM_G2;
    else if (boost::iequals(scoreName, "G1"))            return KAZE::DIFF_PM_G1;
    else if (boost::iequals(scoreName, "WEICKERT"))      return KAZE::DIFF_WEICKERT;
    else if (boost::iequals(scoreName, "CHARBONNIER"))   return KAZE::DIFF_CHARBONNIER;

    E_ERROR << "unknown score type string: " << scoreName;

    return KAZE::DIFF_PM_G2;
}

seq2map::String KAZEFeatureDetextractor::ScoreType2String(int type)
{
    switch (type)
    {
    case KAZE::DIFF_PM_G2         :return "G2";            break;
    case KAZE::DIFF_PM_G1         :return "G1";            break;
    case KAZE::DIFF_WEICKERT      :return "WEICKERT";      break;
    case KAZE::DIFF_CHARBONNIER   :return "CHARBONNIER";   break;
    }

    E_ERROR << "unknown score type: " << type;

    return ScoreType2String(KAZE::DIFF_PM_G2);
}

void BRISKFeatureDetextractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "threshold"    << m_threshold;
    fs << "octaves"       << m_octaves;
    fs << "patternScale" << m_patternScale;
}

bool BRISKFeatureDetextractor::ReadParams(const cv::FileNode & fn)
{
    fn["threshold"]     >> m_threshold;
    fn["octaves"]       >> m_octaves;
    fn["patternScale"]  >> m_patternScale;

    return true;
}

void BRISKFeatureDetextractor::ApplyParams()
{
    SetCvSuperDetextractorPtr(cv::BRISK::create(
        m_threshold,
        m_octaves,
        m_patternScale
    ));
}

Parameterised::Options BRISKFeatureDetextractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        Options o("BRISK (Binary Robust Invariant Scalable Keypoints) Feature Detection Options");
        o.add_options()
            ("threshold",     po::value<int>  (&m_threshold)   ->default_value(m_threshold),    "AGAST detection threshold score.")
            ("octaves",       po::value<int>  (&m_octaves)     ->default_value(m_octaves),      "detection octaves. Use 0 to do single scale.")
            ("m_patternScale",po::value<float>(&m_patternScale)->default_value(m_patternScale), "apply this scale to the pattern used for sampling the neighbourhood of a keypoint.");
            a.add(o);
    }

    return a;
}

void BRIEFFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "descriptorLegth" << m_descriptorLegth;
    fs << "orientation"     << m_orientation;
}

bool BRIEFFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["descriptorLegth"]  >> m_descriptorLegth;
    fn["orientation"]      >> m_orientation;
    return true;
}

void BRIEFFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::BriefDescriptorExtractor::create(
        m_descriptorLegth,
        m_orientation
    ));
}

Parameterised::Options BRIEFFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        Options o("BRIEF () Feature Extraction Options");
        o.add_options()
            ("threshold", po::value<int> (&m_descriptorLegth)->default_value(m_descriptorLegth), "	legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .")
            ("octaves",   po::value<bool>(&m_orientation)->default_value(m_orientation),         "sample patterns using keypoints orientation, disabled by default.");
            a.add(o);
    }

    return a;
}

void AGASTFeatureDetector::WriteParams(cv::FileStorage & fs) const
{
    fs << "threshold" << m_agast->getThreshold();
    fs << "nonmaxSup" << m_agast->getNonmaxSuppression();
    fs << "neighbour" << m_agast->getType();
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
    int type = NeighbourCode2Type(m_neighbour);

    m_agast->setThreshold(m_threshold);
    m_agast->setNonmaxSuppression(m_nonmaxSup);
    m_agast->setType(type);
}

Parameterised::Options AGASTFeatureDetector::GetOptions(int flag)
{
    int  thresh = m_agast->getThreshold();
    bool nonmax = m_agast->getNonmaxSuppression();
    int  neighb = Type2NeighbourCode(m_agast->getType());

    Options o("AGAST (Features from Accelerated Segment Test) Feature Detection Options");

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        o.add_options()
            ("threshold",  po::value<int> (&m_threshold)->default_value(thresh), "Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.")
            ("nonmax-sup", po::value<bool>(&m_nonmaxSup)->default_value(nonmax), "If true, non-maximum suppression is applied to detected corners (keypoints).")
            ("neighbour",  po::value<int> (&m_neighbour)->default_value(neighb), "The neighnourhood code, must be \"916\", \"7120\", \"7121\" or \"58\". The code corresponds to the three neighbourhood type defined in the paper, namely OAST_9_16, AGAST_7_12d, AGAST_7_12s and AGAST_5_8.");
    }

    return o;
}

int AGASTFeatureDetector::NeighbourCode2Type(int neighbour)
{
    switch (neighbour)
    {
    case 58:   return AgastFeatureDetector::AGAST_5_8  ; break;
    case 7120: return AgastFeatureDetector::AGAST_7_12d; break;
    case 7121: return AgastFeatureDetector::AGAST_7_12s; break;
    case 916:  return AgastFeatureDetector::OAST_9_16  ; break;
    }

    E_ERROR << "unknown neighbourhood type: " << neighbour;

    return AgastFeatureDetector::OAST_9_16;
}

int AGASTFeatureDetector::Type2NeighbourCode(int type)
{
    switch (type)
    {
    case AgastFeatureDetector::AGAST_5_8:   return 58;   break;
    case AgastFeatureDetector::AGAST_7_12d: return 7120; break;
    case AgastFeatureDetector::AGAST_7_12s: return 7121; break;
    case AgastFeatureDetector::OAST_9_16:   return 916;  break;
    }

    E_ERROR << "unknown AgastFeatureDetector type: " << type;

    return Type2NeighbourCode(AgastFeatureDetector::OAST_9_16);

}

void FREAKFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "orientation"   << m_orientation;
    fs << "scale"         << m_scale;
    fs << "patternScale"  << m_patternScale;
    fs << "nctaves"       << m_nctaves;
}

bool FREAKFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["orientation"]   >> m_orientation;
    fn["scale"]         >> m_scale;
    fn["patternScale"]  >> m_patternScale;
    fn["nctaves"]       >> m_nctaves;

    return true;
}

void FREAKFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::FREAK::create(
        m_orientation,
        m_scale,
        m_patternScale,
        m_nctaves
    ));
}

Parameterised::Options FREAKFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        Options o("BRIEF () Feature Extraction Options");
        o.add_options()
            ("orientation-normalized", po::value<bool>(&m_orientation)->default_value(m_orientation), "Enable orientation normalization.")
            ("scale-normalized", po::value<bool>(&m_scale)->default_value(m_scale), "Enable scale normalization.")
            ("pattern-scale", po::value<float>(&m_patternScale)->default_value(m_patternScale), "Scaling of the description pattern.")
            ("nctaves", po::value<int>(&m_nctaves)->default_value(m_nctaves), "Number of octaves covered by the detected keypoints.");
             a.add(o);
    }

    return a;
}

void STARFeatureDetector::WriteParams(cv::FileStorage & fs) const
{
    fs << "maxSize"                << m_maxSize;
    fs << "responseThreshold"      << m_responseThreshold;
    fs << "lineThresholdProjected" << m_lineThresholdProjected;
    fs << "lineThresholdBinarized" << m_lineThresholdBinarized;
    fs << "suppressNonmaxSize"     << m_suppressNonmaxSize;
}

bool seq2map::STARFeatureDetector::ReadParams(const cv::FileNode & fn)
{
    fn["maxSize"]               >> m_maxSize;
    fn["responseThreshol"]       >> m_responseThreshold;
    fn["lineThresholdProjected"] >> m_lineThresholdProjected;
    fn["lineThresholdBinarized"] >> m_lineThresholdBinarized;
    fn["suppressNonmaxSize"]     >> m_suppressNonmaxSize; 

    return true;
}

void STARFeatureDetector::ApplyParams()
{
    SetCvDetectorPtr(cv::xfeatures2d::StarDetector::create(
        m_maxSize,
        m_responseThreshold,
        m_lineThresholdProjected,
        m_lineThresholdBinarized,
        m_suppressNonmaxSize
    ));
}

Parameterised::Options STARFeatureDetector::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        Options o("BRIEF () Feature Extraction Options");
        o.add_options()
            ("max-size",                 po::value<int>(&m_maxSize)               ->default_value(m_maxSize), "")
            ("response-threshold",       po::value<int>(&m_responseThreshold)     ->default_value(m_responseThreshold), "")
            ("line-threshold-projected", po::value<int>(&m_lineThresholdProjected)->default_value(m_lineThresholdProjected), "")
            ("line-threshold-binarized", po::value<int>(&m_lineThresholdBinarized)->default_value(m_lineThresholdBinarized), "")
            ("suppress-nonmax-size",     po::value<int>(&m_suppressNonmaxSize)    ->default_value(m_suppressNonmaxSize), "");
        a.add(o);
    }

    return a; 
}

void MSDDFeatureDetector::WriteParams(cv::FileStorage & fs) const
{
    fs << "patchRadius"          <<  m_patchRadius;
    fs << "searchAreaRadius"     <<  m_searchAreaRadius;
    fs << "nmsRadius"            <<  m_nmsRadius;
    fs << "nmsScaleRadius"       <<  m_nmsScaleRadius;
    fs << "thSaliency"           <<  m_thSaliency;
    fs << "kNN"                  <<  m_kNN;
    fs << "scaleFactor"          <<  m_scaleFactor;
    fs << "nScales"              <<  m_nScales;
    fs << "computeRrientation"   <<  m_computeOrientation;
}

bool MSDDFeatureDetector::ReadParams(const cv::FileNode & fn)
{
    fn[" patchRadius"]      >> m_patchRadius;
    fn["searchAreaRadius"]  >> m_searchAreaRadius;
    fn["nmsRadius"]         >> m_nmsRadius;
    fn["nmsScaleRadius"]    >> m_nmsScaleRadius;
    fn["thSaliency"]        >> m_thSaliency;
    fn["kNN"]               >> m_kNN;
    fn["scaleFactor"]       >> m_scaleFactor;
    fn["nScales "]          >> m_nScales;
    fn["computeRrientation"]>> m_computeOrientation;

    return true;
}

void MSDDFeatureDetector::ApplyParams()
{
    SetCvDetectorPtr(cv::xfeatures2d::MSDDetector::create(
       m_patchRadius,
       m_searchAreaRadius,
       m_nmsRadius,
       m_nmsScaleRadius,
       m_thSaliency,
       m_kNN,
       m_scaleFactor,
       m_nScales,
       m_computeOrientation
    ));
}

Parameterised::Options MSDDFeatureDetector::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        Options o("BRIEF () Feature detection Options");
        o.add_options()
            ("patch-radius",        po::value<int>  (&m_patchRadius)        ->default_value(m_patchRadius),         "")
            ("search-area-dadius",  po::value<int>  (&m_searchAreaRadius)   ->default_value(m_searchAreaRadius),    "")
            ("nms-radius",          po::value<int>  (&m_nmsRadius)          ->default_value(m_nmsRadius),           "")
            ("nms-scale-radius",    po::value<int>  (&m_nmsScaleRadius)     ->default_value(m_nmsScaleRadius),      "")
            ("th-saliency",         po::value<float>(&m_thSaliency)         ->default_value(m_thSaliency),          "")
            ("kNN",                 po::value<int>  (&m_kNN)                ->default_value(m_kNN),                 "")
            ("scale-factor",        po::value<float>(&m_scaleFactor)        ->default_value(m_scaleFactor),         "")
            ("n-scales",            po::value<int>  (&m_nScales)            ->default_value(m_nScales),             "")
            ("compute-orientation", po::value<bool> (& m_computeOrientation)->default_value( m_computeOrientation), "");
        a.add(o);
    }

    return a; 
}

void LUCIDFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs <<  "lucidKernel" << m_lucidKernel;
    fs <<  "blurKernel"  << m_blurKernel;
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

    if (flag & FeatureOptionType::DETECTION_OPTIONS)
    {
        Options o("LUCID () Feature Extraction Options");
        o.add_options()
            ("ucid-kernel", po::value<int> (&m_lucidKernel) ->default_value(m_lucidKernel), "")
            ("blur-kernel", po::value<int> (&m_blurKernel)  ->default_value( m_blurKernel), "");
        a.add(o);
    }

    return a; 
}
/**/
void DAISYFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "radius"          <<  m_radius;
    fs << "qRadius"         <<  m_qRadius;
    fs << "qTheta "         <<  m_qTheta;
    fs << "qHist"           <<  m_qHist;
    fs << "norm "           <<  m_norm;
    fs << "interpolation"   <<  m_interpolation;
    fs << "useOrientation " <<  m_useOrientation;
}

bool DAISYFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["radius"]          >>  m_radius;
    fn["qRadius"]         >>  m_qRadius;
    fn["qTheta"]          >>  m_qTheta;
    fn["qHist" ]          >>  m_qHist;
    fn["norm" ]           >>  m_norm;
    fn["interpolation"]   >>  m_interpolation;
    fn["useOrientation"]  >>  m_useOrientation;

    return true;
}

void DAISYFeatureExtractor::ApplyParams()
{
    SetCvExtractorPtr(cv::xfeatures2d::DAISY::create(
         m_radius,
         m_qRadius,
         m_qTheta, 
         m_qHist, 
         m_norm,
         m_interpolation,
         m_useOrientation
    ));
}

Parameterised::Options DAISYFeatureExtractor::GetOptions(int flag)
{
    Options a;

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        Options o("DAISY () Feature Extraction Options");
        o.add_options()
            ("radius",         po::value<float>    (&m_radius)         ->default_value(m_radius),        "")
            ("q-radius",       po::value<int>      (&m_qRadius)        ->default_value(m_qRadius),       "")
            ("q-theta",        po::value<int>      (&m_qTheta)         ->default_value(m_qTheta),        "")
            ("q-hist",         po::value<int>      (&m_qHist)          ->default_value(m_qHist),         "")
            ("norm",           po::value<int>      (&m_norm)           ->default_value(m_norm),          "")
            ("interpolation",  po::value<bool>     (&m_interpolation)  ->default_value(m_interpolation), "")
            ("use-orientation",po::value<bool>     (&m_useOrientation) ->default_value(m_useOrientation),"");
        a.add(o);
    }

    return a; 
}

void LATCHFeatureExtractor::WriteParams(cv::FileStorage & fs) const
{
    fs << "bytess"            <<   m_bytes;
    fs << "rotationInvariance"<<   m_rotationInvariance;
    fs << "halfSSDSize "      <<   m_halfSSDSize;
}

bool seq2map::LATCHFeatureExtractor::ReadParams(const cv::FileNode & fn)
{
    fn["bytess"]             >>   m_bytes;
    fn["rotationInvariance"] >>   m_rotationInvariance;
    fn["halfSSDSize"]        >>   m_halfSSDSize;

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

    if (flag & FeatureOptionType::EXTRACTION_OPTIONS)
    {
        Options o("LATCH () Feature Extraction Options");
        o.add_options()
            ("bytes",              po::value<int>  (&m_bytes)             ->default_value(m_bytes),             "")
            ("rotation-invariance",po::value<bool> (&m_rotationInvariance)->default_value(m_rotationInvariance),"")
            ("half-ssd-size",      po::value<int>  (&m_halfSSDSize)       ->default_value(m_halfSSDSize),       "");
        a.add(o);
    }

    return a; 
}

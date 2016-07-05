#ifndef FEATURES_HPP
#define FEATURES_HPP
#define HAVE_XFEATURES2D

#ifdef HAVE_XFEATURES2D
#include <opencv2\xfeatures2d.hpp>
#endif // HAVE_XFEATURES2D

#include <seq2map\common.hpp>

namespace seq2map
{
	typedef std::vector<cv::KeyPoint> KeyPoints;
    enum FeatureOptionType
    {
        DETECTION_OPTIONS  = 0x000001,
        EXTRACTION_OPTIONS = 0x000002
    };

	class ImageFeature
	{
	public:
        friend class ImageFeatureSet;
	protected:
        /* ctor */ ImageFeature(cv::KeyPoint& keypoint, cv::Mat descriptor)
                   : m_keypoint(keypoint), m_descriptor(descriptor) {}
		/* dtor */ virtual ~ImageFeature() {}
        cv::KeyPoint& m_keypoint;
        cv::Mat       m_descriptor;
    };

	class ImageFeatureSet
	{
    public:
        /* ctor */ ImageFeatureSet(const KeyPoints& keypoints, const cv::Mat& descriptors, int normType = cv::NormTypes::NORM_L2)
                   : m_keypoints(keypoints), m_descriptors(descriptors), m_normType(normType) {}
		/* dtor */ virtual ~ImageFeatureSet() {}
        inline ImageFeature GetFeature(const size_t idx);
        bool Write(const Path& path) const;
    protected:
        static String NormType2String(int type);
        static String MatType2String(int type);

        static const String s_fileMagicNumber;
        static const String s_fileHeaderSep;

        KeyPoints m_keypoints;
        cv::Mat   m_descriptors;
        const int m_normType;
	};

	/**
	 * Feature detector and extractor interfaces
	 */
	
	class FeatureDetector : public virtual Parameterised
	{
	public:
		virtual KeyPoints DetectFeatures(const cv::Mat& im) const = 0;
	};

	class FeatureExtractor : public virtual Parameterised
	{
	public:
		virtual ImageFeatureSet ExtractFeatures(const cv::Mat& im, KeyPoints& keypoints) const = 0;
	};

	class FeatureDetextractor : public virtual Parameterised
	{
	public:
		virtual ImageFeatureSet DetectAndExtractFeatures(const cv::Mat& im) const = 0;
	};

	/**
	 * A concrete feature detection-and-extraction class that ultilises the
	 * composition of FeatureDetector and FeatureExtractor objects to do the work.
	 */

	typedef boost::shared_ptr<FeatureDetector>	   FeatureDetectorPtr;
	typedef boost::shared_ptr<FeatureExtractor>	   FeatureExtractorPtr;
	typedef boost::shared_ptr<FeatureDetextractor> FeatureDetextractorPtr;

    class HetergeneousDetextractor : public FeatureDetextractor
	{
    public:
        /* ctor */ HetergeneousDetextractor(FeatureDetectorPtr detector, FeatureExtractorPtr extractor)
                   : m_detector(detector), m_extractor(extractor) {}
		/* dtor */ virtual ~HetergeneousDetextractor() {}
        virtual ImageFeatureSet DetectAndExtractFeatures(const cv::Mat& im) const;
        virtual void WriteParams(cv::FileStorage& f) const;
        virtual bool ReadParams(const cv::FileNode& f);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    protected:
        static const String s_detectorFileNodeName;
        static const String s_extractorFileNodeName;
        FeatureDetectorPtr  m_detector;
        FeatureExtractorPtr m_extractor;
    };

	/**
	 * Factories to instantiate feature detector/extractor objects
	 * given name strings.
	 */

    class FeatureDetectorFactory :
		public Factory<String, FeatureDetector>
    {
    public:
        /* ctor */ FeatureDetectorFactory();
        /* dtor */ virtual ~FeatureDetectorFactory() {}
    };

    class FeatureExtractorFactory :
		public Factory<String, FeatureExtractor>
    {
    public:
        /* ctor */ FeatureExtractorFactory();
        /* dtor */ virtual ~FeatureExtractorFactory() {}
    };

	class FeatureDetextractorFactory :
		public Factory<String, FeatureDetextractor>
	{
	public:
		/* ctor */ FeatureDetextractorFactory();
		/* dtor */ virtual ~FeatureDetextractorFactory() {}
		FeatureDetextractorPtr Create(const String& detectorName, const String& extractorName);
        inline const FeatureDetectorFactory&  GetDetectorFactory()  const { return m_detectorFactory;  }
        inline const FeatureExtractorFactory& GetExtractorFactory() const { return m_extractorFactory; }
    private:
        FeatureDetectorFactory  m_detectorFactory;
        FeatureExtractorFactory m_extractorFactory;
	};

    /**
     * Feature detectior and extractior adaptors for OpenCV
     */

    class CvFeatureDetectorAdaptor : public FeatureDetector
    {
    public:
        typedef cv::Ptr<cv::FeatureDetector> CvDetectorPtr;
        /* ctor */ CvFeatureDetectorAdaptor(CvDetectorPtr cvDetector) : m_cvDetector(cvDetector) {}
        /* dtor */ virtual ~CvFeatureDetectorAdaptor() {}
        virtual KeyPoints DetectFeatures(const cv::Mat& im) const;
	private:
        CvDetectorPtr m_cvDetector;
    };

    class CvFeatureExtractorAdaptor : public FeatureExtractor
    {
    public:
        typedef cv::Ptr<cv::DescriptorExtractor> CvExtractorPtr;
        /* ctor */ CvFeatureExtractorAdaptor(CvExtractorPtr cvExtractor) : m_cvExtractor(cvExtractor) {}
        /* dtor */ virtual ~CvFeatureExtractorAdaptor() {}
        virtual ImageFeatureSet ExtractFeatures(const cv::Mat& im, KeyPoints& keypoints) const;
	private:
        CvExtractorPtr m_cvExtractor;
    };

    class CvFeatureDetextractorAdaptor : public FeatureDetextractor
    {
    public:
        typedef cv::Ptr<cv::Feature2D> CvDextractorPtr;
        /* ctor */ CvFeatureDetextractorAdaptor(CvDextractorPtr cvDxtor) : m_cvDxtor(cvDxtor) {}
        /* dtor */ virtual ~CvFeatureDetextractorAdaptor() {}
        virtual ImageFeatureSet DetectAndExtractFeatures(const cv::Mat& im) const;
    private:
        CvDextractorPtr m_cvDxtor;
    };

	template<class T> class CvSuperDetextractorAdaptor :
		public CvFeatureDetectorAdaptor,
		public CvFeatureExtractorAdaptor,
		public CvFeatureDetextractorAdaptor
	{
	protected:
		/* ctor */ CvSuperDetextractorAdaptor(cv::Ptr<T> cvDxtor) :
			m_cvDxtor(cvDxtor),
			CvFeatureDetectorAdaptor    (cvDxtor),
			CvFeatureExtractorAdaptor   (cvDxtor),
			CvFeatureDetextractorAdaptor(cvDxtor) {}
		/* dtor */ virtual ~CvSuperDetextractorAdaptor() {}
		cv::Ptr<T> m_cvDxtor;
	};

    class GFTTFeatureDetector :
        public CvFeatureDetectorAdaptor
    {
    public:
        /* ctor */ GFTTFeatureDetector(cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create())
                   : m_gftt(gftt), 
                     m_maxFeatures (gftt->getMaxFeatures()   ),
                     m_minDistance (gftt->getMinDistance()   ),
                     m_qualityLevel(gftt->getQualityLevel()  ),
                     m_blockSize   (gftt->getBlockSize()     ),
                     m_harrisCorner(gftt->getHarrisDetector()),
                     m_harrisK     (gftt->getK()             ),
                     CvFeatureDetectorAdaptor(gftt) {}
        /* dtor */ virtual ~GFTTFeatureDetector() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        cv::Ptr<cv::GFTTDetector> m_gftt;
        int    m_maxFeatures;
        double m_minDistance;
        double m_qualityLevel;
        int    m_blockSize;
        bool   m_harrisCorner;
        double m_harrisK;
    };

	class FASTFeatureDetector :
		public CvFeatureDetectorAdaptor
	{
	public:
        /* ctor */ FASTFeatureDetector(cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create())
                   : m_fast(fast),
                     m_threshold(fast->getThreshold()),
                     m_nonmaxSup(fast->getNonmaxSuppression()),
                     m_neighbour(Type2NeighbourCode(m_fast->getType())),
                     CvFeatureDetectorAdaptor(fast) {}
        /* dtor */ virtual ~FASTFeatureDetector() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
	private:
        static int NeighbourCode2Type(int neighbour);
        static int Type2NeighbourCode(int type);
		cv::Ptr<cv::FastFeatureDetector> m_fast;
        int  m_threshold;
        bool m_nonmaxSup;
        int  m_neighbour;
	};

    class ORBFeatureDetextractor :
		public CvSuperDetextractorAdaptor<cv::ORB>
    {
    public:
        /* ctor */ ORBFeatureDetextractor(cv::Ptr<cv::ORB> orb = cv::ORB::create())
                   : m_maxFeatures  (orb->getMaxFeatures()  ),
                     m_scaleFactor  (orb->getScaleFactor()  ),
                     m_levels       (orb->getNLevels()      ),
                     m_edgeThreshold(orb->getEdgeThreshold()),
                     m_wtaK         (orb->getWTA_K()        ),
                     m_scoreType    (ScoreType2String(orb->getScoreType())),
                     m_patchSize    (orb->getPatchSize()    ),
                     m_fastThreshold(orb->getFastThreshold()),
                     CvSuperDetextractorAdaptor(orb) {}
		/* dtor */ virtual ~ORBFeatureDetextractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        static int String2ScoreType(const String& scoreName);
        static String ScoreType2String(int type);
        int     m_maxFeatures;
        double  m_scaleFactor;
        int     m_levels;
        int     m_edgeThreshold;
        int     m_wtaK;
        String  m_scoreType;
        int     m_patchSize;
        int     m_fastThreshold;
    };

	class BRISKFeatureDetextractor :
		public CvSuperDetextractorAdaptor<cv::BRISK>
	{
	public:
		/* ctor */ BRISKFeatureDetextractor(cv::Ptr<cv::BRISK> brisk = cv::BRISK::create())
			       : CvSuperDetextractorAdaptor(brisk) {}
		/* dtor */ virtual ~BRISKFeatureDetextractor() {}
	};

    class KAZEFeatureDetextractor :
        public CvFeatureDetextractorAdaptor
    {
    public:
        /* ctor */ KAZEFeatureDetextractor(cv::Ptr<cv::KAZE> kaze = cv::KAZE::create())
                   : m_kaze(kaze), CvFeatureDetextractorAdaptor(kaze) {}
        /* dtor */ virtual ~KAZEFeatureDetextractor() {}
    private:
        cv::Ptr<cv::KAZE> m_kaze;
    };

    class AKAZEFeatureDetextractor :
        public CvFeatureDetextractorAdaptor
    {
    public:
        /* ctor */ AKAZEFeatureDetextractor(cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create())
                   : CvFeatureDetextractorAdaptor(akaze) {}
        /* dtor */ virtual ~AKAZEFeatureDetextractor() {}
    private:
        cv::Ptr<cv::AKAZE> m_akaze;
    };
#ifdef HAVE_XFEATURES2D

    class SIFTFeatureDetextractor :
        public CvSuperDetextractorAdaptor<cv::xfeatures2d::SIFT>
    {
    public:
        /* ctor */ SIFTFeatureDetextractor(cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create())
                   : m_maxFeatures      (0    ),
                     m_octaveLayers     (3    ),
                     m_contrastThreshold(0.04f),
                     m_edgeThreshold    (10.0f),
                     m_sigma            (1.6f ),
                     CvSuperDetextractorAdaptor(sift) {}
        /* dtor */ virtual ~SIFTFeatureDetextractor() {}
        void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int    m_maxFeatures;
        int    m_octaveLayers;
        double m_contrastThreshold;
        double m_edgeThreshold;
        double m_sigma;
    };

	class SURFFeatureDetextractor :
        public CvSuperDetextractorAdaptor<cv::xfeatures2d::SURF>
	{
	public:
		/* ctor */ SURFFeatureDetextractor(cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create())
                   : CvSuperDetextractorAdaptor(surf) {}
		/* dtor */ virtual ~SURFFeatureDetextractor() {}
	};

    class BRIEFFeatureExtractor :
        public CvFeatureExtractorAdaptor
    {
    public:
        /* ctor */ BRIEFFeatureExtractor(cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create())
                   : m_brief(brief), CvFeatureExtractorAdaptor(brief) {}
        /* dtor */ virtual ~BRIEFFeatureExtractor() {}
    protected:
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> m_brief;
    };
#endif // HAVE_XFEATURES2D
}
#endif // FEATURES_HPP

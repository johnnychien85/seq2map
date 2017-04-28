#ifndef FEATURES_HPP
#define FEATURES_HPP
#include <list>
#ifdef WITH_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif // WITH_XFEATURES2D
#include <seq2map/common.hpp>

namespace seq2map
{
    typedef std::vector<cv::KeyPoint> KeyPoints;

    /**
     * Flags to indiate what kind of options are requested to a feature
     * detector, an extractor, or a detextractor.
     */
    enum FeatureOptionType
    {
        DETECTION_OPTIONS  = 0x00000001,
        EXTRACTION_OPTIONS = 0x00000002
    };

    /**
     * Image feature class represents a single image feature.
     */
    class ImageFeature
    {
    public:
        friend class ImageFeatureSet;
        /* dtor */ virtual ~ImageFeature() {}
        cv::KeyPoint& keypoint;
        cv::Mat       descriptor;
    protected:
        /* ctor */ ImageFeature(cv::KeyPoint& keypoint, cv::Mat descriptor)
            : keypoint(keypoint), descriptor(descriptor) {}
    };

    /**
     * The class deficated to the collection of image features. As most of the
     * operations related to image feature involve more than one feature, it is
     * reasonable to actualise the concept of image feature collection as a new
     * class.
     */
    class ImageFeatureSet : public Persistent<Path>
    {
    public:
        static String NormType2String(int type);
        static int    String2NormType(const String& type);

        /* ctor */ ImageFeatureSet(const KeyPoints& keypoints, const cv::Mat& descriptors, int normType = cv::NORM_L2)
            : m_keypoints(keypoints), m_descriptors(descriptors), m_normType(normType) {}
        /* ctor */ ImageFeatureSet() : m_normType(cv::NORM_L2) {}
        /* dtor */ virtual ~ImageFeatureSet() {}
        ImageFeature GetFeature(const size_t idx);
        //inline ImageFeature GetFeature(const size_t idx) const;
        inline ImageFeature operator[](size_t idx) { return GetFeature(idx); }
        //inline ImageFeature operator[](size_t idx) const { return GetFeature(idx); }
        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path);
        inline bool IsEmpty() const { return m_keypoints.empty(); }
        inline size_t GetSize() const { return m_keypoints.size(); }
        inline int GetNormType() const { return m_normType; }
        inline const KeyPoints& GetKeyPoints() const { return m_keypoints; }
        inline const cv::Mat GetDescriptors() const { return m_descriptors; }
    protected:
        static String MatType2String(int type);
        static int    String2MatType(const String& type);

        static const String s_fileMagicNumber;
        static const char   s_fileHeaderSep;

        KeyPoints m_keypoints;
        cv::Mat   m_descriptors;
        int       m_normType;
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

    class FeatureDetextractor
    : public virtual Parameterised,
      public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
        typedef boost::shared_ptr<FeatureDetextractor> Ptr;
        typedef boost::shared_ptr<const FeatureDetextractor> ConstPtr;

        inline String GetKeypointName()   const { return m_keypointType;   }
        inline String GetDescriptorName() const { return m_descriptorType; }

        virtual ImageFeatureSet DetectAndExtractFeatures(const cv::Mat& im) const = 0;
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

    private:
        String m_keypointType;
        String m_descriptorType;
        friend class FeatureDetextractorFactory;
    };

    typedef boost::shared_ptr<FeatureDetector>	   FeatureDetectorPtr;
    typedef boost::shared_ptr<FeatureExtractor>	   FeatureExtractorPtr;
    typedef boost::shared_ptr<FeatureDetextractor> FeatureDetextractorPtr;

    /**
     * A concrete feature detection-and-extraction class that ultilises the
     * composition of FeatureDetector and FeatureExtractor objects to do the work.
     */
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
     * from a given name string.
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

    class FeatureDetextractorFactory
    : public Factory<String, FeatureDetextractor>,
      public Singleton<FeatureDetextractorFactory>
    {
    public:
        friend class Singleton<FeatureDetextractorFactory>;

        FeatureDetextractor::Ptr Create(const String& detectorName, const String& extractorName);
        FeatureDetextractor::Ptr Create(const cv::FileNode& fn);
        inline const FeatureDetectorFactory&  GetDetectorFactory()  const { return m_detectorFactory; }
        inline const FeatureExtractorFactory& GetExtractorFactory() const { return m_extractorFactory; }

    protected:
        virtual void Init();

    private:
        FeatureDetectorFactory  m_detectorFactory;
        FeatureExtractorFactory m_extractorFactory;
    };

    /**
    * A feature match is represented by a (srdIdx,dstIdx) tuplie along
    * with their distance in the feature space as well as a state flag.
    */
    struct FeatureMatch
    {
        enum Flag
        {
            INLIER                 = 1 << 0,
            RATIO_TEST_FAILED      = 1 << 1,
            UNIQUENESS_FAILED      = 1 << 2,
            SIGMA_TEST_FAILED      = 1 << 3,
            GEOMETRIC_TEST_FAILED  = 1 << 4,
            INLIER_RECOVERED       = 1 << 8 | INLIER
        };

        /* ctor */ FeatureMatch(size_t srcIdx = INVALID_INDEX, size_t dstIdx = INVALID_INDEX, float distance = -1, int state = INLIER)
        : srcIdx(srcIdx), dstIdx(dstIdx), distance(distance), state(state) {}
        inline void Reject(int flag) { state = (state ^ INLIER) | flag; }

        size_t srcIdx;
        size_t dstIdx;
        float  distance;
        int    state;
    };

    typedef std::vector<FeatureMatch> FeatureMatches;

    /**
     *
     */
    class ImageFeatureMap
    {
    public:
        inline FeatureMatch& operator[] (size_t idx) { return m_matches[idx]; }
        Indices Select(int mask) const;
        void Draw(cv::Mat& canvas);
        cv::Mat Draw(const cv::Mat& src, const cv::Mat& dst);
        inline const ImageFeatureSet& From() const { return m_src; }
        inline const ImageFeatureSet& To()   const { return m_dst; }
        inline const FeatureMatches& GetMatches() const { return m_matches; }
        inline FeatureMatches& GetMatches() { return m_matches; }
    protected:
        ImageFeatureMap(const ImageFeatureSet& src, const ImageFeatureSet& dst)
        : m_src(src), m_dst(dst) {};
        const ImageFeatureSet& m_src;
        const ImageFeatureSet& m_dst;
        FeatureMatches m_matches;

        friend class FeatureMatcher;
    };

    class FeatureMatchFilter
    {
    public:
        virtual bool Filter(ImageFeatureMap& map, Indices& inliers) = 0;
    };

    class FeatureMatcher
    {
    public:
        /* ctor */ FeatureMatcher(bool exhaustive = true, bool symmetric = true, float maxRatio = 0.6f)
            : m_exhaustive(exhaustive), m_symmetric(symmetric), m_maxRatio(maxRatio),
              m_descMatchingMetre("Descriptors Matching", "features/s"),
              m_ratioTestMetre   ("Ratio Test",           "matches/s"),
              m_symmetryTestMetre("Symmetry Test",        "matches/s"),
              m_filteringMetre   ("Filtering",            "matches/s") {};
        /* dtor */ virtual ~FeatureMatcher() {}
        FeatureMatcher& AddFilter(FeatureMatchFilter& filter);
        ImageFeatureMap MatchFeatures(const ImageFeatureSet& src, const ImageFeatureSet& dst);
        String Report() const;
    protected:
        static cv::Mat NormaliseDescriptors(const cv::Mat& desc);
        FeatureMatches MatchDescriptors(const cv::Mat& src, const cv::Mat& dst, int metric);

        std::vector<FeatureMatchFilter*> m_filters;
        bool  m_exhaustive;
        bool  m_symmetric;
        float m_maxRatio;
        Speedometre m_descMatchingMetre;
        Speedometre m_ratioTestMetre;
        Speedometre m_symmetryTestMetre;
        Speedometre m_filteringMetre;
    private:
        static void RunSymmetryTest(FeatureMatches& forward, const FeatureMatches& backward);
    };

    class FundamentalMatrixFilter : public FeatureMatchFilter
    {
    public:
        /* ctor */ FundamentalMatrixFilter(double epsilon = 1, double confidence = 0.99f, bool ransac = true)
            : m_epsilon(epsilon), m_confidence(confidence), m_ransac(ransac) {}
        /* dtor */ virtual ~FundamentalMatrixFilter() {}
        virtual bool Filter(ImageFeatureMap& map, Indices& inliers);
        inline cv::Mat GetFundamentalMatrix() const { return m_fmat.clone(); }
    protected:
        double  m_epsilon;
        double  m_confidence;
        bool    m_ransac;
        cv::Mat m_fmat;
    };

    class EssentialMatrixFilter : public FeatureMatchFilter
    {
    public:
        /* ctor */ EssentialMatrixFilter(double epsilon = 1, double confidence = 0.99f, bool ransac = true, bool poseRecovery = false,
                   const cv::Mat& K0 = cv::Mat::eye(3, 3, CV_64F), const cv::Mat&K1 = cv::Mat::eye(3, 3, CV_64F))
                   : m_epsilon(epsilon), m_confidence(confidence), m_ransac(ransac), m_poseRecovery(poseRecovery) { SetCameraMatrices(K0, K1); }
        /* dtor */ virtual ~EssentialMatrixFilter() {}
        virtual bool Filter(ImageFeatureMap& map, Indices& inliers);
        inline cv::Mat GetEssentialMatrix() const { return m_emat.clone(); }
        inline void GetPose(cv::Mat& rmat, cv::Mat& tvec) const { rmat = m_rmat.clone(); tvec = m_tvec.clone(); };
        inline void SetPoseRecovery(bool poseRecovery) { m_poseRecovery = poseRecovery; }
        void SetCameraMatrices(const cv::Mat& K0, const cv::Mat& K1);
    protected:
        double  m_epsilon;
        double  m_confidence;
        bool    m_ransac;
        bool    m_poseRecovery;
        cv::Mat m_emat;
        cv::Mat m_rmat;
        cv::Mat m_tvec;
    private:
        static void BackprojectPoints(Points2D& pts, const cv::Mat& Kinv);
        cv::Mat m_K0inv;
        cv::Mat m_K1inv;

    };

    class SigmaFilter : public FeatureMatchFilter
    {
    public:
        /* ctor */ SigmaFilter(float k = 2.0f) : m_k(k) {}
        /* dtor */ virtual ~SigmaFilter() {}
        virtual bool Filter(ImageFeatureMap& map, Indices& inliers);
    protected:
        float m_k;
        float m_mean;
        float m_stdev;
    };

    /**
     * Feature detection and extraction adaptors for OpenCV
     */

    class CvFeatureDetectorAdaptor : public FeatureDetector
    {
    public:
        typedef cv::Ptr<cv::FeatureDetector> CvDetectorPtr;
        /* ctor */ CvFeatureDetectorAdaptor(CvDetectorPtr cvDetector) : m_cvDetector(cvDetector) {}
        /* dtor */ virtual ~CvFeatureDetectorAdaptor() {}
        virtual KeyPoints DetectFeatures(const cv::Mat& im) const;
        inline void SetCvDetectorPtr(CvDetectorPtr cvDetector) { m_cvDetector = cvDetector; }
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
        inline void SetCvExtractorPtr(CvExtractorPtr cvExtractor) { m_cvExtractor = cvExtractor; }
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
        inline void SetCvDetextractorPtr(CvDextractorPtr cvDxtor) { m_cvDxtor = cvDxtor; }
    private:
        CvDextractorPtr m_cvDxtor;
    };

    template<class T> class CvSuperDetextractorAdaptor :
        public CvFeatureDetectorAdaptor,
        public CvFeatureExtractorAdaptor,
        public CvFeatureDetextractorAdaptor
    {
    protected:
        /* ctor */ CvSuperDetextractorAdaptor(cv::Ptr<T> cvDxtor)
            : m_cvDxtor(cvDxtor),
            CvFeatureDetectorAdaptor(cvDxtor),
            CvFeatureExtractorAdaptor(cvDxtor),
            CvFeatureDetextractorAdaptor(cvDxtor) {}
        /* dtor */ virtual ~CvSuperDetextractorAdaptor() {}
        inline void SetCvSuperDetextractorPtr(CvDextractorPtr cvDxtor);
        cv::Ptr<T> m_cvDxtor;
    };

    class GFTTFeatureDetector :
        public CvFeatureDetectorAdaptor
    {
    public:
        /* ctor */ GFTTFeatureDetector(cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create())
            : m_gftt(gftt),
            m_maxFeatures (gftt->getMaxFeatures()),
            m_minDistance (gftt->getMinDistance()),
            m_qualityLevel(gftt->getQualityLevel()),
            m_blockSize   (gftt->getBlockSize()),
            m_harrisCorner(gftt->getHarrisDetector()),
            m_harrisK     (gftt->getK()),
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

    class AGASTFeatureDetector :
        public CvFeatureDetectorAdaptor
    {
    public:
        /* ctor */ AGASTFeatureDetector(cv::Ptr<cv::AgastFeatureDetector> agast = cv::AgastFeatureDetector::create())
            : m_agast(agast),
            m_threshold(agast->getThreshold()        ),
            m_nonmaxSup(agast->getNonmaxSuppression()),
            m_neighbour(NeighbourType2String(agast->getType())),
            CvFeatureDetectorAdaptor(agast) {}
        /* dtor */ virtual ~AGASTFeatureDetector() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        static int String2NeighbourType(String neighbour);
        static String NeighbourType2String(int type);
        cv::Ptr<cv::AgastFeatureDetector> m_agast;
        int 	m_threshold;
        bool 	m_nonmaxSup;
        String 	m_neighbour;

    };

    class ORBFeatureDetextractor :
        public CvSuperDetextractorAdaptor<cv::ORB>
    {
    public:
        /* ctor */ ORBFeatureDetextractor(cv::Ptr<cv::ORB> orb = cv::ORB::create())
            : m_maxFeatures(orb->getMaxFeatures()),
            m_scaleFactor  (orb->getScaleFactor()),
            m_levels       (orb->getNLevels()),
            m_edgeThreshold(orb->getEdgeThreshold()),
            m_wtaK         (orb->getWTA_K()),
            m_scoreType    (ScoreType2String(orb->getScoreType())),
            m_patchSize    (orb->getPatchSize()),
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
            :
            m_threshold   (30  ),
            m_levels      (3   ),
            m_patternScale(1.0f),
            CvSuperDetextractorAdaptor(brisk) {}
        /* dtor */ virtual ~BRISKFeatureDetextractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int   m_threshold;
        int   m_levels;
        float m_patternScale;
    };

    class KAZEFeatureDetextractor :
        public CvSuperDetextractorAdaptor<cv::KAZE>
    {
    public:
        /* ctor */ KAZEFeatureDetextractor(cv::Ptr<cv::KAZE> kaze = cv::KAZE::create())
            :
            m_extended       (kaze->getExtended()     ),
            m_upright        (kaze->getUpright()      ),
            m_threshold      (kaze->getThreshold()    ),
            m_levels         (kaze->getNOctaves()     ),
            m_octaveLayers   (kaze->getNOctaveLayers()),
            m_diffusivityType(DiffuseType2String(kaze->getDiffusivity())),
            CvSuperDetextractorAdaptor(kaze) {}
        /* dtor */ virtual ~KAZEFeatureDetextractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        static int String2DiffuseType(const seq2map::String& scoreName);
        static String DiffuseType2String(int type);
        //cv::Ptr<cv::KAZE> m_kaze;
        bool   m_extended;
        bool   m_upright;
        double m_threshold;
        int    m_levels;
        int    m_octaveLayers;
        int    m_diffuseType;
        String m_diffusivityType;

        friend class AKAZEFeatureDetextractor;
    };

    class AKAZEFeatureDetextractor :
        public CvSuperDetextractorAdaptor<cv::AKAZE>
    {
    public:
        /* ctor */ AKAZEFeatureDetextractor(cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create())
            :
            m_descriptorType    (DescriptorType2String(akaze->getDescriptorType())),
            m_descriptorBits    (akaze->getDescriptorSize()    ),
            m_descriptorChannels(akaze->getDescriptorChannels()),
            m_threshold         (akaze->getThreshold()         ),
            m_levels            (akaze->getNOctaves()          ),
            m_octaveLayers      (akaze->getNOctaveLayers()     ),
            m_diffusivityType   (KAZEFeatureDetextractor::DiffuseType2String(akaze->getDiffusivity())),
            CvSuperDetextractorAdaptor(akaze) {}
        /* dtor */ virtual ~AKAZEFeatureDetextractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        static int String2DescriptorType(const seq2map::String& scoreName);
        static String DescriptorType2String(int type);
        String  m_descriptorType;
        int     m_descriptorBits;
        int     m_descriptorChannels;
        double  m_threshold;
        int     m_levels;
        int     m_octaveLayers;
        String  m_diffusivityType;
    };
#ifdef WITH_XFEATURES2D

    class StarFeatureDetector :
        public CvFeatureDetectorAdaptor
    {
    public:
        /* ctor */ StarFeatureDetector(cv::Ptr<cv::xfeatures2d::StarDetector> star = cv::xfeatures2d::StarDetector::create())
            :
            m_maxSize               (45),
            m_responseThreshold     (30),
            m_lineThresholdProjected(10),
            m_lineThresholdBinarized(8),
            m_suppressNonmaxSize    (5),
            CvFeatureDetectorAdaptor(star) {}
        /* dtor */ virtual ~StarFeatureDetector() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int m_maxSize;
        int m_responseThreshold;
        int m_lineThresholdProjected;
        int m_lineThresholdBinarized;
        int m_suppressNonmaxSize;
    };

    class MSDFeatureDetector :
        public CvFeatureDetectorAdaptor
    {
    public:
        /* ctor */ MSDFeatureDetector(cv::Ptr<cv::xfeatures2d::MSDDetector> msdd = cv::xfeatures2d::MSDDetector::create())
            :
            m_patchRadius    (  3     ),
            m_searchRadius   (  5     ),
            m_nmsRadius      (  5     ),
            m_nmsScaleRadius (  0     ),
            m_saliency       (250.0f  ),
            m_neighbours     (  4     ),
            m_scaleFactor    (  1.25f ),
            m_scales         ( -1     ),
            m_oriented       (  false ),
            CvFeatureDetectorAdaptor(msdd) {}
        /* dtor */ virtual ~MSDFeatureDetector() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int   m_patchRadius;
        int   m_searchRadius;
        int   m_nmsRadius;
        int   m_nmsScaleRadius;
        float m_saliency;
        int   m_neighbours;
        float m_scaleFactor;
        int   m_scales;
        bool  m_oriented;
    };

    class SIFTFeatureDetextractor :
        public CvSuperDetextractorAdaptor<cv::xfeatures2d::SIFT>
    {
    public:
        /* ctor */ SIFTFeatureDetextractor(cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create())
            :
            m_maxFeatures      ( 0    ),
            m_octaveLayers     ( 3    ),
            m_contrastThreshold( 0.04f),
            m_edgeThreshold    (10.00f),
            m_sigma            ( 1.60f),
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
            :
            m_hessianThreshold(surf->getHessianThreshold()),
            m_levels          (surf->getNOctaves()        ),
            m_octaveLayers    (surf->getNOctaveLayers()   ),
            m_extended        (surf->getExtended()        ),
            m_upright         (surf->getUpright()         ),
            CvSuperDetextractorAdaptor(surf) {}
        /* dtor */ virtual ~SURFFeatureDetextractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        double m_hessianThreshold;
        int    m_levels;
        int    m_octaveLayers;
        bool   m_extended;
        bool   m_upright;
    };

    class BRIEFFeatureExtractor :
        public CvFeatureExtractorAdaptor
    {
    public:
        /* ctor */ BRIEFFeatureExtractor(cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create())
            :
            m_descriptorBytes (32   ),
            m_oriented        (false),            
            CvFeatureExtractorAdaptor(brief) {}
        /* dtor */ virtual ~BRIEFFeatureExtractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int 	m_descriptorBytes;
        bool 	m_oriented;
    };

    class  DAISYFeatureExtractor :
        public CvFeatureExtractorAdaptor
    {
    public:
        /* ctor */ DAISYFeatureExtractor(cv::Ptr<cv::xfeatures2d::DAISY> daisy = cv::xfeatures2d::DAISY::create())
            :
            m_radius  (  15 ),
            m_iradius (   3 ),
            m_iangle  (   8 ),
            m_ihist   (   8 ),
            m_normType(NormType2String(cv::xfeatures2d::DAISY::NRM_NONE)),
            m_interp  (true ),
            m_oriented(false),
            CvFeatureExtractorAdaptor(daisy) {}
        /* dtor */ virtual ~DAISYFeatureExtractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int String2NormType(String normType);
        String NormType2String(int type);

        float  m_radius;
        int    m_iradius;
        int    m_iangle;
        int    m_ihist; 
        String m_normType;
        bool   m_interp;
        bool   m_oriented;
    };

    class FREAKFeatureExtractor :
        public CvFeatureExtractorAdaptor
    {
    public:
        /* ctor */ FREAKFeatureExtractor(cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create())
            :
            m_levels              ( 4   ),
            m_patternScale        (22.0f),
            m_normaliseOrientation(true ),
            m_normaliseScale      (true ),
            CvFeatureExtractorAdaptor(freak) {}
        /* dtor */ virtual ~FREAKFeatureExtractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int   m_levels;
        float m_patternScale;
        bool  m_normaliseScale;
        bool  m_normaliseOrientation;
    };

    class LATCHFeatureExtractor :
        public CvFeatureExtractorAdaptor
    {
    public:
        /* ctor */ LATCHFeatureExtractor(cv::Ptr<cv::xfeatures2d::LATCH> latch = cv::xfeatures2d::LATCH::create())
            :
            m_bytes             ( 32 ),
            m_rotationInvariance(true),
            m_halfSSDSize       (  3 ),
            CvFeatureExtractorAdaptor(latch) {}
        /* dtor */ virtual ~ LATCHFeatureExtractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int 	m_bytes;
        bool 	m_rotationInvariance;
        int 	m_halfSSDSize;
    };

    class LUCIDFeatureExtractor :
        public CvFeatureExtractorAdaptor
    {
    public:
        /* ctor */ LUCIDFeatureExtractor(cv::Ptr<cv::xfeatures2d::LUCID> lucid = cv::xfeatures2d::LUCID::create(1,1))//creat() parameter not enought
            :
            m_lucidKernel(1),
            m_blurKernel (1),
            CvFeatureExtractorAdaptor(lucid) {}
        /* dtor */ virtual ~LUCIDFeatureExtractor() {}
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);
    private:
        int m_lucidKernel;
        int m_blurKernel;
    };
#endif // WITH_XFEATURES2D
}
#endif // FEATURES_HPP

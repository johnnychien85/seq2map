#ifndef FEATURES_HPP
#define FEATURES_HPP
#include <list>
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

        const cv::KeyPoint& keypoint;
        const cv::Mat       descriptor;

    protected:
        /* ctor */ 
        ImageFeature(const cv::KeyPoint& keypoint, const cv::Mat descriptor)
        : keypoint(keypoint), descriptor(descriptor)
        {
            assert(descriptor.rows == 1);
        }
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

        /**
         * Construct am empty feature set.
         */
        ImageFeatureSet() : m_normType(cv::NORM_L2) {}

        /**
         * Construct a feature set from key points and descriptors.
         */
        ImageFeatureSet(const KeyPoints& keypoints, const cv::Mat& descriptors, int normType = cv::NORM_L2)
        : m_keypoints(keypoints), m_descriptors(descriptors), m_normType(normType)
        {
            if (keypoints.size() != descriptors.rows && !descriptors.empty())
            {
                E_ERROR << "inconsistent size of descriptor matrix";
            }
        }
       
        /**
         * Desctructor.
         */
        virtual ~ImageFeatureSet() {}

        /**
         *
         */
        ImageFeatureSet& operator= (const ImageFeatureSet& set);

        /**
         * Get a feature from set. A constant shallow copy is returned.
         */
        ImageFeature GetFeature(const size_t idx) const;

        /**
         * Same as GetFeature.
         */
        inline const ImageFeature operator[](size_t idx) const { return GetFeature(idx); }
        
        /**
         * Expand the feature set by appending another one.
         */
        bool Append(const ImageFeatureSet& set);

        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path);
        
        inline bool IsEmpty() const    { return m_keypoints.empty(); }
        inline size_t GetSize() const  { return m_keypoints.size();  }
        inline int GetNormType() const { return m_normType; }
        inline const KeyPoints& GetKeyPoints() const { return m_keypoints;   }
        inline const cv::Mat GetDescriptors() const  { return m_descriptors; }

    protected:
        static const String s_fileMagicNumber;
        static const char   s_fileHeaderSep;

        KeyPoints m_keypoints;
        cv::Mat   m_descriptors;
        int       m_normType;
    };

    /**
     * Feature detector and extractor interfaces
     */
    class FeatureDetector
    : public Referenced<FeatureDetector>,
      public virtual Parameterised
    {
    public:
        virtual KeyPoints DetectFeatures(const cv::Mat& im) const = 0;
    };

    class FeatureExtractor
    : public Referenced<FeatureExtractor>,
      public virtual Parameterised
    {
    public:
        virtual ImageFeatureSet ExtractFeatures(const cv::Mat& im, KeyPoints& keypoints) const = 0;
    };

    class FeatureDetextractor
    : public Referenced<FeatureDetextractor>,
      public virtual Parameterised,
      public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
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

    /**
     * A concrete feature detection-and-extraction class that ultilises the
     * composition of FeatureDetector and FeatureExtractor objects to do the work.
     */
    class HetergeneousDetextractor : public FeatureDetextractor
    {
    public:
        /* ctor */ HetergeneousDetextractor(FeatureDetector::Own& detector, FeatureExtractor::Own& extractor)
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
        FeatureDetector::Own  m_detector;
        FeatureExtractor::Own m_extractor;
    };

    /**
     * Factories to instantiate feature detector/extractor objects
     * from a given name string.
     */

    class FeatureDetectorFactory : public Factory<String, FeatureDetector>
    {
    public:
        /* ctor */ FeatureDetectorFactory();
        /* dtor */ virtual ~FeatureDetectorFactory() {}
    };

    class FeatureExtractorFactory : public Factory<String, FeatureExtractor>
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

        FeatureDetextractor::Own Create(const String& detectorName, const String& extractorName);
        FeatureDetextractor::Own Create(const cv::FileNode& fn);
        inline const FeatureDetectorFactory&  GetDetectorFactory()  const { return m_detectorFactory; }
        inline const FeatureExtractorFactory& GetExtractorFactory() const { return m_extractorFactory; }

    protected:
        virtual void Init();

    private:
        FeatureDetectorFactory  m_detectorFactory;
        FeatureExtractorFactory m_extractorFactory;
    };

    /**
     * A feature match is represented by a (srdIdx,dstIdx) tuple along
     * with their distance in the feature space as well as a state flag.
     */
    struct FeatureMatch
    {
        enum Flag
        {
            INLIER                 = 1 << 0,
            RATIO_TEST_FAILED      = 1 << 1,
            UNIQUENESS_FAILED      = 1 << 2,
            SYMMETRIC_FAILED       = 1 << 3,
            SIGMA_TEST_FAILED      = 1 << 4,
            GEOMETRIC_TEST_FAILED  = 1 << 5,
            INLIER_RECOVERED       = 1 << 8 | INLIER
        };

        FeatureMatch(size_t srcIdx = INVALID_INDEX, size_t dstIdx = INVALID_INDEX, float distance = -1, int state = INLIER)
        : srcIdx(srcIdx), dstIdx(dstIdx), distance(distance), state(state) {}

        inline void Reject(int reason) { state = (state & ~INLIER) | reason; }

        size_t srcIdx;
        size_t dstIdx;
        float  distance;
        int    state;
    };

    typedef std::vector<FeatureMatch> FeatureMatches;

    /**
     * A class to actualise the concept of a set of FeatureMatch.
     */
    class ImageFeatureMap
    {
    public:
        inline FeatureMatch& operator[] (size_t idx) { return m_matches[idx]; }
        IndexList Select(int mask) const;
        void Draw(cv::Mat& canvas) const;
        cv::Mat Draw(const cv::Mat& src, const cv::Mat& dst) const;
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

        // can only be constructed by a FeatureMatcher
        friend class FeatureMatcher;
    };

    /**
     * Feature matching from descriptors.
     */
    class FeatureMatcher
    {
    public:
        /**
         * Interface to filter out outliers from a feature map.
         */
        class Filter : public Referenced<Filter>
        {
        public:
            virtual bool operator() (ImageFeatureMap& map, IndexList& inliers) = 0;
        };

        typedef std::vector<Filter::Own> Filters;

        /**
         *
         */
        FeatureMatcher(bool exhaustive = true, bool uniqueness = true, bool symmetric = false, float maxRatio = 0.6f, bool useGpu = true)
        : m_exhaustive(exhaustive), m_uniqueness(uniqueness), m_symmetric(symmetric),
          m_maxRatio(maxRatio), m_useGpu(useGpu), m_maxDistance(0.01f),
          m_descMatchingMetre("DMatching",     "features/s"),
          m_ratioTestMetre   ("Ratio Test",    "matches/s"),
          m_symmetryTestMetre("Symmetry Test", "matches/s"),
          m_filteringMetre   ("Filtering",     "matches/s") {}
        
        /**
         *
         */
        virtual ~FeatureMatcher() {}

        /**
         *
         */
        ImageFeatureMap operator() (const ImageFeatureSet& src, const ImageFeatureSet& dst);

        //
        // Accessors
        //

        inline void SetUniqueness(bool enable) { m_uniqueness = enable; }
        inline bool GetUniqueness() const      { return m_uniqueness;   }

        inline void SetSymmetric(bool enable)  { m_symmetric = enable; }
        inline bool GetSymmetric() const       { return m_symmetric;   }

        inline void SetMaxRatio(float ratio)   { m_maxRatio = ratio; }
        inline double GetMaxRatio() const      { return m_maxRatio;  }

        inline void SetDistanceThreshold(float threshold) { m_maxDistance = threshold; }
        inline float GetDistanceThreshold() const         { return m_maxDistance; }

        /**
         *
         */
        FeatureMatcher& AddFilter(Filter::Own& filter);

        /**
         *
         */
        inline Filters& GetFilters() { return m_filters; }

        /**
         *
         */
        String Report() const;

    protected:
        static cv::Mat NormaliseDescriptors(const cv::Mat& desc);
        FeatureMatches MatchDescriptors(const cv::Mat& src, const cv::Mat& dst, int metric, bool ratioTest = true);

        Filters m_filters;
        bool  m_exhaustive;
        bool  m_uniqueness;
        bool  m_symmetric;
        bool  m_useGpu;
        float m_maxDistance;
        float m_maxRatio;
        Speedometre m_descMatchingMetre;
        Speedometre m_ratioTestMetre;
        Speedometre m_symmetryTestMetre;
        Speedometre m_filteringMetre;

    private:
        static void RunUniquenessTest(FeatureMatches& forward);
        static void RunSymmetryTest(FeatureMatches& forward, const FeatureMatches& backward, const std::vector<size_t>& idmap, size_t maxSrcIdx);
        float GetDistanceThreshold(float ratio, int metric, size_t d);
    };

    class FundamentalMatrixFilter : public FeatureMatcher::Filter
    {
    public:
        /**
         *
         */
        FundamentalMatrixFilter(double epsilon = 1, double confidence = 0.99f, bool ransac = true)
        : m_epsilon(epsilon), m_confidence(confidence), m_ransac(ransac) {}

        /**
         *
         */
        virtual ~FundamentalMatrixFilter() {}

        /**
         *
         */
        virtual bool operator() (ImageFeatureMap& map, IndexList& inliers);

        /**
         *
         */
        inline cv::Mat GetFundamentalMatrix() const { return m_fmat.clone(); }

    protected:
        double  m_epsilon;
        double  m_confidence;
        bool    m_ransac;
        cv::Mat m_fmat;
    };

    class EssentialMatrixFilter : public FeatureMatcher::Filter
    {
    public:
        /**
         *
         */
        EssentialMatrixFilter(double epsilon = 1, double confidence = 0.99f, bool ransac = true, bool poseRecovery = false,
            const cv::Mat& K0 = cv::Mat::eye(3, 3, CV_64F), const cv::Mat&K1 = cv::Mat::eye(3, 3, CV_64F))
        : m_epsilon(epsilon), m_confidence(confidence), m_ransac(ransac), m_poseRecovery(poseRecovery)
        { SetCameraMatrices(K0, K1); }

        /**
         *
         */
        virtual ~EssentialMatrixFilter() {}

        /**
         *
         */
        virtual bool operator() (ImageFeatureMap& map, IndexList& inliers);

        /**
         *
         */
        inline cv::Mat GetEssentialMatrix() const { return m_emat.clone(); }

        /**
         *
         */
        inline void GetPose(cv::Mat& rmat, cv::Mat& tvec) const { rmat = m_rmat.clone(); tvec = m_tvec.clone(); };
        
        /**
         *
         */
        inline void SetPoseRecovery(bool poseRecovery) { m_poseRecovery = poseRecovery; }
        
        /**
         *
         */
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
        /**
         *
         */
        static void BackprojectPoints(Points2D& pts, const cv::Mat& Kinv);

        cv::Mat m_K0inv;
        cv::Mat m_K1inv;
    };

    class SigmaFilter : public FeatureMatcher::Filter
    {
    public:
        /**
         *
         */
        SigmaFilter(float k = 2.0f) : m_k(k) {}
        
        /**
         *
         */
        virtual ~SigmaFilter() {}
        
        /**
         *
         */
        virtual bool operator() (ImageFeatureMap& map, IndexList& inliers);

    protected:
        float m_k;
        float m_mean;
        float m_stdev;
    };
}
#endif // FEATURES_HPP

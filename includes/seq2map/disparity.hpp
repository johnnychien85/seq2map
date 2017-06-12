#ifndef DISPARITY_HPP
#define DISPARITY_HPP
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/cudastereo.hpp>
#include <seq2map/common.hpp>

namespace seq2map
{
    class StereoMatcher
    : public Referenced<StereoMatcher>,
      public virtual Parameterised,
      public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
        typedef LinearSpacedVec<double> DisparitySpace;
        
        virtual void WriteParams(cv::FileStorage& f) const = 0;
        virtual bool ReadParams(const cv::FileNode& f) = 0;
        virtual void ApplyParams() = 0;
        virtual Options GetOptions(int flag = 0) = 0;

        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        virtual String GetMatcherName() const = 0;
        virtual cv::Mat Match(const cv::Mat& left, const cv::Mat& right) = 0;
        virtual DisparitySpace GetDisparitySpace() const = 0;
    };
    
    template<class T>
    class CvStereoMatcher : public StereoMatcher
    {
    public:
        typedef cv::Ptr<T> Ptr;
        CvStereoMatcher(const Ptr& matcher, bool useGpuMat = false) : m_matcher(matcher), m_useGpuMat(useGpuMat) {}

        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);

        virtual cv::Mat Match(const cv::Mat& left, const cv::Mat& right);
        virtual DisparitySpace GetDisparitySpace() const
        {
            return DisparitySpace(
                m_minDisparity,
                m_minDisparity + m_numDisparities,
                m_numDisparities * cv::StereoMatcher::DISP_SCALE - 1
            );
        }

    protected:
        cv::Ptr<T> m_matcher;
        int m_minDisparity;
        int m_numDisparities;
        int m_blockSize;
        int m_speckleWinSize;
        int m_speckleRange;
        int m_disp12MaxDiff;

        const bool m_useGpuMat;
    };

    template<class T>
    class BlockMatcher : public CvStereoMatcher<T>
    {
    public:
        BlockMatcher<T>(const Ptr& matcher, bool useGpuMat) : CvStereoMatcher<T>(matcher, useGpuMat) {}
        virtual String GetMatcherName() const = 0;

        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);

        static String FilterType2String(int type);
        static int String2FilterType(String type);

    protected:
        String m_preFilterType;
        int    m_preFilterCap;
        int    m_textureThreshold;
        int    m_uniquenessRatio;
        int    m_smallerBlockSize;
    };

    class CpuBlockMatcher : public BlockMatcher<cv::StereoBM>
    {
    public:
        CpuBlockMatcher() : BlockMatcher(cv::StereoBM::create(), false) {}
        virtual String GetMatcherName() const { return "BM"; }
    };

    class GpuBlockMatcher : public BlockMatcher<cv::cuda::StereoBM>
    {
    public:
        GpuBlockMatcher() : BlockMatcher(cv::cuda::createStereoBM(), true) {}
        virtual String GetMatcherName() const { return "BM_GPU"; }
    };

    class SemiGlobalBlockMatcher : public CvStereoMatcher<cv::StereoSGBM>
    {
    public:
        SemiGlobalBlockMatcher() : CvStereoMatcher<cv::StereoSGBM>(cv::StereoSGBM::create(0, 64, 21)) {}
        virtual String GetMatcherName() const { return "SGBM"; }

        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag);

        static String Mode2String(int type);
        static int String2Mode(String type);

    protected:
        String m_mode;
        int    m_preFilterCap;
        int    m_uniquenessRatio;
        int    m_p1;
        int    m_p2;
    };

    class StereoMatcherFactory
    : public Factory<String, StereoMatcher>,
      public Singleton<StereoMatcherFactory>
    {
    public:
        friend class Singleton<StereoMatcherFactory>;
        using Factory<String, StereoMatcher>::Create;
        StereoMatcher::Own Create(const cv::FileNode& fn);
    protected:
        virtual void Init();
    };
}

#endif //DISPARITY_HPP

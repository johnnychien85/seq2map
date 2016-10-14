#ifndef DISPARITY_HPP
#define DISPARITY_HPP
#include <opencv2/calib3d/calib3d.hpp>
#include <seq2map/common.hpp>

namespace seq2map
{
    class StereoMatcherAdaptor
    {
    public:
        /* ctor */ StereoMatcherAdaptor(std::string matcherName, int numDisparities)
                   : m_matcherName(matcherName), m_numDisparities(numDisparities) {}
        /* dtor */ virtual ~StereoMatcherAdaptor() {}
	    virtual cv::Mat Match(const cv::Mat& left, const cv::Mat& right) = 0;
	    virtual void    WriteParams(cv::FileStorage& f);
	    inline int      GetNumDisparities() const {return m_numDisparities;}
	    inline uint16_t GetScale() const          {return cv::StereoMatcher::DISP_SCALE;}
	    inline uint16_t GetNormalisationFactor() const {return (uint16_t)(std::floor((double)USHRT_MAX / GetScale() / m_numDisparities));}
    protected:
	    int				  m_numDisparities;
        const std::string m_matcherName;
    };

    class BlockMatcher : public StereoMatcherAdaptor
    {
    public:
        /* ctor */ BlockMatcher(int numDisparities, int SADWindowSize);
        /* dtor */ ~BlockMatcher();
        virtual cv::Mat	Match(const cv::Mat& left, const cv::Mat& right);
        virtual void    WriteParams(cv::FileStorage& f);
    private:
        cv::Ptr<cv::StereoBM> m_BM;
    };

    class SemiGlobalMatcher : public StereoMatcherAdaptor
    {
    public:
	    /* ctor */ SemiGlobalMatcher(int numDisparities, int SADWindowSize,
                    int P1, int P2, int disp12MaxDiff,
                    int preFilterCap, int uniquenessRatio,
                    int speckleWindowSize, int speckleRange, bool fullDP);
	    /* dtor */ ~SemiGlobalMatcher();
        virtual cv::Mat	Match(const cv::Mat& left, const cv::Mat& right);
        virtual void WriteParams(cv::FileStorage& f);
    private:
	    cv::Ptr<cv::StereoSGBM> m_SGBM;
    };

    class DisparityIO
    {
    public:
        /* ctor */ DisparityIO() : m_denorm(0), m_scale(0) {}
        /* ctor */ DisparityIO(uint16_t denorm, double scale) : m_denorm(denorm), m_scale(scale) {}
        /* ctor */ DisparityIO(const StereoMatcherAdaptor& matcher)
                   : m_denorm(matcher.GetNormalisationFactor()), m_scale(matcher.GetScale()) {}
        /* dtor */ virtual ~DisparityIO() {}

        bool    Write(const cv::Mat& dp, const Path& path);
        cv::Mat Read(const Path& path);

    protected:
        uint16_t m_denorm;
        double m_scale;
    };
}

namespace cv
{
    static void write(FileStorage& fs, const std::string&, const seq2map::DisparityIO& dpio)
    {

    }

    static void read(const FileNode& fn, seq2map::DisparityIO& dpio,
        const seq2map::DisparityIO& empty = seq2map::DisparityIO())
    {
        dpio = empty;

        if (fn.empty()) return;
        try
        {

        }
        catch (std::exception& ex)
        {
            E_ERROR << "error loading seq2map::DisparityIO";
            E_ERROR << ex.what();
        }
    }
}

#endif //DISPARITY_HPP

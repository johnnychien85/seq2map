#ifndef CALIBGRAPHBUILDER_HPP
#define CALIBGRAPHBUILDER_HPP
#include "calibgraph.hpp"

using namespace seq2map;

class CalibGraphBuilder
{
public:
    /* ctor */ CalibGraphBuilder(size_t cams, size_t views) : m_cams(cams), m_views(views) {}
    /* dcot */ virtual ~CalibGraphBuilder() {}
    bool SetFileList(const String& filelist);
    bool Build(CalibGraph& graph) const;
    bool SetTargetDef(const String& def)             { return m_targetDef.FromString(def); }
    String GetTargetDef() const                      { return m_targetDef.ToString(); }
    inline void SetAdaptiveThresholding(bool enable) { SetFlag(CV_CALIB_CB_ADAPTIVE_THRESH, enable); }
    inline void SetImageNormalisation(bool enable)   { SetFlag(CV_CALIB_CB_NORMALIZE_IMAGE, enable); }
    inline void SetFastCheck(bool enable)            { SetFlag(CV_CALIB_CB_FAST_CHECK,      enable); }
    inline void SetSubPixelWinSize(size_t width)     { m_subpxWinSize.height = m_subpxWinSize.width = width; }
    inline void SetSubPixelTerm(size_t iters, double eps) { m_subpxTerm = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, iters, eps); }

private:
    class TargetDef
    {
    public:
        /* ctor */ TargetDef() {}
        /* ctor */ TargetDef(const cv::Size& patternSize, const cv::Size2f& patternMetric) : patternSize(patternSize), patternMetric(patternMetric) {}
        bool FromString(const String& def);
        String ToString() const; 
        bool IsOkay() const { return patternSize.area() >= 4 && patternMetric.area() > 0; }
        void GetObjectPoints(Points3F& pts, Indices& corners) const;

        cv::Size   patternSize;
        cv::Size2f patternMetric;
    };

    class ImageFileList
    {
    public:
        inline Path operator() (size_t cam, size_t img) const { return m_list[cam][img]; }
        bool FromPattern(const String& pattern, size_t cams, size_t imgs);
        bool FromFile(const Path& listFile, size_t cams, size_t imgs);
        inline bool IsOkay() const { return !m_list.empty(); }
        bool CheckExists() const;
        static bool CheckPattern(const String& pattern);
    protected:
        std::vector<Strings> m_list;
    };

    void SetFlag(int flag, bool enable);

    const size_t m_cams;
    const size_t m_views;
    int m_flag;
    cv::TermCriteria m_subpxTerm;
    TargetDef m_targetDef;
    ImageFileList m_imageFiles;
    cv::Size m_subpxWinSize;
};

#endif // CALIBGRAPHBUILDER_HPP

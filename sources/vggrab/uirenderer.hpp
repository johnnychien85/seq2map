#ifndef UIRENDERER_HPP
#define UIRENDERER_HPP
#include "grabber.hpp"
#include "recorder.hpp"

class LabelDrawer
{
public:
    LabelDrawer()
        : FontFace(cv::FONT_HERSHEY_COMPLEX_SMALL), FontScale(1), FontWeight(1), BorderWidth(0),
        BackgroundColour(cv::Scalar(0,0,0)), ForegroundColour(cv::Scalar(255,255,255)),
        BorderColour(ForegroundColour), DrawBackground(false) {};

    void Draw(cv::Mat& canvas, const cv::Point& origin, const String& text, cv::Size& labelSize);
    void Draw(cv::Mat& canvas, const cv::Point& origin, const String& text);

    int FontFace;
    double FontScale;
    int FontWeight;
    int BorderWidth;
    cv::Scalar BackgroundColour;
    cv::Scalar ForegroundColour;
    cv::Scalar BorderColour;
    bool DrawBackground;
};

class UIRenderer
{
public:
    UIRenderer() {}
    virtual bool Draw(cv::Mat& canvas) = 0;
};

class StereoImageRenderer : UIRenderer
{
public:
    enum RenderingMode {UNDEFINED, FUSION, SIDE, CAM0, CAM1};

    StereoImageRenderer(ImageGrabber& cam0, ImageGrabber& cam1)
        : _cam0(cam0), _cam1(cam1), Origin(32, -32) {};
    inline void SetMode(const RenderingMode& mode) {_mode = mode;};
    inline RenderingMode GetMode() const {return _mode;}
    virtual bool Draw(cv::Mat& canvas);

    static const std::vector<RenderingMode> ListedModes;

    cv::Point Origin;

protected:
    void DrawLabels(cv::Mat& canvas);

    ImageGrabber& _cam0;
    ImageGrabber& _cam1;
    LabelDrawer _labelDrawer;

private:
    static inline std::string _GetModeDisplay(const RenderingMode& mode);
    static inline cv::Size _GetCanvasSize(const cv::Size& size0, const cv::Size& size1, RenderingMode mode);
    static inline void _FuseImages(const cv::Mat& im0, const cv::Mat& im1, cv::Mat& canvas);

    RenderingMode _mode;
};

class BufferWriterStatsRenderer : UIRenderer
{
public:
    BufferWriterStatsRenderer(const BufferWriter::Ptr& writer, size_t numRecords = 100)
        : _writer(writer), _records(BpsRecords(numRecords)), _currentIdx(0), _bestIdx(0) {};
    virtual bool Draw(cv::Mat& canvas);

    cv::Rect Rectangle;

protected:
    typedef std::vector<double> BpsRecords;

    BufferWriter::Ptr _writer;
    BpsRecords _records;
    size_t _currentIdx;
    size_t _bestIdx;
};

class BufferUsageIndicator : UIRenderer
{
public:
    BufferUsageIndicator(const SyncBuffer::Ptr& buffer) : m_buffer(buffer) {}
    virtual bool Draw(cv::Mat& canvas);

    cv::Rect Rectangle;

protected:
    SyncBuffer::Ptr m_buffer;
};

class BufferRecorderStatsRenderer
{
public:
    BufferRecorderStatsRenderer(BufferRecorder& rec) : m_rec(rec), m_seq(0) {}
    virtual bool Draw(cv::Mat& canvas);

    cv::Point Origin;

protected:
    BufferRecorder& m_rec;
    size_t m_seq;
};

#endif // UIRENDERER_HPP

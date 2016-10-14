#include <boost\assign\list_of.hpp>
#include "grabber.hpp"
#include "uirenderer.hpp"

void LabelDrawer::Draw(cv::Mat& canvas, const cv::Point& origin, const String& text, cv::Size& labelSize)
{
    int baseline = 0;

    labelSize = cv::getTextSize(text, FontFace, FontScale, FontWeight, &baseline);
    //baseline += FontWeight;

    // draw background
    if (DrawBackground)
    {
        rectangle(canvas,
            origin + cv::Point(0, 0),
            origin + cv::Point(labelSize.width, -labelSize.height),
            BackgroundColour, cv::FILLED);
    }

    // draw the bounding box
    if (BorderWidth > 0)
    {
        int w = BorderWidth + 1;
        rectangle(canvas,
            origin + cv::Point(-w, w),
            origin + cv::Point(labelSize.width+w, -labelSize.height-w),
            BorderColour, BorderWidth);
    }

    putText(canvas, text, origin, FontFace, FontScale, ForegroundColour, FontWeight);
}

void LabelDrawer::Draw(cv::Mat& canvas, const cv::Point& origin, const String& text)
{
    cv::Size labelSize;
    Draw(canvas, origin, text, labelSize);
}

const std::vector<StereoImageRenderer::RenderingMode> StereoImageRenderer::ListedModes =
boost::assign::list_of(FUSION)(SIDE)(CAM0)(CAM1);

String StereoImageRenderer::_GetModeDisplay(const RenderingMode& mode)
{
    switch (mode)
    {
    case CAM0: return "CAM0"; break;
    case CAM1: return "CAM1"; break;
    case FUSION: return "FUSION"; break;
    case SIDE: return "SIDE"; break;
    default: return "UNDEFINED"; break;
    }
}

cv::Size StereoImageRenderer::_GetCanvasSize(const cv::Size& size0, const cv::Size& size1,
    StereoImageRenderer::RenderingMode mode)
{
    cv::Size canvasSize;

    switch (mode)
    {
    case CAM0:
        canvasSize = size0;
        break;
    case CAM1:
        canvasSize = size1;
        break;
    case FUSION:
        canvasSize.height = std::max(size0.height, size1.height);
        canvasSize.width = std::max(size0.width, size1.width);
        break;
    case SIDE:
        canvasSize.height = std::max(size0.height, size1.height);
        canvasSize.width = size0.width + size1.width;
        break;
    case UNDEFINED:
    default:
        E_ERROR << "undefined mode: " << mode;
    }

    return canvasSize;
}

void StereoImageRenderer::_FuseImages(const cv::Mat& im0, const cv::Mat& im1, cv::Mat& canvas)
{
    std::vector<cv::Mat> src, bgr;
    split(canvas, bgr);

    src.push_back(im0);
    src.push_back(im0 * 0.5f + im1 * 0.5f);
    src.push_back(im1);

    for (size_t i = 0; i < 3; i++)
    {
        src[i].copyTo(bgr[i](cv::Rect(0, 0, src[i].cols, src[i].rows)));
    }

    merge(bgr, canvas);
}

void StereoImageRenderer::DrawLabels(cv::Mat& canvas)
{
    cv::Scalar colour1(32, 64, 255);
    cv::Scalar colour2(255, 255, 255);

    _labelDrawer.FontScale = 0.8f;

    cv::Point pt = Origin;
    pt.x = pt.x > 0 ? pt.x : canvas.cols + pt.x;
    pt.y = pt.y > 0 ? pt.y : canvas.rows + pt.y;

    BOOST_FOREACH(RenderingMode mode, ListedModes) 
    {
        const String labelText = _GetModeDisplay(mode);
        bool active = (mode == _mode);

        cv::Size labelSize;

        if (active)
        {
            _labelDrawer.BorderWidth = 2;
            _labelDrawer.DrawBackground = true;
            _labelDrawer.ForegroundColour = _labelDrawer.BorderColour = colour1;
            _labelDrawer.BackgroundColour = colour2;
        }
        else
        {
            _labelDrawer.BorderWidth = 1;
            _labelDrawer.DrawBackground = false;
            _labelDrawer.ForegroundColour = _labelDrawer.BorderColour = colour2;
            _labelDrawer.BackgroundColour = colour1;
        }

        _labelDrawer.Draw(canvas, pt, labelText, labelSize);
        pt.x += labelSize.width + 10;
    }
}

bool StereoImageRenderer::Draw(cv::Mat& canvas)
{
    if (_mode == UNDEFINED)
    {
        E_ERROR << "rendering mode undefined";
        return false;
    }

    cv::Mat im0 = _cam0.GetImage();
    cv::Mat im1 = _cam1.GetImage();

    if (im0.empty() || im1.empty())
    {
        E_ERROR << "invalid image(s) retrieved";
        return false;
    }

    cv::Size neededSize = _GetCanvasSize(im0.size(), im1.size(), _mode);

    if (canvas.size() != neededSize ||
        canvas.type() != CV_8UC3)
    {
        canvas = cv::Mat::zeros(neededSize, CV_8UC3);
    }

    // colour conversion first
    //
    //  -----------------------------
    //  Mode      im0        im1
    //  -----------------------------
    //  CAM0      CV_8UC3    x
    //  CAM1      x          CV_8UC3
    //  FUSION    CV_8UC1    CV_8UC1
    //  SIDE      CV_8UC3    CV_8UC3
    //
    switch (_mode)
    {
    case CAM0:
        if (im0.channels() == 1) cvtColor(im0, im0, cv::COLOR_GRAY2BGR);
        break;
    case CAM1:
        if (im1.channels() == 1) cvtColor(im1, im1, cv::COLOR_GRAY2BGR);
        break;
    case FUSION:
        if (im0.channels() == 3) cvtColor(im0, im0, cv::COLOR_BGR2GRAY);
        if (im1.channels() == 3) cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
        break;
    case SIDE:
        if (im0.channels() == 1) cvtColor(im0, im0, cv::COLOR_GRAY2BGR);
        if (im1.channels() == 1) cvtColor(im1, im1, cv::COLOR_GRAY2BGR);
        break;
    }

    switch (_mode)
    {
    case CAM0: canvas = im0; break;
    case CAM1: canvas = im1; break;
    case FUSION:
        _FuseImages(im0, im1, canvas);
        break;
    case SIDE:
        im0.copyTo(canvas(cv::Rect(0, 0, im0.cols, im0.rows)));
        im1.copyTo(canvas(cv::Rect(im0.cols, 0, im0.cols, im0.rows)));
        break;
    default:
        E_ERROR << "unsupported rendering mode";
        return false;
    }

    DrawLabels(canvas);

    return true;
}

bool BufferWriterStatsRenderer::Draw(cv::Mat& canvas)
{
    Speedometre metre = _writer->GetMetre();
    LabelDrawer labeller;
    cv::Size labelSize;
    cv::Point pt(Rectangle.x, Rectangle.y);
    std::vector<String> labels;
    std::stringstream ss1, ss2, ss3;
    cv::Rect plotRegion;

    ss1 << "seq=" << std::setfill('0') << std::setw(8) << _writer->GetSeq(); 
    ss2 << std::fixed << std::setprecision(2) << std::setfill('0') << std::setw(2) << "fps=" << metre.GetFrequency();
    ss3 << std::fixed << std::setprecision(2) << std::setfill('0') << std::setw(2) << "delay=" << _writer->GetDelta() << "ms";

    labels.push_back(ss1.str());
    labels.push_back(ss2.str());
    labels.push_back(ss3.str());

    _records[_currentIdx] = metre.GetSpeed();
    _bestIdx = _records[_currentIdx] > _records[_bestIdx] ? _currentIdx : _bestIdx;

    rectangle(canvas, Rectangle, cv::Scalar(255, 255, 255));

    labeller.BackgroundColour = labeller.BorderColour = cv::Scalar(255, 255, 255);
    labeller.ForegroundColour = cv::Scalar(0, 0, 0);
    labeller.FontScale = 0.6;
    labeller.BorderWidth = 0;
    labeller.DrawBackground = true;
    labeller.Draw(canvas, pt, _writer->ToString(), labelSize);

    // draw seq
    pt.x += 4;
    pt.y += Rectangle.height - 10;
    labeller.ForegroundColour = cv::Scalar(255, 255, 255);
    labeller.DrawBackground = false;

    BOOST_FOREACH(const String& labelText, labels)
    {
        labeller.Draw(canvas, pt, labelText, labelSize);
        pt.x += labelSize.width + 3;
    }

    plotRegion.x = Rectangle.x + 1;
    plotRegion.y = Rectangle.y + 4;
    plotRegion.width = Rectangle.width - 1;
    plotRegion.height = Rectangle.height - labelSize.height - (plotRegion.y - Rectangle.y) - 14;

    //rectangle(canvas, plotRegion, Scalar(255, 255, 255));

    cv::Point pt0, pt1(0, 0);
    for (size_t i = 0; i < _records.size(); i++)
    {
        size_t idx = (_currentIdx - i) % _records.size();

        pt0 = pt1;
        pt1.x = plotRegion.x + (plotRegion.width - 1) * (1.0f - (double)i / (double)_records.size());
        pt1.y = plotRegion.y + (plotRegion.height - 1) * (1.0f - 0.8f * (double)_records[idx] / (double)_records[_bestIdx]);
        //pt0.x = plotRegion.x + plotRegion.width * (1.0f - (double)i / (double)_records.size());
        //pt0.y = plotRegion.y + plotRegion.height;
        //pt1.x = pt0.x;
        //pt1.y = pt0.y - (double)plotRegion.height * 0.9f * (double)_records[idx] / (double)_records[_bestIdx];

        if (i == 0)
        {
            std::stringstream ss;
            ss << (int) floor((metre.GetSpeed()) / 1024 / 1024) << "MB/s";
            labeller.BorderWidth = 1;
            labeller.FontScale = 0.5;
            labeller.Draw(canvas, pt1 + cv::Point(3,0), ss.str());
        }
        else
        {
            line(canvas, pt0, pt1, labeller.ForegroundColour, 2);
        }
    }

    _currentIdx = (_currentIdx + 1) % _records.size();

    return true;
}

bool BufferUsageIndicator::Draw(cv::Mat& canvas)
{
    const size_t n = 10;
    const double spacing = 2;
    double k = m_buffer->GetUsage();
    double freePercent = (1.0f - k) * 100.0f;

    LabelDrawer labeller;
    cv::Point pt(Rectangle.x, Rectangle.y);
    cv::Size labelSize;
    std::stringstream ss;
    ss << "BUFFER: " << std::setprecision(0) << std::fixed << freePercent << "% FREE";

    cv::Scalar colour1((1.0f - k) * 255.0f, (1.0f - 0.5f * k) * 255.0f, 255.0f);
    cv::Scalar colour2(255.0f, 255.0f, 255.0f);

    pt.x = pt.x > 0 ? pt.x : pt.x + canvas.cols;
    pt.y = pt.y > 0 ? pt.y : pt.y + canvas.rows;

    labeller.FontScale = 0.5;
    labeller.BorderWidth = 0;
    labeller.DrawBackground = false;
    labeller.ForegroundColour = colour1;
    labeller.Draw(canvas, pt, ss.str(), labelSize);

    pt.y += labelSize.height;
    labelSize.height = Rectangle.height - labelSize.height;

    double w = (double)Rectangle.width / n;
    size_t p = (size_t) (n * (1.0f - k));

    for (size_t i = 0; i < n; i++)
    {
        if (i <= p) rectangle(canvas, pt, pt + cv::Point(w - spacing, labelSize.height), colour1, -1);
        else        rectangle(canvas, pt, pt + cv::Point(w - spacing, labelSize.height), colour2, +1);

        pt.x += w;
    }

    return true;
}

bool BufferRecorderStatsRenderer::Draw(cv::Mat& canvas)
{
    if (!m_rec.IsRecording()) return true;

    cv::Size labelSize;
    LabelDrawer labeller;
    cv::Point pt = Origin;

    pt.x = pt.x > 0 ? pt.x : pt.x + canvas.cols;
    pt.y = pt.y > 0 ? pt.y : pt.y + canvas.rows;

    if (m_seq < 7)
    {
        labeller.BackgroundColour = cv::Scalar(0, 0, 255);
        labeller.ForegroundColour = labeller.BorderColour = cv::Scalar(255, 255, 255);        
    }
    else
    {
        labeller.BackgroundColour = cv::Scalar(255, 255, 255);
        labeller.ForegroundColour = labeller.BorderColour = cv::Scalar(0, 0, 255);
    }

    labeller.FontFace = cv::FONT_HERSHEY_TRIPLEX;
    labeller.FontScale = 2;
    labeller.FontWeight = 3;
    labeller.DrawBackground = true;
    labeller.BorderWidth = 2;
    labeller.Draw(canvas, pt, " REC ", labelSize);

    std::stringstream ss;
    ss << m_rec.GetDropped() << " DROPPED / " << m_rec.GetWritten() << " WRITTEN";
    pt.y += labelSize.height;

    LabelDrawer statsLbl;
    statsLbl.FontFace = cv::FONT_HERSHEY_SIMPLEX;
    statsLbl.FontScale = 0.4;
    statsLbl.FontWeight = 1;
    statsLbl.ForegroundColour = cv::Scalar(192, 192, 192);
    statsLbl.DrawBackground = false;
    statsLbl.BorderWidth = 0;
    statsLbl.Draw(canvas, pt, ss.str());

    m_seq = (m_seq + 1) % 10;

    return true;
}

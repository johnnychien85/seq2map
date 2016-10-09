#include "grabber.hpp"
#include <sstream>
#include <boost/assign/list_of.hpp>

using namespace std;
using namespace cv;

//const int ImageGrabber::INDEX_ANY = -1;
int DummyImageGrabber::_newId = 0;

#ifdef WITH_STCAM
int StImageGrabber::_newId = 0;
#endif

void BufferData::Create(size_t seq, const Time& timestamp)
{
    _seq    = seq;
    _time   = timestamp;
    _synced = true;
    _clear  = false;
    _numReads = _numWrites = 0;

    //E_TRACE << "frame " << seq << " created, ts=" << timestamp;
}

double SyncBuffer::GetUsage()
{
    boost::lock_guard<boost::mutex> locker(_mtx);
    return _seqHead > _seqTail ? (double)(_seqHead - _seqTail) / _bufferSize : 0.0f;
}

bool SyncBuffer::_TryWrite(size_t& seq, unsigned long& bestDelta, BufferData*& data)
{
    Time now = UNow();

    for(;;)
    {
        // pointer to the #seq
        data = &(*this)[seq];
        DataLocker locker(*data);

        // first write on this data slot
        if (seq != data->_seq)
        {
            if (!data->_clear)
            {
                // the first write is not possible
                E_WARNING << "buffer overflow!!";
                //E_TRACE << "creation of frame " << seq << " failed, old frame " << data->_seq << " is not clear";

                return false;
            }

            _timestamp += boost::posix_time::milliseconds(_interval);
            data->Create(seq, _timestamp);
        }

        assert(data->_numReads == 0);
        assert(data->_numWrites < _numWriters);

        // delta = |t - t0|
        long delta = abs(data->GetDelta(now));

        bool inSlot = delta <= _halfInterval;
        bool synced = delta <= bestDelta;

        //E_TRACE << "seq=" << seq << " delta=" << delta << " ts=" << now;

        //         
        //   ||                   ||                   ||  
        // --||-----[   o   ]-----||-----[   o   ]-----||--
        //   ||          \        ||                   ||
        //                timestamp
        if (synced)
        {
            boost::lock_guard<boost::mutex> locker(_mtx);
            _seqHead = _seqHead > seq ? _seqHead : seq;

            bestDelta = delta;

            return true; // write immediately for synced data
        }

        // don't write this
        if (inSlot) return false;

        // out of sync
        data->_synced &= bestDelta < _sigma; // ever synced before advancing to next data frame?
        data->_numWrites++; // writing committed
        bestDelta = _sigma;
        seq++;

        // E_TRACE << "commited << " << (seq - 1) << " synced=" << data->_synced;
    }
}

bool SyncBuffer::_TryRead(size_t seq, BufferData*& data)
{
    data = &(*this)[seq];
    DataLocker locker(*data);

    return !data->_clear && data->_numWrites == _numWriters;
}

void SyncBuffer::_CommitRead(size_t& seq)
{
    BufferData& data = (*this)[seq];
    DataLocker locker(data);

    assert(data._numReads < _numReaders);

    data._numReads++;
    seq++;

    if (data._numReads == _numReaders) 
    {
        boost::lock_guard<boost::mutex> locker(_mtx);
        _seqTail = seq;

        data._clear = true; // completely consumed
    }
}

bool BufferThread::Start()
{
    if (!_IsOkay()) return false;
    if (IsRunning())
    {
        E_WARNING << "thread already launched";
        return false;
    }

    _speedometre.Reset();
    _speedometre.Start();
    _thread = boost::thread(_Entry, this);

    return true;
}

void BufferThread::Stop()
{
    if (!_IsOkay()) return;
    if (!IsRunning())
    {
        E_WARNING << "thread not launched";
        return;
    }

    _thread.interrupt();
    _thread.join();

    return;
}

void BufferThread::_Entry(BufferThread* thread)
{
    E_INFO << "thread #" <<  thread->GetId() << " launched";

    try
    {
        for (;;)
        {
            thread->_Loop();
            boost::this_thread::interruption_point();
        }
    }
    catch (boost::thread_interrupted const&)
    {
        //...
    }

    E_INFO << "thread #" <<  thread->GetId() << " terminated";

    return;
}

void BufferWriter::_Loop()
{
    BufferData* data;
    if (_GrabNextData() && _buffer._TryWrite(_seq, _bestDelta, data))
    {
        size_t bytesWritten = _Write(*data);  
        _speedometre.Update(bytesWritten);
        //E_TRACE << "frame " << _seq << " written with " << bytesWritten << " bytes, delta=" << _bestDelta;
    }
}

void BufferReader::_Loop()
{
    BufferData* data = NULL;
    if (_buffer._TryRead(_seq, data))
    {
        size_t bytesRead = _Read(*data);
        _buffer._CommitRead(_seq);
        _speedometre.Update(bytesRead);
    }
}

/******************************************************************************
* Image Grabber & Subclasses
******************************************************************************/

bool ImageGrabber::_GrabNextData()
{
    boost::lock_guard<boost::mutex> locker(_mtx);

    if (_image.empty())
    {
        Size imageSize;
        int imageType;

        if (!_GetImageInfo(imageSize, imageType))
        {
            E_ERROR << "error retrieving image size";
            return false;
        }

        _image = Mat::zeros(imageSize, imageType);
    }

    return _Grab(_image);
}

size_t ImageGrabber::_Write(BufferData& data)
{
    TimedImage& image = _Pick(data);
    Mat& im = image.payload;

    if (im.empty())
    {
        im = _image.clone();
    }
    else
    {
        assert(im.total() == _image.total());
        memcpy(im.data, _image.data, im.total() * im.elemSize());
    }

    image.Touch();
    return im.total() * im.elemSize();
}

Mat ImageGrabber::GetImage()
{
    boost::lock_guard<boost::mutex> locker(_mtx);
    return _image.clone();
}

bool DummyImageGrabber::_GetImageInfo(Size& imageSize, int& type) const
{
    imageSize = Size(1024, 768);
    type = CV_8UC3;

    return true;
}

string DummyImageGrabber::ToString() const
{
    stringstream ss;
    ss << "Dummy Camera #" << _index;

    return ss.str();
}

bool DummyImageGrabber::_Grab(Mat& im)
{
    Mat noise = Mat(320, 240, CV_8UC3);
    randu(noise, Scalar::all(0), Scalar::all(255));
    resize(noise, im, im.size());

    return true;
}

#ifdef WITH_STCAM
StImageGrabber::StImageGrabber(SyncBuffer& buffer) :
    ImageGrabber(buffer), _index(_newId++), _handle(NULL)
{
    DWORD dwCameraId;

    _handle = StCam_Open(_index);

    if (!_IsOkay())
    {
        E_ERROR << "error opening capture #" << _index;
        return;
    }

    if (!StCam_StartTransfer(_handle))
    {
        StCam_Close(_handle);
        _handle = NULL;

        E_ERROR << "error starting transfer from capture #" << _index;
        return;
    }

    _cameraName = _GetStCamName();

    E_INFO << "Sentech Camera #" << _index << " (" << _cameraName << ") opened";
}

StImageGrabber::~StImageGrabber()
{
    if (!_IsOkay()) return;

    if (!StCam_StopTransfer(_handle))
    {
        E_ERROR << "error stopping transfer from capture #" << _index;
    }

    StCam_Close(_handle);
    _handle = NULL;
}

void StImageGrabber::_LogStCamError(const string& message) const
{
    DWORD dwLastErrorNo = StCam_GetLastError(_handle);
    E_ERROR << message << " (idx=" << _index << ")";
    E_ERROR << "last error of capture #" << _index << " is " << dwLastErrorNo;
}

std::string StImageGrabber::_GetStCamName() const
{
    stringstream ss;
    const size_t bufferSize = 64;
    DWORD dwCameraId;
    TCHAR szCameraId[bufferSize];

    //if (!StCam_GetProductNameA(_handle, szBuffer, dwBufferSize))
    //{
    //    E_WARNING << "error reading camera serial of capture #" << _index;
    //    return "UNKNOWN";
    //}

    memset(szCameraId, 0, sizeof(TCHAR) * bufferSize);

    if (!StCam_ReadCameraUserID(_handle, &dwCameraId, szCameraId, bufferSize))
    {
        E_WARNING << "error reading camera ID of capture #" << _index;
        return "UNIDENTIFIED";
    }

    ss << szCameraId;

    return ss.str();
}

bool StImageGrabber::_GetImageInfo(cv::Size& imageSize, int& type) const
{
    BOOL   bResult = TRUE;
    DWORD  dwReserved;
    DWORD  dwOffsetX, dwOffsetY;
    DWORD  dwWidth, dwHeight;
    WORD   wScanMode;
    DWORD  dwPixelFormat;

    //Get Image Size
    bResult = StCam_GetImageSize(
        _handle, &dwReserved, &wScanMode,
        &dwOffsetX, &dwOffsetY, &dwWidth, &dwHeight);

    if (!bResult)
    {
        _LogStCamError("error retrieving image size");
        return false;
    }

    //Get Preview Pixel Format
    bResult = StCam_GetPreviewPixelFormat(_handle, &dwPixelFormat);

    if (!bResult)
    {
        _LogStCamError("error retrieving pixel format");
        return false;
    }

    imageSize.height = dwHeight;
    imageSize.width  = dwWidth;

    switch (dwPixelFormat)
    {
    case STCAM_PIXEL_FORMAT_24_BGR: type = CV_8UC3;	 break;
    case STCAM_PIXEL_FORMAT_32_BGR: type = CV_8UC4;	 break;
    default:
        E_ERROR << "unsupported pixel format " << dwPixelFormat << " (idx=" << _index << ")";
        return false;
    }

    return true;
}

bool StImageGrabber::_Grab(Mat& im)
{
    //take a snapshot
    DWORD dwNumberOfByteTrans, dwFrameNo;
    DWORD dwMilliseconds = 1000;
    BOOL  bResult;

    bResult = StCam_TakePreviewSnapShot(
        _handle, im.data, im.total() * im.elemSize(),
        &dwNumberOfByteTrans, &dwFrameNo, dwMilliseconds);

    if(!bResult)
    {
        _LogStCamError("error taking preview snapshot");
        return false;
    }
}
#endif

string BufferRecorder::_GetDateTimeString(const string& format, const Time& when)
{
    using namespace boost::posix_time; // for time_facet

    stringstream ss;
    time_facet* facet = new time_facet(format.c_str());

    ss.imbue(locale(ss.getloc(), facet));
    ss << when;

    return ss.str();
}

void BufferRecorder::StartRecording()
{
    if (_isRecording) return;

    _recStorage.Create(_seqDirPath / _GetDateCode() / _GetTimeCode(), 2);
    _isRecording = true;

    _numDropped = _numWritten = 0;
}

size_t BufferRecorder::_Read(BufferData& data)
{
    if (!data.IsSynced())
    {
        _numDropped++;
        return 0;
    }

    size_t bytesWritten = 0;
    size_t imageDataBytes = 0;

    vector<Mat> images;

    BOOST_FOREACH (const TimedImage& image, data.images)
    {
        Mat im = image.payload;

        images.push_back(im.clone());
        imageDataBytes += im.elemSize() * im.total();
    }

    vector<VOStorage*> storages;

    if (_isRecording)    storages.push_back(&_recStorage);
    if (_doSnapshooting) storages.push_back(&_snapStorage);

    _doSnapshooting = false;

    BOOST_FOREACH (VOStorage* storage, storages)
    {
        size_t seq = storage->GetNumOfFrames();
        stringstream ss; ss << setw(8) << setfill('0') << seq;
        storage->AddFrame(ss.str(), images);
        _numWritten++;
        bytesWritten += imageDataBytes;
    }

    return bytesWritten;
}

void LabelDrawer::Draw(Mat& canvas, const Point& origin, const string& text, Size& labelSize)
{
    int baseline = 0;

    labelSize = getTextSize(text, FontFace, FontScale, FontWeight, &baseline);
    //baseline += FontWeight;

    // draw background
    if (DrawBackground)
    {
        rectangle(canvas,
            origin + Point(0, 0),
            origin + Point(labelSize.width, -labelSize.height),
            BackgroundColour, cv::FILLED);
    }

    // draw the bounding box
    if (BorderWidth > 0)
    {
        int w = BorderWidth + 1;
        rectangle(canvas,
            origin + Point(-w, w),
            origin + Point(labelSize.width+w, -labelSize.height-w),
            BorderColour, BorderWidth);
    }

    putText(canvas, text, origin, FontFace, FontScale, ForegroundColour, FontWeight);
}

void LabelDrawer::Draw(Mat& canvas, const Point& origin, const string& text)
{
    Size labelSize;
    Draw(canvas, origin, text, labelSize);
}

const vector<StereoImageRenderer::RenderingMode> StereoImageRenderer::ListedModes =
boost::assign::list_of
(RenderingMode::FUSION)(RenderingMode::SIDE)(RenderingMode::CAM0)(RenderingMode::CAM1);

string StereoImageRenderer::_GetModeDisplay(const RenderingMode& mode)
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

Size StereoImageRenderer::_GetCanvasSize(const Size& size0, const Size& size1,
    StereoImageRenderer::RenderingMode mode)
{
    cv::Size canvasSize;

    switch (mode)
    {
    case RenderingMode::CAM0:
        canvasSize = size0;
        break;
    case RenderingMode::CAM1:
        canvasSize = size1;
        break;
    case RenderingMode::FUSION:
        canvasSize.height = max(size0.height, size1.height);
        canvasSize.width = max(size0.width, size1.width);
        break;
    case RenderingMode::SIDE:
        canvasSize.height = max(size0.height, size1.height);
        canvasSize.width = size0.width + size1.width;
        break;
    case RenderingMode::UNDEFINED:
    default:
        E_ERROR << "undefined mode: " << mode;
    }

    return canvasSize;
}

void StereoImageRenderer::_FuseImages(const cv::Mat& im0, const cv::Mat& im1, cv::Mat& canvas)
{
    vector<Mat> src, bgr;
    split(canvas, bgr);

    src.push_back(im0);
    src.push_back(im0 * 0.5f + im1 * 0.5f);
    src.push_back(im1);

    for (size_t i = 0; i < 3; i++)
    {
        src[i].copyTo(bgr[i](Rect(0, 0, src[i].cols, src[i].rows)));
    }

    merge(bgr, canvas);
}

void StereoImageRenderer::_DrawLabels(Mat& canvas)
{
    Scalar colour1 = Scalar(32, 64, 255);
    Scalar colour2 = Scalar(255, 255, 255);

    _labelDrawer.FontScale = 0.8f;

    Point pt = Origin;
    pt.x = pt.x > 0 ? pt.x : canvas.cols + pt.x;
    pt.y = pt.y > 0 ? pt.y : canvas.rows + pt.y;

    BOOST_FOREACH(RenderingMode mode, ListedModes) 
    {
        const string labelText = _GetModeDisplay(mode);
        bool active = (mode == _mode);

        Size labelSize;

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

bool StereoImageRenderer::Draw(Mat& canvas)
{
    if (_mode == RenderingMode::UNDEFINED)
    {
        E_ERROR << "rendering mode undefined";
        return false;
    }

    Mat im0 = _cam0.GetImage();
    Mat im1 = _cam1.GetImage();

    if (im0.empty() || im1.empty())
    {
        E_ERROR << "invalid image(s) retrieved";
        return false;
    }

    Size neededSize = _GetCanvasSize(im0.size(), im1.size(), _mode);

    if (canvas.size() != neededSize ||
        canvas.type() != CV_8UC3)
    {
        canvas = Mat::zeros(neededSize, CV_8UC3);
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
    case RenderingMode::CAM0:
        if (im0.channels() == 1) cvtColor(im0, im0, COLOR_GRAY2BGR);
        break;
    case RenderingMode::CAM1:
        if (im1.channels() == 1) cvtColor(im1, im1, COLOR_GRAY2BGR);
        break;
    case RenderingMode::FUSION:
        if (im0.channels() == 3) cvtColor(im0, im0, COLOR_BGR2GRAY);
        if (im1.channels() == 3) cvtColor(im1, im1, COLOR_BGR2GRAY);
        break;
    case RenderingMode::SIDE:
        if (im0.channels() == 1) cvtColor(im0, im0, COLOR_GRAY2BGR);
        if (im1.channels() == 1) cvtColor(im1, im1, COLOR_GRAY2BGR);
        break;
    }

    switch (_mode)
    {
    case RenderingMode::CAM0: canvas = im0; break;
    case RenderingMode::CAM1: canvas = im1; break;
    case RenderingMode::FUSION:
        _FuseImages(im0, im1, canvas);
        break;
    case RenderingMode::SIDE:
        im0.copyTo(canvas(Rect(0, 0, im0.cols, im0.rows)));
        im1.copyTo(canvas(Rect(im0.cols, 0, im0.cols, im0.rows)));
        break;
    default:
        E_ERROR << "unsupported rendering mode";
        return false;
    }

    _DrawLabels(canvas);

    return true;
}

bool BufferWriterStatsRenderer::Draw(cv::Mat& canvas)
{
    Speedometre metre = _writer.GetMetre();
    LabelDrawer labeller;
    Size labelSize;
    Point pt(Rectangle.x, Rectangle.y);
    vector<string> labels;
    stringstream ss1, ss2, ss3;
    Rect plotRegion;

    ss1 << "seq=" << setfill('0') << setw(8) << _writer.GetSeq(); 
    ss2 << fixed << setprecision(2) << setfill('0') << setw(2) << "fps=" << metre.GetFrequency();
    ss3 << fixed << setprecision(2) << setfill('0') << setw(2) << "delay=" << _writer.GetDelta() << "ms";

    labels.push_back(ss1.str());
    labels.push_back(ss2.str());
    labels.push_back(ss3.str());

    _records[_currentIdx] = metre.GetSpeed();
    _bestIdx = _records[_currentIdx] > _records[_bestIdx] ? _currentIdx : _bestIdx;

    rectangle(canvas, Rectangle, Scalar(255, 255, 255));

    labeller.BackgroundColour = labeller.BorderColour = Scalar(255, 255, 255);
    labeller.ForegroundColour = Scalar(0, 0, 0);
    labeller.FontScale = 0.6;
    labeller.BorderWidth = 0;
    labeller.DrawBackground = true;
    labeller.Draw(canvas, pt, _writer.ToString(), labelSize);

    // draw seq
    pt.x += 4;
    pt.y += Rectangle.height - 10;
    labeller.ForegroundColour = Scalar(255, 255, 255);
    labeller.DrawBackground = false;

    BOOST_FOREACH(const string& labelText, labels)
    {
        labeller.Draw(canvas, pt, labelText, labelSize);
        pt.x += labelSize.width + 3;
    }

    plotRegion.x = Rectangle.x + 1;
    plotRegion.y = Rectangle.y + 4;
    plotRegion.width = Rectangle.width - 1;
    plotRegion.height = Rectangle.height - labelSize.height - (plotRegion.y - Rectangle.y) - 14;

    //rectangle(canvas, plotRegion, Scalar(255, 255, 255));

    Point pt0, pt1(0, 0);
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
            stringstream ss;
            ss << (int) floor((metre.GetSpeed()) / 1024 / 1024) << "MB/s";
            labeller.BorderWidth = 1;
            labeller.FontScale = 0.5;
            labeller.Draw(canvas, pt1 + Point(3,0), ss.str());
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
    double k = _buffer.GetUsage();
    double freePercent = (1.0f - k) * 100.0f;

    LabelDrawer labeller;
    Point pt(Rectangle.x, Rectangle.y);
    Size labelSize;
    stringstream ss;
    ss << "BUFFER: " << setprecision(0) << fixed << freePercent << "% FREE";

    Scalar colour1((1.0f - k) * 255.0f, (1.0f - 0.5f * k) * 255.0f, 255.0f);
    Scalar colour2(255.0f, 255.0f, 255.0f);

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
        if (i <= p) rectangle(canvas, pt, pt + Point(w - spacing, labelSize.height), colour1, -1);
        else        rectangle(canvas, pt, pt + Point(w - spacing, labelSize.height), colour2, +1);

        pt.x += w;
    }

    return true;
}

bool BufferRecorderStatsRenderer::Draw(cv::Mat& canvas)
{
    if (!_rec.IsRecording()) return true;

    Size labelSize;
    LabelDrawer labeller;
    Point pt = Origin;

    pt.x = pt.x > 0 ? pt.x : pt.x + canvas.cols;
    pt.y = pt.y > 0 ? pt.y : pt.y + canvas.rows;

    if (_seq < 7)
    {
        labeller.BackgroundColour = Scalar(0, 0, 255);
        labeller.ForegroundColour = labeller.BorderColour = Scalar(255, 255, 255);        
    }
    else
    {
        labeller.BackgroundColour = Scalar(255, 255, 255);
        labeller.ForegroundColour = labeller.BorderColour = Scalar(0, 0, 255);
    }

    labeller.FontFace = cv::FONT_HERSHEY_TRIPLEX;
    labeller.FontScale = 2;
    labeller.FontWeight = 3;
    labeller.DrawBackground = true;
    labeller.BorderWidth = 2;
    labeller.Draw(canvas, pt, " REC ", labelSize);

    stringstream ss;
    ss << _rec.GetDropped() << " DROPPED / " << _rec.GetWritten() << " WRITTEN";
    pt.y += labelSize.height;

    LabelDrawer statsLbl;
    statsLbl.FontFace = cv::FONT_HERSHEY_SIMPLEX;
    statsLbl.FontScale = 0.4;
    statsLbl.FontWeight = 1;
    statsLbl.ForegroundColour = Scalar(192, 192, 192);
    statsLbl.DrawBackground = false;
    statsLbl.BorderWidth = 0;
    statsLbl.Draw(canvas, pt, ss.str());

    _seq = (_seq + 1) % 10;

    return true;
}

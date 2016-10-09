#ifndef GRABBER_HPP
#define GRABBER_HPP
//#define WITH_STCAM

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <seq2map/common.hpp>
#include "storage.hpp"

using namespace seq2map;

typedef boost::posix_time::ptime Time;

template<class T> class TimedData
{
public:
    inline void Touch() {time = SyncBuffer::UNow();}

    Time time;
    T payload;
};

typedef TimedData<cv::Mat> TimedImage;

class BufferData
{
public:
    TimedImage images[2];

    inline void Lock() {_mtx.lock();}
    inline void Unlock() {_mtx.unlock();}
    inline bool IsWritable() const {return _numReads == _numReads;}
    inline bool IsEmpty() const {return _numWrites == 0;}
    inline void Create(size_t siq, const Time& timestamp);
    inline long GetDelta(const Time& time) const {return (time - _time).total_milliseconds();}
    inline void Dispose() {_clear = true;}
    inline bool IsSynced() const {return _synced;}

protected:
    friend class SyncBuffer;
    BufferData() : _seq((size_t)-1), _numReads(0), _numWrites(0), _clear(true), _synced(true) {}
    ~BufferData() {}

    size_t _seq;
    size_t _numReads;
    size_t _numWrites;
    bool _clear;
    bool _synced;
    boost::mutex _mtx;
    Time _time;
};

class DataLocker
{
public:
    DataLocker(BufferData& data) : _data(data) {_data.Lock();}
    ~DataLocker() {_data.Unlock();}

private:
    BufferData& _data;
};

class SyncBuffer
{
public:
    SyncBuffer(size_t numWriters, size_t numReaders, size_t bufferSize, double fps, double epsilon)
        : _numWriters(numWriters), _numReaders(numReaders), _bufferSize(bufferSize),
        _interval(1000.0f / fps), _halfInterval(_interval * 0.5f), _sigma(_halfInterval * epsilon),
        _data(new BufferData[bufferSize]), _timestamp(UNow()), _seqHead(0), _seqTail(0)
    {assert(fps > 0 && epsilon > 0.0f && epsilon < 1.0f);};
    ~SyncBuffer() {delete[] _data;};
    unsigned long GetInterval() const {return _interval;};
    double GetUsage();

    static inline Time UNow() {return boost::posix_time::microsec_clock::local_time();};

protected:
    friend class BufferWriter;
    friend class BufferReader;
    bool _TryWrite(size_t& seq, unsigned long& bestDelta, BufferData*& data);
    bool _TryRead(size_t seq, BufferData*& data);
    void _CommitRead(size_t& seq);

private:
    BufferData& operator[] (size_t seq) {return _data[seq % _bufferSize];};

    BufferData* _data;
    const size_t _numReaders;
    const size_t _numWriters;
    const size_t _bufferSize;
    const unsigned long _interval;
    const unsigned long _halfInterval;
    const unsigned long _sigma;
    Time _timestamp;
    size_t _seqHead;
    size_t _seqTail;
    boost::mutex _mtx;
};

class BufferThread
{
public:
    BufferThread(SyncBuffer& buffer) : _buffer(buffer), _seq(0) {};
    bool Start();
    void Stop();
    boost::thread::id GetId() const {return _thread.get_id();};
    inline bool IsRunning() const {return _thread.get_id() != boost::thread::id();};
    inline size_t GetSeq() const {return _seq;};
    inline const Speedometre& GetMetre() const {return _speedometre;};
    virtual std::string ToString() const = 0;

protected:
    virtual bool _IsOkay() const = 0;
    virtual void _Loop() = 0;
    SyncBuffer& _buffer;
    size_t _seq;
    Speedometre _speedometre;

private:
    static void _Entry(BufferThread* thread);
    boost::thread _thread;
};

class BufferWriter : public BufferThread
{
public:
    BufferWriter(SyncBuffer& buffer) : BufferThread(buffer), _bestDelta(_buffer.GetInterval()/2) {};
    inline unsigned long GetDelta() const {return _bestDelta;};
    virtual std::string ToString() const = 0;

protected:
    virtual void _Loop();
    virtual bool _GrabNextData() = 0;
    virtual size_t _Write(BufferData& data) = 0;

    unsigned long _bestDelta;
};

class BufferReader : public BufferThread
{
public:
    virtual std::string ToString() const = 0;

protected:
    BufferReader(SyncBuffer& buffer) : BufferThread(buffer) {};
    virtual void _Loop();
    virtual size_t _Read(BufferData& data) = 0;
};

/******************************************************************************
* Image Grabber & Subclasses
******************************************************************************/

class ImageGrabber : public BufferWriter
{
public:
    ImageGrabber(SyncBuffer& buffer) : BufferWriter(buffer) {};
    virtual ~ImageGrabber() {};
    virtual std::string ToString() const = 0;
    virtual cv::Mat GetImage();
    cv::Size GetImageSize() const {return _image.size();};

protected:
    virtual bool _GetImageInfo(cv::Size& imageSize, int& type) const = 0;
    inline virtual TimedImage& _Pick(BufferData& data) const = 0;
    virtual bool _Grab(cv::Mat& im) = 0;
    virtual bool _IsOkay() const = 0;
    virtual bool _GrabNextData();
    virtual size_t _Write(BufferData& data);

private:
    boost::mutex _mtx;
    cv::Mat _image;
};

class DummyImageGrabber : public ImageGrabber
{
public:
    DummyImageGrabber(SyncBuffer& buffer) : ImageGrabber(buffer), _index(_newId++) {};
    virtual ~DummyImageGrabber() {};
    virtual std::string ToString() const;

protected:
    virtual bool _GetImageInfo(cv::Size& imageSize, int& type) const;
    inline virtual TimedImage& _Pick(BufferData& data) const {return data.images[_index];};
    virtual bool _Grab(cv::Mat& im);
    virtual bool _IsOkay() const {return true;};

    static int _newId;
    size_t _index;
};

#ifdef WITH_STCAM
#include <windows.h>
#include <StCamD.h>
class StImageGrabber : public ImageGrabber
{
public:
    StImageGrabber(SyncBuffer& buffer);
    virtual ~StImageGrabber();
    virtual std::string ToString() const {return _cameraName;}

protected:
    virtual bool _GetImageInfo(cv::Size& imageSize, int& type) const;
    inline virtual TimedImage& _Pick(BufferData& data) const {return data.images[_index];};
    virtual bool _Grab(cv::Mat& im);
    virtual bool _IsOkay() const {return _handle != NULL;};

private:
    void _LogStCamError(const std::string& message) const;
    std::string _GetStCamName() const;

    static int _newId;
    size_t _index;
    HANDLE _handle;
    std::string _cameraName;
};
#endif

class BufferRecorder : public BufferReader
{
public:
    BufferRecorder(SyncBuffer& buffer, const Path& root)
        : BufferReader(buffer), _seqDirPath(root), _isRecording(false), _doSnapshooting(false),
        _snapStorage(root / _GetDateCode() / "snapshot", 2), _numDropped(0) {};
    inline bool IsRecording() const {return _isRecording;};
    inline size_t GetDroppedFrames() const {return _numDropped;}
    void StartRecording();
    void StopRecording() {_isRecording = false;}
    void Snapshot() {_doSnapshooting = true;}
    virtual std::string ToString() const {return "Recorder";}
    size_t GetDropped() const {return _numDropped;}
    size_t GetWritten() const {return _numWritten;}

protected:
    virtual bool _IsOkay() const {return true;};
    virtual size_t _Read(BufferData& data);

    static std::string _GetDateTimeString(
        const std::string& format,
        const Time& when = SyncBuffer::UNow());
    static std::string _GetDateCode() {return _GetDateTimeString("%Y%m%d");}
    static std::string _GetTimeCode() {return _GetDateTimeString("%H%M%S");}

    Path _seqDirPath;
    VOStorage _recStorage;
    VOStorage _snapStorage;
    bool _isRecording;
    bool _doSnapshooting;
    size_t _numDropped;
    size_t _numWritten;
};

class LabelDrawer
{
public:
    LabelDrawer()
        : FontFace(cv::FONT_HERSHEY_COMPLEX_SMALL), FontScale(1), FontWeight(1), BorderWidth(0),
        BackgroundColour(cv::Scalar(0,0,0)), ForegroundColour(cv::Scalar(255,255,255)),
        BorderColour(ForegroundColour), DrawBackground(false) {};

    void Draw(cv::Mat& canvas, const cv::Point& origin, const std::string& text, cv::Size& labelSize);
    void Draw(cv::Mat& canvas, const cv::Point& origin, const std::string& text);

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
    void _DrawLabels(cv::Mat& canvas);

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
    BufferWriterStatsRenderer(BufferWriter& writer, size_t numRecords)
        : _writer(writer), _records(BpsRecords(numRecords)), _currentIdx(0), _bestIdx(0) {};
    virtual bool Draw(cv::Mat& canvas);

    cv::Rect Rectangle;

protected:
    typedef std::vector<double> BpsRecords;

    BufferWriter& _writer;
    BpsRecords _records;
    size_t _currentIdx;
    size_t _bestIdx;
};

class BufferUsageIndicator : UIRenderer
{
public:
    BufferUsageIndicator(SyncBuffer& buffer) : _buffer(buffer) {}
    virtual bool Draw(cv::Mat& canvas);

    cv::Rect Rectangle;

protected:
    SyncBuffer& _buffer;
};

class BufferRecorderStatsRenderer
{
public:
    BufferRecorderStatsRenderer(BufferRecorder& rec) : _rec(rec), _seq(0) {}
    virtual bool Draw(cv::Mat& canvas);

    cv::Point Origin;

protected:
    BufferRecorder& _rec;
    size_t _seq;
};

#endif // GRABBER_HPP

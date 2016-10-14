
/*
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
*/


/******************************************************************************
* ImageGrabber Derivatives
******************************************************************************/

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
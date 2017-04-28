#ifndef GRABBER_STCAM_IMPL_HPP
#define GRABBER_STCAM_IMPL_HPP

#ifdef _WIN32 || _WIN64
#include <windows.h>
#endif
#include <StCamD.h>
#include "grabber.hpp"


class StImageGrabberBuilder : public ImageGrabberBuilder
{
public:
	virtual ~StImageGrabberBuilder() {}
	virtual ImageGrabber::Ptr Build(const SyncBuffer::Ptr& buffer, size_t index);
	virtual size_t GetDevices() const { return m_deviceList.size(); }
	virtual Strings GetDeviceIdList() const { return m_deviceList; }
private:
	friend class Factory<String, ImageGrabberBuilder>;
	StImageGrabberBuilder() { m_deviceList = EnumerateDevices(); }
	Strings EnumerateDevices() const;
	Strings m_deviceList;
};

class StImageGrabber : public ImageGrabber
{
public:
    virtual ~StImageGrabber();
    virtual String ToString() const { return m_name; }

protected:
	friend class StImageGrabberBuilder;
	StImageGrabber(size_t index, const String& name, const SyncBuffer::Ptr& buffer);

	virtual bool GetImageInfo(cv::Size& imageSize, int& type) const;
    virtual bool Grab(cv::Mat& im);
    virtual bool IsOkay() const {return m_handle != NULL;};

    //void _LogStCamError(const std::string& message) const;
    //std::string _GetStCamName() const;

private:
    HANDLE m_handle;
    const String m_name;
	cv::Size m_imageSize;
	int m_imageType;
};

Strings StImageGrabberBuilder::EnumerateDevices() const
{
	Strings deviceNames;
	const size_t devices = (size_t) StCam_CameraCount();
	const size_t bufferSize = 64;

	std::vector<HANDLE> handles;

	for (size_t idx = 0; idx < devices; idx++)
	{
		HANDLE handle = StCam_Open(idx);

		if (handle == NULL)
		{
			E_ERROR << "error opening capture " << idx;
			continue;
		}

		std::stringstream ss;
		DWORD dwCameraId;
		TCHAR szCameraId[bufferSize];

		//if (!StCam_GetProductNameA(m_handle, szBuffer, dwBufferSize))
		//{
		//    E_WARNING << "error reading camera serial of capture #" << _index;
		//    return "UNKNOWN";
		//}

		memset(szCameraId, 0, sizeof(TCHAR) * bufferSize);

		if (StCam_ReadCameraUserID(handle, &dwCameraId, szCameraId, bufferSize))
		{
			ss << szCameraId;
			E_INFO << "identified capture " << idx << ": " << ss.str();
		}
		else
		{
			ss << "StCam #" << idx << " UNIDENTIFIED";
			E_WARNING << "error reading camera ID from capture #" << idx;
		}
		
		handles.push_back(handle);
		deviceNames.push_back(ss.str());
	}

	BOOST_FOREACH(HANDLE h, handles)
	{
		StCam_Close(h);
	}

	return deviceNames;
}

ImageGrabber::Ptr StImageGrabberBuilder::Build(const SyncBuffer::Ptr& buffer, size_t index)
{
	if (index >= GetDevices())
	{
		E_ERROR << "index out of bound";
		return ImageGrabber::Ptr();
	}

	ImageGrabber::Ptr grabber(static_cast<ImageGrabber*>(new StImageGrabber(index, m_deviceList[index], buffer)));
	grabber->SetIndex(index);

	return grabber;
}

StImageGrabber::StImageGrabber(size_t index, const String& name, const SyncBuffer::Ptr& buffer)
: ImageGrabber(buffer), m_name(name)
{
	m_handle = StCam_Open(index);

    if (!IsOkay())
    {
        E_ERROR << "error opening camera " << index;
        return;
    }

	if (!GetImageInfo(m_imageSize, m_imageType))
	{
		E_ERROR << "error retrieving image info of camera " << index;
		return;
	}

    if (!StCam_StartTransfer(m_handle))
    {
        StCam_Close(m_handle);
        m_handle = NULL;

        E_ERROR << "error starting transfer with camera " << index;
        return;
    }

    E_INFO << "Sentech camera " << index << " (" << name << ") opened";
}

StImageGrabber::~StImageGrabber()
{
    if (!IsOkay()) return;

    if (!StCam_StopTransfer(m_handle))
    {
        E_ERROR << "error stopping transfer with camera #" << GetIndex();
    }

    StCam_Close(m_handle);
	m_handle = NULL;
}

bool StImageGrabber::GetImageInfo(cv::Size& imageSize, int& type) const
{
    BOOL   bResult = TRUE;
    DWORD  dwReserved;
    DWORD  dwOffsetX, dwOffsetY;
    DWORD  dwWidth, dwHeight;
    WORD   wScanMode;
    DWORD  dwPixelFormat;

    //Get Image Size
    bResult = StCam_GetImageSize(
        m_handle, &dwReserved, &wScanMode,
        &dwOffsetX, &dwOffsetY, &dwWidth, &dwHeight);

    if (!bResult)
    {
        E_ERROR << "error retrieving image size";
        return false;
    }

    //Get Preview Pixel Format
    bResult = StCam_GetPreviewPixelFormat(m_handle, &dwPixelFormat);

    if (!bResult)
    {
		E_ERROR << "error retrieving pixel format";
        return false;
    }

    imageSize.height = dwHeight;
    imageSize.width  = dwWidth;

    switch (dwPixelFormat)
    {
    case STCAM_PIXEL_FORMAT_24_BGR: type = CV_8UC3;	 break;
    case STCAM_PIXEL_FORMAT_32_BGR: type = CV_8UC4;	 break;
    default:
		E_ERROR << "unsupported pixel format " << dwPixelFormat;
        return false;
    }

    return true;
}

bool StImageGrabber::Grab(cv::Mat& im)
{
    //take a snapshot
    DWORD dwNumberOfByteTrans, dwFrameNo;
    DWORD dwMilliseconds = 1000;
    BOOL  bResult;

	cv::Mat buff(m_imageSize, m_imageType);

    bResult = StCam_TakePreviewSnapShot(
        m_handle, buff.data, buff.total() * buff.elemSize(),
        &dwNumberOfByteTrans, &dwFrameNo, dwMilliseconds);

    if (!bResult)
    {
        return false;
    }

	buff.copyTo(im);

	return true;
}
#endif
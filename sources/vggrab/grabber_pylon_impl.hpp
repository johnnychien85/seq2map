#ifndef GRABBER_PYLON_IMPL_HPP
#define GRABBER_PYLON_IMPL_HPP

#include <pylon/PylonIncludes.h>
#include "grabber.hpp"

class PylonImageGrabberBuilder : public ImageGrabberBuilder
{
public:
    virtual ~PylonImageGrabberBuilder() {}
    virtual ImageGrabber::Ptr Build(const SyncBuffer::Ptr& buffer, size_t index);
    virtual size_t GetDevices() const { return m_deviceInfoList.size(); };
    virtual Strings GetDeviceIdList() const;
private:
    friend class Factory<String, ImageGrabberBuilder>;
    PylonImageGrabberBuilder() { EnumerateDevices(); }
    void EnumerateDevices();

    Pylon::PylonAutoInitTerm m_autoInitTerm;
    Pylon::DeviceInfoList m_deviceInfoList;
};

class PylonImageGrabber : public ImageGrabber
{
public:
    virtual ~PylonImageGrabber() { if (m_camera.IsOpen()) m_camera.Close(); }
    virtual String ToString() const { return m_displayName; }

protected:
    friend class PylonImageGrabberBuilder;
    PylonImageGrabber(Pylon::IPylonDevice* device, const String& displayName, const SyncBuffer::Ptr& buffer);

    virtual bool GetImageInfo(cv::Size& imageSize, int& type) const;
    virtual bool Grab(cv::Mat& im);
    virtual bool IsOkay() const { return m_camera.IsOpen(); }

private:
    static int Pylon2CvType(const Pylon::EPixelType& pxfmt);

    Pylon::CInstantCamera m_camera;
    const String m_displayName;
    cv::Size m_imageSize;
    int m_imageType;
};

void PylonImageGrabberBuilder::EnumerateDevices()
{
    Pylon::CTlFactory::GetInstance().EnumerateDevices(m_deviceInfoList);
}

inline Strings PylonImageGrabberBuilder::GetDeviceIdList() const
{
    Strings ids;

    for (size_t i = 0; i < m_deviceInfoList.size(); i++)
    {
        ids.push_back(m_deviceInfoList[i].GetSerialNumber().c_str());
    }

    return ids;
}

ImageGrabber::Ptr PylonImageGrabberBuilder::Build(const SyncBuffer::Ptr& buffer, size_t index)
{
    if (index >= m_deviceInfoList.size())
    {
        E_ERROR << "index out of bound";
        return ImageGrabber::Ptr();
    }

    const Pylon::CDeviceInfo& info = m_deviceInfoList[index];

    Pylon::IPylonDevice* device =
        Pylon::CTlFactory::GetInstance().CreateDevice(info);

    if (device == NULL)
    {
        E_ERROR << "error creating device " << index;
        return ImageGrabber::Ptr();
    }

    // make device name, e.g. Basler AC-1300 (xxxxxxxx)
    std::stringstream ss;
    ss << info.GetVendorName().c_str() << " " << info.GetModelName().c_str();
    ss << " (" << info.GetSerialNumber().c_str() << ")";

    ImageGrabber::Ptr grabber(static_cast<ImageGrabber*>(new PylonImageGrabber(device, ss.str(), buffer)));
    grabber->SetIndex(index);

    return grabber;
}

PylonImageGrabber::PylonImageGrabber(Pylon::IPylonDevice* device, const String& displayName, const SyncBuffer::Ptr& buffer)
: m_camera(device), m_displayName(displayName), m_imageType(-1), ImageGrabber(buffer)
{
    assert(device != NULL && buffer);

    try
    {
        m_camera.Open();
    }
    catch (Pylon::GenericException& ex)
    {
        E_ERROR << "error opening device " << GetIndex();
        E_ERROR << ex.GetDescription();

        return;
    }

    // try to get image size and pixel format
    GenApi::INodeMap& nodemap = m_camera.GetNodeMap();
    GenApi::CIntegerPtr width (nodemap.GetNode("Width" ));
    GenApi::CIntegerPtr height(nodemap.GetNode("Height"));

    if (width == NULL || height == NULL)
    {
        m_camera.Close();

        E_ERROR << "error retrieving image dimension for " << ToString();
        return;
    }

    m_imageSize.width  = static_cast<int>(width ->GetValue());
    m_imageSize.height = static_cast<int>(height->GetValue());

    GenApi::CEnumerationPtr pxfmtNode(nodemap.GetNode("PixelFormat"));
    Pylon::CPixelTypeMapper pxfmt(pxfmtNode);

    if (pxfmtNode == NULL || !pxfmt.IsValid())
    {
        m_camera.Close();
        E_ERROR << "error retrieving pixel format for device " << ToString();

        return;
    }

    Pylon::EPixelType pxtype = pxfmt.GetPylonPixelTypeFromNodeValue(pxfmtNode->GetIntValue());
    m_imageType = Pylon2CvType(pxtype);

    try
    {
        m_camera.StartGrabbing();
    }
    catch (Pylon::GenericException& ex)
    {
        m_camera.Close();

        E_ERROR << "error starting grabbing for device " << ToString();
        E_ERROR << ex.GetDescription();

        return;
    }

    E_INFO << "device opened " << ToString();
    E_INFO << "frame size "    << size2string(m_imageSize);
    E_INFO << "pixel format "  << pxfmt.GetNameByPixelType(pxtype);
}

bool PylonImageGrabber::GetImageInfo(cv::Size& imageSize, int& type) const
{
    if (m_imageSize.area() == 0 || m_imageType < 0) return false;

    imageSize = m_imageSize;
    type = m_imageType;

    return true;
}

bool PylonImageGrabber::Grab(cv::Mat& im)
{
    Pylon::CGrabResultPtr result;

    if (!m_camera.RetrieveResult(0, result, Pylon::TimeoutHandling_Return) ||
        !result->GrabSucceeded())
    {
        return false;
    }

    const uchar *buf = (uchar*) result->GetBuffer();
    cv::Mat(m_imageSize, m_imageType, (void*)buf).copyTo(im);

    return true;
}

int PylonImageGrabber::Pylon2CvType(const Pylon::EPixelType& pxfmt)
{
    using namespace Pylon;

    switch (pxfmt)
    {
    case PixelType_Mono8:
    case PixelType_BayerGR8:
    case PixelType_BayerRG8:
    case PixelType_BayerGB8:
    case PixelType_BayerBG8:
        return CV_8U;
    case PixelType_Mono8signed:
        return CV_8S;
    case PixelType_BGR8packed:
    case PixelType_RGB8packed:
        return CV_8UC3;
    case PixelType_Mono16:
    case PixelType_BayerBG12:
    case PixelType_BayerBG16:
        return CV_16U;
    case PixelType_BGR10packed:
    case PixelType_RGB10packed:
    case PixelType_RGB12packed:
    case PixelType_BGR12packed:
    case PixelType_RGB16packed:
        return CV_16UC3;
    case PixelType_Double:
        return CV_64F;
    default:
        return CV_USRTYPE1;
    }
}

#endif // GRABBER_PYLON_IMPL_HPP

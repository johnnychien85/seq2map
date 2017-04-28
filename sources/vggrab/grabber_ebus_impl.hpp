#ifndef GRABBER_EBUS_IMPL_HPP
#define GRABBER_EBUS_IMPL_HPP

#include <PvSystem.h>
#include <PvInterface.h>
#include <PvDevice.h>
#include <PvDeviceGEV.h>
#include <PvDeviceU3V.h>
#include <PvStream.h>
#include <PvPipeline.h>
#include <PvStreamGEV.h>
#include <PvStreamU3V.h>

#include "grabber.hpp"

class EbusImageGrabberBuilder : public ImageGrabberBuilder
{
public:
    virtual ~EbusImageGrabberBuilder();
    virtual ImageGrabber::Ptr Build(const SyncBuffer::Ptr& buffer, size_t index);
    virtual size_t GetDevices() const { return m_devicesInfo.size(); }
    virtual Strings GetDeviceIdList() const;
private:
    friend class Factory<String, ImageGrabberBuilder>;

    static bool CompareDeviceInfo(const PvDeviceInfo* d0, const PvDeviceInfo* d1);

    EbusImageGrabberBuilder();
    bool EnumerateDevices();

    PvSystem m_system;
    std::vector<const PvDeviceInfo*> m_devicesInfo;
};

class EbusImageGrabber : public ImageGrabber
{
public:
    EbusImageGrabber(PvDevice* device, const PvString& conn, const SyncBuffer::Ptr& buffer);
    virtual ~EbusImageGrabber();
    virtual std::string ToString() const;

protected:
    virtual bool GetImageInfo(cv::Size& imageSize, int& type) const;
    virtual bool Grab(cv::Mat& im);
    virtual bool IsOkay() const;

private:
    template<typename T> static T* GetGenParam(PvGenParameterArray* params, const PvString& name);
    static int Pv2CvType(int64_t pxfmt);

    cv::Size m_imageSize;
    int m_imageType;

    bool m_acquiring;

    PvDevice* m_device;
    PvStream* m_stream;
    PvPipeline* m_pipeline;
    PvGenCommand* m_acqStartCmd;
    PvGenCommand* m_acqStopCmd;
};

// TODO: initialise SDK
EbusImageGrabberBuilder::EbusImageGrabberBuilder()
{
    if (!EnumerateDevices())
    {
        E_ERROR << "error enumerating devices";
        return;
    }
}

// TODO: uninitialise SDK
EbusImageGrabberBuilder::~EbusImageGrabberBuilder()
{
}

bool EbusImageGrabberBuilder::EnumerateDevices()
{
    PvResult result;

    m_devicesInfo.clear();
    m_system.SetDetectionTimeout(200);
    result = m_system.Find();

    if (!result.IsOK())
    {
        E_ERROR << "error finding devices " << result.GetCodeString().GetAscii();
        return false;
    }

    uint32_t interfaces = m_system.GetInterfaceCount();

    for (uint32_t i = 0; i < interfaces; i++)
    {
        const PvInterface* interface = m_system.GetInterface(i);
        uint32_t deviceCount = interface->GetDeviceCount();

        for (uint32_t j = 0; j < deviceCount; j++)
        {
            m_devicesInfo.push_back(interface->GetDeviceInfo(j));
        }
    }

    std::sort(m_devicesInfo.begin(), m_devicesInfo.end(), CompareDeviceInfo);

    return true;
}

bool EbusImageGrabberBuilder::CompareDeviceInfo(const PvDeviceInfo* d0, const PvDeviceInfo* d1)
{
    String s0 = d0->GetUniqueID().GetAscii();
    String s1 = d1->GetUniqueID().GetAscii();

    size_t n = std::min(s0.length(), s1.length());

    for (size_t i = 0; i < n; i++)
    {
        if (s0[i] != s1[i]) return s0[i] < s1[i];
    }

    return true;
}


// return a list of connected device IDs
Strings EbusImageGrabberBuilder::GetDeviceIdList() const
{
    Strings ids;
    for (size_t i = 0; i < m_devicesInfo.size(); i++)
    {
        ids.push_back(m_devicesInfo[i]->GetDisplayID().GetAscii());
    }
    return ids;
}

// TODO: build a eBUS grabber object of the given device index
ImageGrabber::Ptr EbusImageGrabberBuilder::Build(const SyncBuffer::Ptr & buffer, size_t index)
{
    if (index >= m_devicesInfo.size())
    {
        E_ERROR << "index out of bound";
        return ImageGrabber::Ptr();
    }

    PvResult result;
    PvDevice* device = PvDevice::CreateAndConnect(m_devicesInfo[index], &result);

    if (!result.IsOK())
    {
        E_ERROR << "error creating/connecting to device " << m_devicesInfo[index]->GetDisplayID().GetAscii();
        return ImageGrabber::Ptr();
    }

    ImageGrabber::Ptr grabber = ImageGrabber::Ptr(static_cast<ImageGrabber*>(
        new EbusImageGrabber(device, m_devicesInfo[index]->GetConnectionID(), buffer)
    ));

    if (grabber != NULL)
    {
        grabber->SetIndex(index);
    }

    return grabber;
}

// initialise the eBUS device here and start transmission
EbusImageGrabber::EbusImageGrabber(PvDevice* device, const PvString& conn, const SyncBuffer::Ptr& buffer)
: m_device(device), m_stream(NULL), m_acqStartCmd(NULL), m_acqStopCmd(NULL), m_imageType(-1), m_acquiring(false), ImageGrabber(buffer)
{
    PvGenParameterArray* params = m_device->GetParameters();
    PvGenInteger* pWidth  = GetGenParam<PvGenInteger>(params, "Width" );
    PvGenInteger* pHeight = GetGenParam<PvGenInteger>(params, "Height");

    PvGenEnum* pTriggerMode = GetGenParam<PvGenEnum>(params, "TriggerMode");
    pTriggerMode->SetValue("On");

    PvGenInteger* pOffsetY = GetGenParam<PvGenInteger>(params, "OffsetY");

    pHeight->SetValue(512); // 720
    pOffsetY->SetValue(272); // 64

    if (pWidth == NULL || pHeight == NULL)
    {
        E_ERROR << "error retrieving IGen parameters of image dimension";
        return;
    }
    else
    {
        int64_t width, height;

        if (!(pWidth ->GetValue(width)).IsOK() ||
            !(pHeight->GetValue(height)))
        {
            E_ERROR << "error retrieving image dimension from IGen parameters";
            return;
        }

        m_imageSize.height = static_cast<int>(height);
        m_imageSize.width  = static_cast<int>(width );
    }

    PvGenEnum* pPxfmt = GetGenParam<PvGenEnum>(params, "PixelFormat");

    if (pPxfmt == NULL)
    {
        E_ERROR << "error retrieving IGen parameter of pixel format";
    }
    else
    {
        int64_t pxfmt;

        if (!pPxfmt->GetValue(pxfmt))
        {
            E_ERROR << "error retrieving pixel format";
            return;
        }

        m_imageType = Pv2CvType(pxfmt);
    }

    PvResult result;
    m_stream = PvStream::CreateAndOpen(conn, &result);

    if (!result.IsOK())
    {
        E_ERROR << "error creating stream";
        return;
    }

    // configure stream for ethernet
    PvDeviceGEV* gev = dynamic_cast<PvDeviceGEV*>(m_device);
    if (gev != NULL)
    {
        PvStreamGEV *gevStream = static_cast<PvStreamGEV*>(m_stream);
        gev->NegotiatePacketSize(); // negotiate packet size
        gev->SetStreamDestination(gevStream->GetLocalIPAddress(), gevStream->GetLocalPort()); // configure device streaming destination
    }

    m_pipeline = new PvPipeline(m_stream);

    if (m_pipeline == NULL)
    {
        E_ERROR "error creating pipeline";
        return;
    }
    else
    {
        m_pipeline->SetBufferCount(16); // more buffers will reduced frame drop but increase latency as well
        m_pipeline->SetBufferSize(m_device->GetPayloadSize());
    }

    if (!m_pipeline->Start().IsOK())
    {
        E_ERROR << "error starting pipeline";
        return;
    }

    if (!m_device->StreamEnable().IsOK())
    {
        E_ERROR << "error enabling streaming";
        return;
    }

    m_acqStartCmd = GetGenParam<PvGenCommand>(params, "AcquisitionStart");
    m_acqStopCmd  = GetGenParam<PvGenCommand>(params, "AcquisitionStop" );

    if (m_acqStartCmd == NULL || m_acqStopCmd == NULL)
    {
        E_ERROR << "error retrieving acquisition control commands from IGen parameters";
        return;
    }

    // send start acquisition command
    if (!m_acqStartCmd->Execute().IsOK())
    {
        E_ERROR << "error executing start acquisition command";
        return;
    }

    m_acquiring = true;
}

// uninitialise the eBUS device
EbusImageGrabber::~EbusImageGrabber()
{
    if (m_acquiring)
    {
        assert(m_acqStopCmd != NULL);

        E_INFO << "sending stop acquisition command";

        m_acqStopCmd->Execute();
        m_acquiring = false;
    }

    if (m_pipeline != NULL)
    {
        E_INFO << "stopping pipeline";

        m_pipeline->Stop();
        delete m_pipeline;
    }

    if (m_stream != NULL)
    {
        E_INFO << "closing stream";

        m_stream->Close();
        PvStream::Free(m_stream);
    }

    if (m_device != NULL)
    {
        E_INFO << "disconnecting to the device";

        m_device->Disconnect();
        PvDevice::Free(m_device);
    }
}

// return the human readable ID of the eBUS device
String EbusImageGrabber::ToString() const
{
    return "unknown";
}

// return the dimension and CV type of the images streamed by the eBUS device
bool EbusImageGrabber::GetImageInfo(cv::Size& imageSize, int& type) const
{
    if (m_imageSize.area() == 0 || m_imageType < 0) return false;

    imageSize = m_imageSize;
    type = m_imageType;

    return true;
}

// try to grab a frame from the eBUS device and convert to the given cv::Mat object
bool EbusImageGrabber::Grab(cv::Mat& im)
{
    PvBuffer* buffer = NULL;
    PvResult result;

    if (!m_pipeline->RetrieveNextBuffer(&buffer, 500, &result).IsOK())
    {
        E_ERROR << "error retrieving next buffer";
        return false;
    }

    if (!result.IsOK())
    {
        // the frame is not ready for now
        return false;
    }

    PvImage* image = buffer->GetImage();
    assert(image->GetWidth() == m_imageSize.width && image->GetHeight() == m_imageSize.height);

    cv::Mat(m_imageSize, m_imageType, image->GetDataPointer()).copyTo(im);

    m_pipeline->ReleaseBuffer(buffer);

    return true;
}

bool EbusImageGrabber::IsOkay() const
{
    return m_acquiring;
}

template<typename T>
T* EbusImageGrabber::GetGenParam(PvGenParameterArray* params, const PvString& name)
{
    return dynamic_cast<T*>(params->Get(name));
}

int EbusImageGrabber::Pv2CvType(int64_t pxfmt)
{
    #include <PvPixelType.h>

    switch (pxfmt)
    {
    case PvPixelMono8:
        return CV_8U;
    default:
        return CV_USRTYPE1;
    }

}

#endif // GRABBER_EBUS_IMPL_HPP

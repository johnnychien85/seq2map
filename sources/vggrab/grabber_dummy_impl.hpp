#ifndef GRABBER_DUMMY_IMPL_HPP
#define GRABBER_DUMMY_IMPL_HPP

#include "grabber.hpp"

class DummyImageGrabberBuilder : public ImageGrabberBuilder
{
public:
    virtual ~DummyImageGrabberBuilder() {}
    virtual ImageGrabber::Ptr Build(const SyncBuffer::Ptr& buffer, size_t index);
    virtual size_t GetDevices() const { return m_numGrabbers; }
    virtual Strings GetDeviceIdList() const;
private:
    friend class Factory<String, ImageGrabberBuilder>;
    DummyImageGrabberBuilder();
    static size_t m_numGrabbers;
};

class DummyImageGrabber : public ImageGrabber
{
public:
    DummyImageGrabber(const SyncBuffer::Ptr& buffer) : ImageGrabber(buffer) { m_fpsLimiter.Start();  }
    virtual ~DummyImageGrabber() {}
    virtual std::string ToString() const;

protected:
    virtual bool GetImageInfo(cv::Size& imageSize, int& type) const;
    virtual bool Grab(cv::Mat& im);
    virtual bool IsOkay() const { return true; }

private:
    static cv::Size s_imageSize;
    static cv::Size s_noiseSize;
    static double s_interval;

    Speedometre m_fpsLimiter;
};

size_t DummyImageGrabberBuilder::m_numGrabbers = 5;
cv::Size DummyImageGrabber::s_imageSize = cv::Size(1024, 768);
cv::Size DummyImageGrabber::s_noiseSize = cv::Size( 512, 384);
double   DummyImageGrabber::s_interval  = 0.05f;

DummyImageGrabberBuilder::DummyImageGrabberBuilder()
{
    E_TRACE << "dummy image grabber builder instantiated";
}

ImageGrabber::Ptr DummyImageGrabberBuilder::Build(const SyncBuffer::Ptr& buffer, size_t index)
{
    if (index >= GetDevices()) return ImageGrabber::Ptr();

    ImageGrabber::Ptr grabber = ImageGrabber::Ptr(new DummyImageGrabber(buffer));
    grabber->SetIndex(index);

    return grabber;
}

Strings DummyImageGrabberBuilder::GetDeviceIdList() const
{
    Strings ids(GetDevices());

    for (size_t i = 0; i < ids.size(); i++)
    {
        std::stringstream ss;
        ss << "Dummy Camera for Testing, idx=" << i;
        ids[i] = ss.str();
    }
    
    return ids;
}

String DummyImageGrabber::ToString() const
{
    std::stringstream ss;
    ss << "dummy camera " << GetIndex();

    return ss.str();
}

bool DummyImageGrabber::Grab(cv::Mat& im)
{
    if (m_fpsLimiter.GetElapsedSeconds() < s_interval) return false;
    
    cv::Mat noise = cv::Mat(s_noiseSize, CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::resize(noise, im, im.size());

    m_fpsLimiter.Reset();
    m_fpsLimiter.Start();

    return true;
}

bool DummyImageGrabber::GetImageInfo(cv::Size& imageSize, int& type) const
{
    imageSize = s_imageSize;
    type = CV_8UC3;

    return true;
}
#endif // GRABBER_DUMMY_IMPL_HPP

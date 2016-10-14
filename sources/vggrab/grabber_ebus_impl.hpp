#ifndef GRABBER_EBUS_IMPL_HPP
#define GRABBER_EBUS_IMPL_HPP

#include "grabber.hpp"

class EbusImageGrabberBuilder : public ImageGrabberBuilder
{
public:
    virtual ~EbusImageGrabberBuilder();
    virtual ImageGrabber::Ptr Build(const SyncBuffer::Ptr& buffer, size_t index);
    virtual size_t GetDevices() const;
    virtual Strings GetDeviceIdList() const;
private:
    friend class Factory<String, ImageGrabberBuilder>;
    EbusImageGrabberBuilder();
};

class EbusImageGrabber : public ImageGrabber
{
public:
    EbusImageGrabber(const SyncBuffer::Ptr& buffer) : ImageGrabber(buffer);
    virtual ~EbusImageGrabber() {}
    virtual std::string ToString() const;

protected:
    virtual bool GetImageInfo(cv::Size& imageSize, int& type) const;
    virtual bool Grab(cv::Mat& im);
    virtual bool IsOkay() const;
};

#endif // GRABBER_EBUS_IMPL_HPP

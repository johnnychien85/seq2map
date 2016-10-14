#ifndef GRABBER_HPP
#define GRABBER_HPP

#include "syncbuf.hpp"

class ImageGrabber : public BufferWriter, public Indexed
{
public:
    typedef boost::shared_ptr<ImageGrabber> Ptr;
    typedef std::vector<Ptr> Ptrs;

    ImageGrabber(const SyncBuffer::Ptr& buffer) : BufferWriter(buffer) {}
    virtual ~ImageGrabber() {}
    virtual std::string ToString() const = 0;
    virtual cv::Mat GetImage();
    inline cv::Size GetImageSize() const { return m_image.size(); }

protected:
    virtual bool GetImageInfo(cv::Size& imageSize, int& type) const = 0;
    inline virtual TimedImage& Pick(BufferData& data) const { return data.images[GetIndex()]; }
    virtual bool Grab(cv::Mat& im) = 0;
    virtual bool IsOkay() const = 0;
    virtual bool GrabNextData();
    virtual size_t Write(BufferData& data);

private:
    boost::mutex m_mtx;
    cv::Mat      m_image;
};

class ImageGrabberBuilder
{
public:
    typedef boost::shared_ptr<ImageGrabberBuilder> Ptr;
    virtual ImageGrabber::Ptr Build(const SyncBuffer::Ptr& buffer, size_t index) = 0;
    virtual size_t GetDevices() const = 0;
    virtual Strings GetDeviceIdList() const = 0;

protected:
    ImageGrabberBuilder() {}
    virtual ~ImageGrabberBuilder() {}
};

class ImageGrabberBuilderFactory : public Factory<String, ImageGrabberBuilder>
{
public:
    ImageGrabberBuilderFactory();
};
#endif // GRABBER_HPP

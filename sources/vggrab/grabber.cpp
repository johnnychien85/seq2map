#include <sstream>
#include <boost/assign/list_of.hpp>
#include "grabber.hpp"
#include "grabber_dummy_impl.hpp"

#ifdef WITH_STCAM
#include "grabber_stcam_impl.hpp"
#endif // WITH_STCAM
#ifdef WITH_EBUS
#include "grabber_ebus_impl.hpp"
#endif // WITH_EBUS

ImageGrabberBuilderFactory::ImageGrabberBuilderFactory()
{
    Factory::Register<DummyImageGrabberBuilder>("DUMMY");
#ifdef WITH_EBUS
    Factory::Register<EBusImageGrabberBuilder> ("EBUS");
#endif // WITH_EBUS
}

bool ImageGrabber::GrabNextData()
{
    boost::lock_guard<boost::mutex> locker(m_mtx);

    if (m_image.empty())
    {
        cv::Size imageSize;
        int imageType;

        if (!GetImageInfo(imageSize, imageType))
        {
            E_ERROR << "error retrieving image size";
            return false;
        }

        m_image = cv::Mat::zeros(imageSize, imageType);
    }

    return Grab(m_image);
}

size_t ImageGrabber::Write(BufferData& data)
{
    TimedImage& image = Pick(data);
    cv::Mat& im = image.payload;

    if (im.empty())
    {
        im = m_image.clone();
    }
    else
    {
        assert(im.total() == m_image.total());
        memcpy(im.data, m_image.data, im.total() * im.elemSize());
    }

    image.Touch();
    return im.total() * im.elemSize();
}

cv::Mat ImageGrabber::GetImage()
{
    boost::lock_guard<boost::mutex> locker(m_mtx);
    return m_image.clone();
}

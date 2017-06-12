#include <boost/date_time/posix_time/posix_time_io.hpp>
#include "recorder.hpp"

String BufferRecorder::GetDateTimeString(const String& format, const Time& when)
{
    using namespace boost::posix_time; // for time_facet

    std::stringstream ss;
    time_facet* facet = new time_facet(format.c_str());

    ss.imbue(std::locale(ss.getloc(), facet));
    ss << when;

    return ss.str();
}

void BufferRecorder::StartRecording()
{
    if (m_recording) return;

    m_recStorage.Create(m_seqDirPath / GetDateCode() / GetTimeCode(), m_buffer->GetWritters());
    m_recording = true;

    m_dropped = m_written = 0;
}

size_t BufferRecorder::Read(BufferData& data)
{
    if (!data.IsSynced())
    {
        m_dropped++;
        return 0;
    }

    size_t bytesWritten = 0;
    size_t imageDataBytes = 0;

    std::vector<cv::Mat> images;

    BOOST_FOREACH (const TimedImage& image, data.images)
    {
        cv::Mat im = image.payload;

        images.push_back(im.clone());
        imageDataBytes += im.elemSize() * im.total();
    }

    std::vector<VOStorage*> storages;

    if (m_recording)    storages.push_back(&m_recStorage);
    if (m_snapshooting) storages.push_back(&m_snapStorage);

    m_snapshooting = false;

    BOOST_FOREACH (VOStorage* storage, storages)
    {
        size_t seq = storage->GetNumOfFrames();
        std::stringstream ss; ss << std::setw(8) << std::setfill('0') << seq;
        storage->AddFrame(ss.str(), images);

        m_written++;
        bytesWritten += imageDataBytes;
    }

    return bytesWritten;
}

#ifndef RECORDER_HPP
#define RECORDER_HPP
#include "syncbuf.hpp"
#include "storage.hpp"

class BufferRecorder : public BufferReader
{
public:
    BufferRecorder(const SyncBuffer::Ptr& buffer, const Path& root)
    : BufferReader(buffer), m_seqDirPath(root), m_recording(false), m_snapshooting(false),
      m_snapStorage(root / GetDateCode() / "snapshot", buffer->GetWritters()), m_dropped(0) {};
    inline bool IsRecording() const {return m_recording;};
    inline size_t GetDroppedFrames() const {return m_dropped;}
    void StartRecording();
    void StopRecording() {m_recording = false;}
    void Snapshot() { m_snapshooting = true; }
    virtual std::string ToString() const {return "Recorder";}
    size_t GetDropped() const {return m_dropped;}
    size_t GetWritten() const {return m_written;}

protected:
    virtual bool IsOkay() const {return true;};
    virtual size_t Read(BufferData& data);

    static String GetDateTimeString(const std::string& format, const Time& when = unow());
    static String GetDateCode() {return GetDateTimeString("%Y%m%d");}
    static String GetTimeCode() {return GetDateTimeString("%H%M%S");}

    Path m_seqDirPath;
    VOStorage m_recStorage;
    VOStorage m_snapStorage;
    bool m_recording;
    bool m_snapshooting;
    size_t m_dropped;
    size_t m_written;
};
#endif

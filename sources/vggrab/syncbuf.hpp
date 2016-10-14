#ifndef SYNCBUF_HPP
#define SYNCBUF_HPP

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <seq2map/common.hpp>

using namespace seq2map;

typedef boost::posix_time::ptime Time;

inline Time UNow()
{
    return boost::posix_time::microsec_clock::local_time();
}

template<class T> class TimedData
{
public:
    inline void Touch() { time = UNow(); }

    Time time;
    T payload;
};

typedef TimedData<cv::Mat> TimedImage;

class BufferData
{
public:
    TimedImage images[2];

    inline void Create(size_t seq, const Time& timestamp);
    inline void Lock()             { m_mtx.lock();            }
    inline void Unlock()           { m_mtx.unlock();          }
    inline bool IsEmpty()    const { return m_numWrites == 0; }
    inline bool IsSynced()   const { return m_synced;         }
    inline void Dispose()          { m_clear = true;          }
    inline long GetDelta(const Time& time) const { return (time - m_time).total_milliseconds(); }

protected:
    friend class SyncBuffer;
    BufferData() : m_seq(INVALID_INDEX), m_numReads(0), m_numWrites(0), m_clear(true), m_synced(true) {}
    virtual ~BufferData() {}

    size_t m_seq;
    size_t m_numReads;
    size_t m_numWrites;
    bool   m_clear;
    bool   m_synced;
    Time   m_time;
    boost::mutex m_mtx;
};

class AutoBufferLocker
{
public:
    AutoBufferLocker(BufferData& data) : m_data(data) { m_data.Lock();   }
    virtual ~AutoBufferLocker()                       { m_data.Unlock(); }
private:
    BufferData& m_data;
};

class SyncBuffer
{
public:
    typedef boost::shared_ptr<SyncBuffer> Ptr;

    SyncBuffer(size_t numWriters, size_t numReaders, size_t bufferSize, double fps, double epsilon)
        : m_numWriters(numWriters), m_numReaders(numReaders), m_bufferSize(bufferSize),
        m_interval(1000.0f / fps), m_halfInterval(m_interval * 0.5f), m_sigma(m_halfInterval * epsilon),
        m_data(new BufferData[bufferSize]), m_timestamp(UNow()), m_seqHead(0), m_seqTail(0)
    { assert(fps > 0 && epsilon > 0.0f && epsilon < 1.0f); }
    virtual ~SyncBuffer()             { delete[] m_data;   }
    unsigned long GetInterval() const { return m_interval; }
    double GetUsage();

protected:
    friend class BufferWriter;
    friend class BufferReader;
    bool TryWrite(size_t& seq, unsigned long& bestDelta, BufferData*& data);
    bool TryRead(size_t seq, BufferData*& data);
    void CommitRead(size_t& seq);

private:
    inline BufferData& operator[] (size_t seq) { return m_data[seq % m_bufferSize]; }

    BufferData*  m_data;
    const size_t m_numReaders;
    const size_t m_numWriters;
    const size_t m_bufferSize;
    Time         m_timestamp;
    size_t       m_seqHead;
    size_t       m_seqTail;
    boost::mutex m_mtx;
    const unsigned long m_interval;     // the width of slots
    const unsigned long m_halfInterval; // half width of slots
    const unsigned long m_sigma;        // acceptable time difference
};

class BufferThread
{
public:
    BufferThread(const SyncBuffer::Ptr& buffer) : m_buffer(buffer), m_seq(0) {};
    inline void SetBuffer(const SyncBuffer::Ptr& buffer) { m_buffer = buffer; }
    bool Start();
    void Stop();
    boost::thread::id GetId() const { return m_thread.get_id(); }
    inline bool IsRunning()   const { return m_thread.get_id() != boost::thread::id(); }
    inline size_t GetSeq()    const { return m_seq; }
    inline const Speedometre& GetMetre() const { return m_speedometre; }
    virtual std::string ToString() const = 0;

protected:
    virtual bool IsOkay() const = 0;
    virtual void Loop() = 0;

    SyncBuffer::Ptr m_buffer;
    size_t          m_seq;
    Speedometre     m_speedometre;

private:
    static void Entry(BufferThread* thread);
    boost::thread   m_thread;
};

class BufferWriter : public BufferThread
{
public:
    typedef boost::shared_ptr<BufferWriter> Ptr;

    BufferWriter(const SyncBuffer::Ptr& buffer)
        : BufferThread(buffer), m_bestDelta(m_buffer->GetInterval()/2) {}
    inline unsigned long GetDelta() const { return m_bestDelta; }
    virtual std::string ToString() const = 0;

protected:
    virtual void Loop();
    virtual bool GrabNextData() = 0;
    virtual size_t Write(BufferData& data) = 0;

    unsigned long m_bestDelta;
};

class BufferReader : public BufferThread
{
public:
    virtual std::string ToString() const = 0;

protected:
    BufferReader(const SyncBuffer::Ptr& buffer) : BufferThread(buffer) {};
    virtual void Loop();
    virtual size_t Read(BufferData& data) = 0;
};
#endif

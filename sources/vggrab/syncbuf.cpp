#include "syncbuf.hpp"

using namespace std;
using namespace cv;

void BufferData::Create(size_t seq, const Time& timestamp)
{
    m_seq    = seq;
    m_time   = timestamp;
    m_synced = true;
    m_clear  = false;
    m_numReads  = 0;
    m_numWrites = 0;

    //E_TRACE << "frame " << seq << " created, ts=" << timestamp;
}

double SyncBuffer::GetUsage()
{
    boost::lock_guard<boost::mutex> locker(m_mtx);
    return m_seqHead > m_seqTail ? (double)(m_seqHead - m_seqTail) / (double)m_bufferSize : 0.0f;
}

bool SyncBuffer::TryWrite(size_t& seq, unsigned long& bestDelta, BufferData*& data)
{
    Time now = UNow();

    for(;;)
    {
        // pointer to the #seq
        data = &(*this)[seq];
        AutoBufferLocker locker(*data);

        // check if this is the first write on this data slot
        if (seq != data->m_seq)
        {
            if (!data->m_clear)
            {
                // the first write is not possible
                //E_WARNING << "buffer overflow!!";
                //E_TRACE   << "creation of frame " << seq << " failed, old frame " << data->m_seq << " is not clear";

                return false;
            }

            m_timestamp += boost::posix_time::milliseconds(m_interval);
            data->Create(seq, m_timestamp);
        }

        assert(data->m_numReads == 0);
        assert(data->m_numWrites < m_numWriters);

        long delta = abs(data->GetDelta(now)); // delta = |t - t0|
        bool inSlot = delta <= m_halfInterval;
        bool synced = delta <= bestDelta;

        //E_TRACE << "seq=" << seq << " delta=" << delta << " ts=" << now;

        //        out       in
        //   ||  /         /      ||                   ||
        // --||-x---[   o x ]-----||-----[   o   ]-----||-----[
        //   ||seq=0     \        ||seq=1              ||seq=2
        //                 timestamp
        if (synced)
        {
            boost::lock_guard<boost::mutex> locker(m_mtx);
            m_seqHead = m_seqHead > seq ? m_seqHead : seq;

            bestDelta = delta;

            return true; // it's synced, write data immediately!!
        }

        // we are at the right slot but the delta is not acceptable
        // in this case we withdraw the writing attempt
        if (inSlot) return false;

        // out of sync
        data->m_synced &= bestDelta < m_sigma; // ever synced before advancing to next data frame?
        data->m_numWrites++;                   // writing committed
        bestDelta = m_sigma;
        seq++;

        // E_TRACE << "commited << " << (seq - 1) << " synced=" << data->_synced;
    }
}

bool SyncBuffer::TryRead(size_t seq, BufferData*& data)
{
    data = &(*this)[seq];
    AutoBufferLocker locker(*data);

    bool dataExists  = !data->m_clear;
    bool writingDone =  data->m_numWrites == m_numWriters;

    return dataExists & writingDone;
}

void SyncBuffer::CommitRead(size_t& seq)
{
    BufferData& data = (*this)[seq];
    AutoBufferLocker locker(data);

    assert(data.m_numReads < m_numReaders);

    data.m_numReads++;
    seq++;

    if (data.m_numReads == m_numReaders)
    {
        boost::lock_guard<boost::mutex> locker(m_mtx);
        m_seqTail = seq;

        data.m_clear = true; // completely consumed
    }
}

bool BufferThread::Start()
{
    if (!IsOkay()) return false;
    if (IsRunning())
    {
        E_WARNING << "thread already launched";
        return false;
    }

    m_speedometre.Reset();
    m_speedometre.Start();
    m_thread = boost::thread(Entry, this);

    return true;
}

void BufferThread::Stop()
{
    if (!IsOkay()) return;
    if (!IsRunning())
    {
        E_WARNING << "thread not launched";
        return;
    }

    m_thread.interrupt();
    m_thread.join();

    return;
}

void BufferThread::Entry(BufferThread* thread)
{
    E_INFO << "buffer thread lunched, id=" <<  thread->GetId();

    try
    {
        for (;;)
        {
            thread->Loop();
            boost::this_thread::interruption_point();
        }
    }
    catch (boost::thread_interrupted const&)
    {
        E_INFO << "interruption captured";
    }

    E_INFO << "buffer thread terminated";
    return;
}

void BufferWriter::Loop()
{
    BufferData* data;
    if (GrabNextData() && m_buffer->TryWrite(m_seq, m_bestDelta, data))
    {
        size_t bytesWritten = Write(*data);
        m_speedometre.Update(bytesWritten);
        //E_TRACE << "frame " << _seq << " written with " << bytesWritten << " bytes, delta=" << _bestDelta;
    }
}

void BufferReader::Loop()
{
    BufferData* data = NULL;
    if (m_buffer->TryRead(m_seq, data))
    {
        size_t bytesRead = Read(*data);
        m_buffer->CommitRead(m_seq);
        m_speedometre.Update(bytesRead);
    }
}

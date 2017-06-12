#ifndef SEQ_FILE_STORE_HPP
#define SEQ_FILE_STORE_HPP

#include <seq2map/common.hpp>

namespace seq2map
{
    /**
     * 
     */
    template <typename T>
    class SequentialFileStore
    : public Persistent<cv::FileStorage, cv::FileNode>,
      public Persistent<Path>
    {
    public:
        bool Create(const Path& root, size_t allocated = 128)
        {
            if (!makeOutDir(root))
            {
                E_ERROR << "error creating directory " << root;
                return false;
            }

            m_root = root;
            m_filenames.clear();
            m_filenames.reserve(allocated);

            return true;
        }

        bool Create(const Path& root, const Strings& filenames)
        {
            if (!Create(root, filenames.size()))
            {
                return false;
            }

            m_filenames = filenames;

            return true;
        }

        void FromExistingFiles(const Path& root, const String& ext = "")
        {
            Paths files = enumerateFiles(root, ext);
            Strings filenames;

            BOOST_FOREACH (const Path& file, files)
            {
                filenames.push_back(file.filename().string());
            }

            Create(root, filenames);
        }

        bool Append(const String& filename, const T& data)
        {
            // TODO: duplication check
            // ...
            // ..
            // .

            Path to = m_root / filename;

            if (!Append(to, data))
            {
                E_ERROR << "error storing to " << filename;
                return false;
            }

            m_filenames.push_back(filename);

            return true;
        }

        bool Retrieve(size_t idx, T& data) const
        {
            if (idx >= m_filenames.size())
            {
                E_ERROR << "index out of bound (index=" << idx << ", size=" << m_filenames.size() << ")";
                return false;
            }

            return Retrieve(m_root / m_filenames[idx], data);
        }

        T operator[] (size_t idx) const
        {
            T data;

            if (!Retrieve(idx, data))
            {
                E_ERROR << "error retrieving data, index=" << idx;
                return T();
            }

            return data;
        }

        inline size_t GetItems() const
        {
            return m_filenames.size();
        }

        inline Path GetRoot() const
        {
            return m_root;
        }

        inline Path GetItemPath(size_t idx) const
        {
            return m_root / m_filenames[idx];
        }

        inline const Strings& GetFileNames() const
        {
            return m_filenames;
        }

        //...
        virtual bool Store(cv::FileStorage& fs) const
        {
            try
            {
                fs << "root"  << m_root.string();
                fs << "items" << m_filenames.size();
                fs << "files" << "[";
                BOOST_FOREACH(const String& filename, m_filenames)
                {
                    fs << filename;
                }
                fs << "]";
            }
            catch (std::exception& ex)
            {
                E_ERROR << "error storing sequential file store";
                E_ERROR << ex.what();

                return false;
            }

            return true;
        }

        virtual bool Restore(const cv::FileNode& fn)
        {
            m_filenames.clear();

            try
            {
                cv::FileNode files = fn["files"];
                String root;
                size_t items = 0;

                fn["root"]  >> root;
                fn["items"] >> items;

                if (!Create(root, items))
                {
                    E_ERROR << "error creating store " << root;
                    return false;
                }

                for (cv::FileNodeIterator itr = files.begin(); itr != files.end(); itr++)
                {
                    m_filenames.push_back((String)*itr);
                }

                if (items != m_filenames.size())
                {
                    E_WARNING << "the number of items " << items << " does not agree with file list size " << m_filenames.size();
                    E_WARNING << "possible file corruption";
                }
            }
            catch (std::exception& ex)
            {
                E_ERROR << "error restoring sequential file store";
                E_ERROR << ex.what();

                return false;
            }

            return true;
        }

        virtual bool Store(Path& to) const
        {
            cv::FileStorage fs(to.string(), cv::FileStorage::WRITE);

            if (!fs.isOpened())
            {
                E_ERROR << "error opening file " << to << " for writing";
                return false;
            }

            return Store(fs);
        }

        virtual bool Restore(const Path& from)
        {
            cv::FileStorage fs(from.string(), cv::FileStorage::READ);

            if (!fs.isOpened())
            {
                E_ERROR << "error opening file " << from << " for reading";
                return false;
            }

            return Restore(fs.root());
        }

    protected:
        virtual bool Append(Path& to, const T& data) const     { return data.Store(to);     }
        virtual bool Retrieve(const Path& from, T& data) const { return data.Restore(from); }

    private:
        Path    m_root;
        Strings m_filenames;
    };
}
#endif // SEQ_FILE_STORE_HPP

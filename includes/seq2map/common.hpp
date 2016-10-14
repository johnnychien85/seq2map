#ifndef COMMON_HPP
#define COMMON_HPP
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <opencv2/opencv.hpp>

#if defined ( WIN32 )
#define __func__ __FUNCTION__
#endif

#define E_LOG(lvl)	BOOST_LOG_TRIVIAL(lvl) << __func__  << " : "
#define E_TRACE     E_LOG(trace)
#define E_INFO		E_LOG(info)
#define E_WARNING	E_LOG(warning)
#define E_ERROR		E_LOG(error)
#define E_FATAL		E_LOG(fatal)

namespace seq2map
{
    typedef std::string String;
    typedef std::vector<String> Strings;

    typedef boost::filesystem::path Path;
    typedef std::vector<Path> Paths;

    typedef cv::Point2f Point2F;
    typedef cv::Point3f Point3F;
    typedef std::vector<Point2F> Points2F;
    typedef std::vector<Point3F> Points3F;

    typedef cv::Point2d Point2D;
    typedef cv::Point3d Point3D;
    typedef std::vector<Point2D> Points2D;
    typedef std::vector<Point3D> Points3D;

    typedef std::list<size_t> Indices;
    const size_t INVALID_INDEX = (size_t) -1;
    Indices makeIndices(size_t start, size_t end);

    double rad2deg(double radian);
    double deg2rad(double degree);
    double rms(const cv::Mat& e);
    String mat2string(const cv::Mat& x, const String& name = "", size_t precision = 6);
    bool mat2raw(const cv::Mat& im, const Path& path);
    bool checkCameraMatrix(const cv::Mat& K);

    // file system
    bool dirExists(const Path& path);
    bool fileExists(const Path& path);
    bool makeOutDir(const Path& path);
    bool initLogFile(const Path& path = "");
    size_t filesize(const Path& path);
    Paths enumerateFiles(const Path& root, const String& ext);
    Paths enumerateFiles(const Path& sample);
    Paths enumerateDirs(const Path& root);
    String removeFilenameExt(const String& filename);
    Path fullpath(const Path& path);
    Path getRelativePath(const Path& path, const Path& base);

    // string processing
    String makeNameList(Strings names);
    Strings explode(const String& string, char delimiter);
    String indices2string(const Indices& idx);
    String size2string(const cv::Size& size);
    cv::Mat strings2mat(const Strings& strings, const cv::Size& matSize);
    bool replace(String& subject, const String& from, const String& to);

    // image processing
    cv::Mat rgb2gray(const cv::Mat& rgb);
    cv::Mat gray2rgb(const cv::Mat& gray);
    cv::Mat imfuse(const cv::Mat& im0, const cv::Mat& im1);

    /**
     * Indexed item
     */
    class Indexed
    {
    public:
        Indexed(size_t index = INVALID_INDEX) : m_index(index) {}
        virtual ~Indexed() {}
        inline void SetIndex(size_t index) { m_index = index; }
        inline size_t GetIndex() const {return m_index;}
    private:
        size_t m_index;
    };

    /**
     * A common interface for parameterised objects.
     */
    class Parameterised
    {
    public:
        typedef boost::program_options::options_description Options;

        virtual void WriteParams(cv::FileStorage& fs) const = 0;
        virtual bool ReadParams(const cv::FileNode& fn) = 0;
        virtual void ApplyParams() = 0;
        virtual Options GetOptions(int flag) = 0;
    };

    /**
     * An interface to represent any object that is storable to and later
     * restorable from a file.
     */
    template<class T1, class T2 = T1>
    class Persistent
    {
    public:
        virtual bool Store(T1& path) const = 0;
        virtual bool Restore(const T2& path) = 0;
    };

    /**
     * The storage class presents a collection of persistent objects
     * sequentially stored in the same folder.
     */
    template<class DerivedPersistent> class SequentialPersistentLoader : public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
        SequentialPersistentLoader() {}
        SequentialPersistentLoader(const Path& root, size_t capacity) : m_root(root) { Allocate(capacity); }
        virtual ~SequentialPersistentLoader() {}
        //inline void Add(const String& filename) { m_persistents.insert(_Tp(filename)); }
        inline void SetRootPath(const Path& root) {m_root = root; }
        inline Path GetRootPath() const { return m_root; }
        inline void Add(const String& filename) { m_filenames.push_back(filename); }
        inline void Allocate(size_t capacity) { m_filenames.reserve(capacity); }
        inline const Strings& GetFilenames() const { return m_filenames; }
        inline size_t GetSize() const { return m_filenames.size(); }
        virtual bool Store(cv::FileStorage& fs) const { return Store(fs, m_root); };
        virtual bool Store(cv::FileStorage& fs, const Path& root) const
        {
            fs << "root" << root.string();
            fs << "items" << (int) m_filenames.size();
            fs << "files" << "[";
            BOOST_FOREACH(const String& filename, m_filenames)
            {
                fs << filename;
            }
            fs << "]";
            return true;
        }

        virtual bool Restore(const cv::FileNode& fn)
        {
            m_filenames.clear();
            try
            {
                cv::FileNode filesNode = fn["files"];
                String root;
                int capacity = 0;

                fn["root"]  >> root;
                fn["items"] >> capacity;

                m_root = root;
                m_filenames.reserve(capacity);

                for (cv::FileNodeIterator itr = filesNode.begin(); itr != filesNode.end(); itr++)
                {
                    m_filenames.push_back((String)*itr);
                }
            }
            catch (std::exception& ex)
            {
                E_ERROR << "error restoring sequential persistent loader";
                E_ERROR << ex.what();
                return false;
            }
            return true;
        }

        /**
         * element accessor
         */
        bool Retrieve(size_t idx, DerivedPersistent& persistent) const
        {
            return idx < m_filenames.size() ? persistent.Restore(m_root / m_filenames[idx]) : false;
        }
        //DerivedPersistent& operator[] (size_t idx)
        //{
        //    _Tp& tp = m_persistents.at(idx);
        //    tp.restored = tp.restored ? tp.restored : tp.persistent.Restore(m_root / tp.filename);
        //
        //    if (!tp.restored)
        //    {
        //        E_ERROR << "error restoring from \"" << (m_root / tp.filename).string() << "\"";
        //    }
        //
        //    return tp.persistent;
        //}
    protected:
        //struct _Tp
        //{
        //    _Tp(const String& filename) : filename(filename) {}
        //    const String filename;
        //    DerivedPersistent persistent;
        //    bool restored = false;
        //};
        Path m_root;
        Strings m_filenames;
        //std::vector<_Tp> m_persistents;
    };

    /**
     * A wrapper of cv::Mat with disk storage backend.
     */
    class PersistentImage : public Persistent<Path>
    {
    public:
        virtual bool Store(Path& path) const { return cv::imwrite(path.string(), im); };
        virtual bool Restore(const Path& path) { return !(im = cv::imread(path.string())).empty(); };

        cv::Mat im;
    };

    /**
     * A class to measure the output of a process in unit time.
     */
    class Speedometre
    {
    public:
        /* ctor */ Speedometre(const String& name = "Unamed Speedometre", const String& unit = "unit/s")
            : m_displayName(name), m_displayUnit(unit) { Reset(); }
        void Start();
        void Stop(size_t amount);
        void Update(size_t amount);
        void Reset();
        String GetUnit() const { return m_displayUnit; }
        double GetElapsedSeconds() const;
        inline double GetSpeed() const {return (double)m_accumulated / GetElapsedSeconds(); }
        inline double GetFrequency() const { return (double)m_freq / GetElapsedSeconds(); }
        String ToString() const;
    protected:
        String m_displayName;
        String m_displayUnit;
        bool   m_activated;
        size_t m_accumulated;
        size_t m_freq;
        boost::timer::cpu_timer m_timer;
    };

    class AutoSpeedometreMeasure
    {
    public:
        /* ctor */ AutoSpeedometreMeasure(Speedometre& metre, size_t amount)
                   : m_metre(metre), m_amount(amount) { m_metre.Start(); };
        /* dtor */ virtual ~AutoSpeedometreMeasure()  { m_metre.Stop(m_amount); }
    protected:
        Speedometre& m_metre;
        const size_t m_amount;
    };

    /**
     * Templated factory class to instantiate an object derived from Base
     * given a key indicating its concrete class.
     */
    template<class Key, class Base> class Factory
    {
    public:
        typedef boost::shared_ptr<Base> BasePtr;
        typedef BasePtr(*CtorType)();

        BasePtr Create(const Key& key) const
        {
            typename Registry::const_iterator itr = m_ctors.find(key);

            if (itr == m_ctors.end())
            {
                E_ERROR << "unknown key " << key;
                return BasePtr();
            }

            return itr->second();
        }

        std::vector<Key> GetRegisteredKeys() const
        {
            std::vector<Key> keys;
            BOOST_FOREACH(typename Registry::value_type v, m_ctors)
            {
                keys.push_back(v.first);
            }

            return keys;
        }

    protected:
        template<class Derived> static BasePtr Constructor()
        {
            return BasePtr(new Derived());
        }

        template<class Derived> void Register(const Key& key)
        {
            CtorType ctor = &Factory::Constructor<Derived>;

            bool newKey = m_ctors.insert(typename Registry::value_type(key, ctor)).second;

            if (!newKey)
            {
                E_WARNING << "duplicated key " << key;
            }
        }

    private:
        typedef std::map<Key, CtorType> Registry;
        Registry m_ctors;
    };

    /**
     * Templated singleton class to create the only one static instance of class.
     */
    template<class T> class Singleton
    {
    public:
        typedef boost::shared_ptr<T> Ptr;

        static T  GetInstancePtr() { return m_instance ? m_instance : (m_instance = Ptr(new T)); }
        static T& GetInstance()    { return *GetInstancePtr(); }

    protected:


    private:
        static Ptr m_instance;
    };
}

#endif //COMMON_HPP

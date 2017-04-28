#ifndef COMMON_HPP
#define COMMON_HPP
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <opencv2/opencv.hpp>

/**
 * macros for logging
 */
#if defined ( WIN32 )
#define __func__ __FUNCTION__
#endif

#define E_LOG(lvl)	BOOST_LOG_TRIVIAL(lvl) << __func__  << " : "
#define E_TRACE     E_LOG(trace)
#define E_INFO		E_LOG(info)
#define E_WARNING	E_LOG(warning)
#define E_ERROR		E_LOG(error)
#define E_FATAL		E_LOG(fatal)

/**
 * type definitions
 */
namespace seq2map
{
    typedef std::string String;
    typedef std::vector<String> Strings;

    typedef boost::filesystem::path Path;
    typedef std::vector<Path> Paths;

    typedef boost::posix_time::ptime Time;

    typedef cv::Point2f Point2F;
    typedef cv::Point3f Point3F;
    typedef std::vector<Point2F> Points2F;
    typedef std::vector<Point3F> Points3F;

    typedef cv::Point2d Point2D;
    typedef cv::Point3d Point3D;
    typedef std::vector<Point2D> Points2D;
    typedef std::vector<Point3D> Points3D;

    typedef std::list<size_t> Indices;
    const size_t INVALID_INDEX = (size_t)-1;
    Indices makeIndices(size_t start, size_t end);
}

/**
 * helper functions
 */
namespace seq2map
{
    // math
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

    // miscs.
    Time unow();
    String time2string(const Time& time);
}

/**
 * classes
 */
namespace seq2map
{
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
        virtual Options GetOptions(int flag = 0) = 0;
    };

    /**
     * An interface to represent any class that is storable to and later
     * restorable from type T1 and T2 respectively.
     */
    template<class T1, class T2 = T1>
    class Persistent
    {
    public:
        virtual bool Store(T1& to) const = 0;
        virtual bool Restore(const T2& from) = 0;
    };

    /**
     * Linearily spaced vector.
     */
    template<typename T>
    class LinearSpacedVec
    {
    public:
        LinearSpacedVec(T begin, T end, size_t segs = 0)
        : begin(begin), end(end), segs(segs > 0 ? segs : static_cast<size_t>(end - begin)) {}
        
        virtual ~LinearSpacedVec() {}

        inline T operator[] (size_t idx) const
        { return idx < segs ? static_cast<T>(static_cast<double>(begin) + static_cast<double>(idx) * static_cast<double>((end - begin)) / static_cast<double>(segs)) : end; }

        void GetLinearMappingTo(const LinearSpacedVec<T>& dst, double& alpha, double& beta) const
        {
            alpha = static_cast<double>(dst.segs) / static_cast<double>(dst.end - dst.begin);
            beta  = -alpha * static_cast<double>(begin);
        }

        T begin;
        T end;
        size_t segs;
    };

    /**
     * The storage class presents a collection of persistent objects
     * sequentially stored in the same folder.
     */
    /*
    template<class DerivedPersistent>
    class SequentialPersistentLoader : public Persistent<cv::FileStorage, cv::FileNode>
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

        //
        // element accessor
        //
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
    */

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

    /**
     * A scoped implementation of speedometre class.
     */
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
        typedef std::vector<Key> Keys;

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

        Keys GetRegisteredKeys() const
        {
            Keys keys;
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

            bool unique = m_ctors.insert(typename Registry::value_type(key, ctor)).second;

            if (!unique)
            {
                E_WARNING << "duplicated key " << key;
            }
        }

    private:
        typedef std::map<Key, CtorType> Registry;
        Registry m_ctors;
    };

    /**
     * Templated singleton class to create the only one static instance of class T.
     */
    template<class T> class Singleton
    {
    public:
        Singleton(Singleton const&);         // deleted - no copy constructor is allowed
        void operator=(Singleton<T> const&); // deleted - no copying is allowed

        static T& GetInstance()
        {
            static T instance;

            if (!s_init)
            {
                instance.Init();
                s_init = true;
            }
            
            return instance;
        }

    protected:
        Singleton()  {}
        ~Singleton() {}

        virtual void Init() {}

        static bool s_init;
    };

    template<class T> bool Singleton<T>::s_init = false;

    /**
     * Application class
     */
    class App
    {
    public:
        int Run();

    protected:
        typedef boost::program_options::options_description Options;
        typedef boost::program_options::positional_options_description Positional;

        /* ctor */ App(int argc, char* argv[]);
        /* dtor */ virtual ~App() {}
        virtual void SetOptions(Options& general, Options& hidden, Positional& positional) = 0;
        virtual void ShowSynopsis() const;
        virtual void ShowHelp(const Options& options) const = 0;
        virtual bool ProcessUnknownArgs(const Strings& args);
        virtual bool Init() = 0;
        virtual bool Execute() = 0;

        const Path m_exec;

    private:
        boost::program_options::command_line_parser m_parser;
        bool m_help;
        Path m_logfile;
    };
}

/**
 * Some extensions for OpenCV
 */
namespace cv
{
    static void write(FileStorage& fs, const cv::String& bane, size_t value)
    {
        fs << static_cast<int>(value);
    }

    static void read(const FileNode& fn, size_t& value, const size_t& default_value = 0)
    {
        int x;
        fn >> x;

        value = static_cast<size_t>(x);
    }

    static void write(FileStorage& fs, const cv::String& name, const seq2map::Path& path)
    {
        fs << path.string();
    }

    static void read(const FileNode& fn, seq2map::Path& value, const seq2map::Path& default_value = seq2map::Path())
    {
        String x;
        fn >> x;

        value = fn.empty() ? default_value : seq2map::Path(x.c_str());
    }
}

#endif //COMMON_HPP

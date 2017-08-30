#ifndef COMMON_HPP
#define COMMON_HPP
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <opencv2/opencv.hpp>

/**
 * Macros for logging
 */
#if defined ( WIN32 )
#define __func__ __FUNCTION__
#endif

#define E_LOG(lvl) BOOST_LOG_TRIVIAL(lvl) << __func__  << " : "
#define E_TRACE    E_LOG(trace)
#define E_DEBUG    E_LOG(debug)
#define E_INFO	   E_LOG(info)
#define E_WARNING  E_LOG(warning)
#define E_ERROR    E_LOG(error)
#define E_FATAL    E_LOG(fatal)

/**
 * Type definitions
 */
namespace seq2map
{
    typedef std::string String;
    typedef std::vector<String> Strings;

    typedef boost::filesystem::path Path;
    typedef std::vector<Path> Paths;

    typedef boost::posix_time::ptime Time;

    typedef cv::Point2i Point2I;
    typedef cv::Point2f Point2F;
    typedef cv::Point3f Point3F;
    typedef std::vector<Point2I> Points2I;
    typedef std::vector<Point2F> Points2F;
    typedef std::vector<Point3F> Points3F;

    typedef cv::Point2d Point2D;
    typedef cv::Point3d Point3D;
    typedef std::vector<Point2D> Points2D;
    typedef std::vector<Point3D> Points3D;

    typedef std::list<size_t> IndexList;
    typedef std::vector<size_t> Indices;
    const size_t INVALID_INDEX = (size_t)-1;
    IndexList makeIndices(size_t start, size_t end);
}

/**
 * Helper functions
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
    int sub2symind(int i, int j, int n);
    cv::Mat symmat(const cv::Mat& A);
    cv::Mat symmat(const cv::Mat& a, int n);
    bool checkPositiveDefinite(const cv::Mat& A);

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
    String indices2string(const IndexList& idx);
    String size2string(const cv::Size& size);
    cv::Mat strings2mat(const Strings& strings, const cv::Size& matSize);
    bool replace(String& subject, const String& from, const String& to);

    // image processing
    cv::Mat rgb2gray(const cv::Mat& rgb);
    cv::Mat gray2rgb(const cv::Mat& gray);
    cv::Mat imfuse(const cv::Mat& im0, const cv::Mat& im1);
    cv::Mat interp(const cv::Mat& src, const cv::Mat& sub, int method = cv::INTER_NEAREST, bool useGpu = false);

    // miscs.
    Time unow();
    String time2string(const Time& time);
}

/**
 * Classes
 */
namespace seq2map
{
    /**
     * Named item
     */
    class Named
    {
    public:
        Named() : m_name("unnamed") {}

        inline void SetName(const String& name) { m_name = name; }
        inline String GetName() const           { return m_name; }

    private:
        String m_name;
    };

    /**
     * Indexed item
     */
    class Indexed
    {
    public:
        Indexed(size_t index = INVALID_INDEX) : m_index(index) {}
        virtual ~Indexed() {}
        inline virtual void SetIndex(size_t index) { m_index = index; }
        inline size_t GetIndex() const { return m_index; }
        inline bool IsOkay() const { return m_index != INVALID_INDEX; }
        inline bool operator<  (const Indexed& rhs) const { return m_index <  rhs.m_index; }
        inline bool operator== (const Indexed& rhs) const { return m_index == rhs.m_index; }
        inline bool operator!= (const Indexed& rhs) const { return !(*this == rhs); }

    private:
        size_t m_index;
    };

    /**
     * An interface for referenced element.
     */
    template<typename T>
    class Referenced : public boost::enable_shared_from_this<T>
    {
    public:
        typedef T* Ptr;
        typedef const T* ConstPtr;

        typedef boost::shared_ptr<T> Own;
        typedef boost::shared_ptr<const T> ConstOwn;

        typedef boost::weak_ptr<T> Ref;
        typedef boost::weak_ptr<const T> ConstRef;
    };

    /**
     * An compound interface for indexed and referenced classes.
     */
    template<typename T>
    class IndexReferenced
    : public Indexed, public Referenced<T>
    {
    public:
        struct Less
        {
            bool operator() (const Own& lhs, const Own& rhs)
            {
                return *lhs < *rhs; // refer to Indexed::operator<
            }
        };

        typedef std::map<size_t, Own> Map;
        typedef std::set<Own, Less> Set;

        static Own New(size_t index) { return Own(new T(index)); }
        
        static ConstOwn Find(const Map& map, size_t index)
        {
            Map::const_iterator itr = map.find(index);
            return (itr == map.end()) ? ConstOwn() : itr->second;
        }

        IndexReferenced(size_t index) : Indexed(index) {}

        // disallow change of index
        virtual void SetIndex(size_t index) { E_ERROR << "index-referenced item cannot set index after creation"; }

        bool Join(Map& map) { return map.insert(Map::value_type(GetIndex(), shared_from_this())).second; }

        static const T Null;
    };

    template<class T> bool IndexReferenced<T>::Null(INVALID_INDEX);

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
            BOOST_FOREACH (typename Registry::value_type v, m_ctors)
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
        Singleton(Singleton const&);         // deleted - no copy constructor allowed
        void operator=(Singleton<T> const&); // deleted - no copying allowed

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
     * An interface to represent any class that can be stored to T.
     */
    template<class T> class Storable
    {
    public:
        virtual bool Store(T& to) const = 0;
    };

    /**
     * An interface to represent any class that can be restored from T.
     */
    template<class T> class Restorable
    {
    public:
        virtual bool Restore(const T& from) = 0;
    };

    /**
     * An interface to represent any class that is storable to and later
     * restorable from type T1 and T2 respectively.
     */
    template<class T1, class T2 = T1>
    class Persistent
    : public Storable<T1>,
      public Restorable<T2>
    {
    };

    /**
     *
     */
    class PersistentMat : Persistent<Path>
    {
    public:
        PersistentMat(cv::Mat& mat) : mat(mat) {}

        virtual bool Store(Path& to) const;
        virtual bool Restore(const Path& from);

        static String CvDepthToString(int depth);
        static int    StringToCvDepth(const String& depth);

        static void Dump(const cv::Mat& mat, std::ostream& os);
        static void Dump(std::istream& is, cv::Mat& mat);

        cv::Mat mat;

    private:
        static const String s_magicString;
    };

    /**
     * An interface to represent any class that can be serialised into
     * an array of type T.
     */
    template<typename T>
    class Vectorisable : public Persistent<std::vector<T>>
    {
    public:
        typedef std::vector<T> Vec;

        virtual bool Store(Vec& v) const = 0;
        virtual bool Restore(const Vec& v) = 0;

        /**
         * Assignment form of Store.
         */
        Vec ToVector() const
        {
            Vec v;
            return Store(v) ? v : Vec();
        }

        /**
         * Alias of Restore.
         */
        inline bool FromVector(const Vec& v) { return Restore(v); }

        /**
         * Get the dimension of the vector represent of class.
         * \return size_t Number of elements needed to store the class.
         */
        virtual size_t GetDimension() const = 0;
    };

    typedef Vectorisable<float>  VectorisableF;
    typedef Vectorisable<double> VectorisableD;

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
            //alpha = static_cast<double>(dst.segs) / static_cast<double>(dst.end - dst.begin);
            //alpha = static_cast<double>(dst.segs + 1) / static_cast<double>(segs + 1);
            //beta  = -alpha * static_cast<double>(begin);

            alpha = static_cast<double>(dst.end - dst.begin) / static_cast<double>(end - begin);
            beta  = -alpha * static_cast<double>(dst.begin);
        }

        inline bool operator== (const LinearSpacedVec<T>& vec)
        {
            return begin == vec.begin && end == vec.end && segs == vec.segs;
        }

        inline bool operator!= (const LinearSpacedVec<T>& vec)
        {
            return !((*this) == vec);
        }

        T begin;
        T end;
        size_t segs;
    };

    /**
     * Progress display.
     */
    class Progress
    {
    public:
        Progress(size_t tasks, size_t freq = 10)
        : m_divisor(static_cast<size_t>(std::ceil(static_cast<float>(tasks) / static_cast<float>(freq)))) {}

        inline bool IsMilestone(size_t i) const { return (i % m_divisor) == 0; }

    private:
        size_t m_tasks;
        size_t m_divisor;
    };

    /**
     * A class to measure the output of a process in unit time.
     */
    class Speedometre
    {
    public:
        Speedometre(const String& name = "Unamed Speedometre", const String& unit = "unit/s")
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
     *
     */
    class ColourMap
    {
    public:
        ColourMap(size_t colours);
        cv::Scalar GetColour(double val, double min, double max);

    protected:
        cv::Mat m_cmap;
    };
}

/**
 * Some extensions for OpenCV
 */
namespace cv
{
    static void write(FileStorage& fs, const cv::String& nane, size_t value)
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

#include<seq2map/common_impl.hpp>

#endif //COMMON_HPP

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

    typedef std::list<size_t> Indices;
    typedef std::vector<cv::Point2f> Points2F;
    typedef std::vector<cv::Point3f> Points3F;

    double rad2deg(double radian);
    double deg2rad(double degree);

    bool dirExists(const Path& path);
    bool makeOutDir(const Path& path);
    bool initLogFile(const Path& path = "");
    Paths enumerateFiles(const Path& root, const std::string& ext);
    Paths enumerateFiles(const Path& sample);
    Paths enumerateDirs(const Path& root);

    cv::Mat rgb2gray(const cv::Mat& rgb);
    cv::Mat imfuse(const cv::Mat& im0, const cv::Mat& im1);

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
    class Persistent
    {
    public:
        virtual bool Store(const Path& path) const = 0;
        virtual bool Restore(const Path& path) = 0;
    };

    class Speedometre
    {
    public:
        /* ctor */ Speedometre(const String& name = "Unamed Speedometre", const String& unit = "unit/s")
            : m_displayName(name), m_displayUnit(unit) { Reset(); }
        void Start();
        void Stop(size_t amount);
        void Reset();
        inline double GetSpeed() const;
        String ToString() const;
    protected:
        String m_displayName;
        String m_displayUnit;
        bool   m_activated;
        size_t m_accumulated;
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
            BOOST_FOREACH(Registry::value_type v, m_ctors)
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
}

#endif //COMMON_HPP

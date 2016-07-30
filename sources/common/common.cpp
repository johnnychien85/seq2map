#include <iomanip>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes/attribute.hpp>
#include <boost/log/attributes/clock.hpp>
#include <boost/algorithm/string.hpp>
#include <seq2map/common.hpp>

namespace fs = boost::filesystem;
namespace logging = boost::log;

namespace seq2map
{
    double rad2deg(double radian)
    {
        return radian * 180.0f / CV_PI;
    }

    double deg2rad(double degree)
    {
        return degree * CV_PI / 180.0f;
    }

    bool dirExists(const Path& path)
    {
        return fs::exists(path) && fs::is_directory(path);
    }

    bool makeOutDir(const Path& path)
    {
        if (dirExists(path)) return true;
        else return fs::create_directories(path);
    }

    Paths enumerateFiles(const Path& root, const std::string& ext)
    {
        Paths files;
        fs::directory_iterator endItr;

        for (fs::directory_iterator itr(root) ;
            itr != endItr ; itr++)
        {
            if (!fs::is_regular(*itr)) continue;

            bool extCheck = ext.empty() || boost::iequals(ext, itr->path().extension().string());
            if(extCheck) files.push_back(*itr);
        }

        return files;
    }

    Paths enumerateFiles(const Path& sample)
    {
        std::string ext = sample.extension().string();
        return enumerateFiles(ext.empty() ? sample : sample.parent_path(), ext);
    }

    Paths enumerateDirs(const Path& root)
    {
        Paths dirs;
        fs::directory_iterator endItr;

        for (fs::directory_iterator itr(root) ;
            itr != endItr ; itr++)
        {
            if (!fs::is_directory(*itr)) continue;
            dirs.push_back(*itr);
        }

        return dirs;
    }

    bool initLogFile(const Path& path)
    {
        try
        {
            logging::core::get()->add_global_attribute("TimeStamp", logging::attributes::local_clock());

            logging::add_file_log(path, logging::keywords::format = "[%TimeStamp%] %Message%");
            logging::add_console_log(std::cout);
        }
        catch (std::exception& ex)
        {
            std::cerr << "error logging to file \"" << path.string() << "\"" << std::endl;
		    std::cerr << ex.what() << std::endl;

            return false;
        }

        return true;
    }

    cv::Mat rgb2gray(const cv::Mat& rgb)
    {
        cv::Mat gray;
        switch (rgb.channels())
        {
        case 1: gray = rgb.clone(); break;
        case 3: cv::cvtColor(rgb, gray, CV_RGB2GRAY); break;
        default: E_ERROR << "the source image has " << rgb.channels() << " channels, while 1 or 3 expected!!";
        }
        return gray;
    }

    cv::Mat imfuse(const cv::Mat& im0, const cv::Mat& im1)
    {
        cv::Size sz = cv::Size(std::max(im0.cols, im1.cols), std::max(im0.rows, im1.rows));
        cv::Mat r = cv::Mat::zeros(sz, CV_8U);
        cv::Mat g = cv::Mat::zeros(sz, CV_8U);
        cv::Mat b = cv::Mat::zeros(sz, CV_8U);

        rgb2gray(im0).copyTo(r.rowRange(0, im0.rows).colRange(0, im0.cols));
        rgb2gray(im1).copyTo(b.rowRange(0, im1.rows).colRange(0, im1.cols));
        g = 0.5f * r + 0.5f * b;

        std::vector<cv::Mat> bgr;
        bgr.push_back(b);
        bgr.push_back(g);
        bgr.push_back(r);

        cv::Mat im;
        cv::merge(bgr, im);

        return im;
    }

    void Speedometre::Start()
    {
        if (!m_activated)
        {
            m_activated = true;
            m_timer.start();

            return;
        }

        if (!m_timer.is_stopped())
        {
            E_WARNING << "the operation takes no effect as the timer is already ticking!!";
            return;
        }

        m_timer.resume();
    }

    void Speedometre::Stop(size_t amount)
    {
        if (!m_activated || m_timer.is_stopped())
        {
            E_WARNING << "the operation takes no effect as the timer is not ticking!!";
            return;
        }

        m_timer.stop();
        m_accumulated += amount;
    }

    void Speedometre::Reset()
    {
        m_activated = false;
        m_accumulated = 0;
        m_timer.stop();
    }

    inline double Speedometre::GetSpeed() const
    {
        boost::timer::cpu_times elapsed = m_timer.elapsed();

        return (double) m_accumulated / 
            (double) boost::chrono::duration_cast<boost::chrono::seconds>(
                boost::chrono::nanoseconds(elapsed.system + elapsed.user)).count();
    }

    String Speedometre::ToString() const
    {
        std::stringstream ss;
        ss << m_displayName << ": " << std::fixed << std::setprecision(2) << GetSpeed() << " " << m_displayUnit;
        
        return ss.str();
    }
}

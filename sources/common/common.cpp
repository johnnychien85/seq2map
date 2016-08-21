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

    double rms(const cv::Mat& e)
    {
        size_t n = e.total();
        bool rowvec = n == e.rows;
        bool colvec = n == e.cols;

        assert(rowvec || colvec);

        cv::Mat e2 = rowvec ? e.t() * e : e * e.t();
        cv::Mat e64f;
        e2.convertTo(e64f, CV_64F);

        return std::sqrt(e2.ptr<double>()[0] / n);
    }

    String mat2string(const cv::Mat& x, const String& name, size_t precision)
    {
        cv::Mat x64f;
        x.convertTo(x64f, CV_64F);

        std::stringstream ss;

        ss << (name.empty() ? "" : name + " = ") << "[";

        for (size_t i = 0; i < x64f.rows; i++)
        {
            bool lastrow = i == x64f.rows - 1;
            for (size_t j = 0; j < x64f.cols; j++)
            {
                bool lastcol = j == x64f.cols - 1;

                ss << std::setprecision(precision) << x64f.at<double>(static_cast<int>(i), static_cast<int>(j));
                ss << (lastcol ? "" : ", ");
            }
            ss << (lastrow ? "" : "; ");
        }

        ss << "];";

        return ss.str();
    }

    bool checkCameraMatrix(const cv::Mat& K)
    {
        if (K.rows != 3 || K.cols != 3 || (K.type() != CV_32F && K.type() != CV_64F))
        {
            E_WARNING << "the camera matrix has to be a 3-by-3 single/double matrix";
            return false;
        }

        cv::Mat K64f;
        K.convertTo(K64f, CV_64F);

        if (K64f.at<double>(1,0) != 0 || 
            K64f.at<double>(2,0) != 0 || K64f.at<double>(2,1) != 0 ||
            K64f.at<double>(2,2) != 1)
        {
            E_WARNING << "the camera matrix has to be an upper triangular matrix with the third diagonal entry set to 1";
            return false;
        }

        return true;
    }

    bool dirExists(const Path& path)
    {
        return fs::exists(path) && fs::is_directory(path);
    }

    bool fileExists(const Path& path)
    {
        return fs::exists(path) && fs::is_regular(path);
    }

    bool makeOutDir(const Path& path)
    {
        if (dirExists(path)) return true;
        else return fs::create_directories(path);
    }

    size_t filesize(const Path& path)
    {
        return fs::file_size(path);
    }

    Paths enumerateFiles(const Path& root, const String& ext)
    {
        Paths files;
        fs::directory_iterator endItr;

        for (fs::directory_iterator itr(root); itr != endItr ; itr++)
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

    String removeFilenameExt(const String& filename)
    {
        return filename.substr(0, filename.find_last_of("."));
    }

    Path fullpath(const Path& path)
    {
        return fs::absolute(fs::canonical(path));
    }

    Path getRelativePath(const Path& path, const Path& base)
    {
        return path.empty() ? path : fs::relative(fullpath(path), fullpath(base));
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

    String makeNameList(Strings names)
    {
        std::stringstream ss;
        for (size_t i = 0; i < names.size(); i++)
        {
            if (i > 0) // make a comma-separated sentence
            {
                ss << (i < names.size() - 1 ? ", " : " and ");
            }
            ss << "\"" << names[i] << "\"";
        }
        ss << ".";

        return ss.str();
    }

    Strings explode(const String& string, char delimiter)
    {
        String tok;
        Strings toks;
        std::istringstream iss(string);

        while (getline(iss, tok, delimiter)) toks.push_back(tok);
        return toks;
    }
    
    String size2string(const cv::Size& size)
    {
        std::stringstream ss;
        ss << size.height << "x" << size.width;

        return ss.str();
    }

    cv::Mat strings2mat(const Strings& strings, const cv::Size& matSize)
    {
        if (matSize.height * matSize.width != strings.size())
        {
            E_ERROR << "given strings with " << strings.size()
                    << " element(s) do not fit the desired matrix size of " << size2string(matSize);
            return cv::Mat();
        }

        std::vector<float> data(strings.size());

        for (size_t i = 0; i < strings.size(); i++)
        {
            data[i] = (float) atof(strings[i].c_str());
        }

        return cv::Mat(data).reshape(0, matSize.height).clone();
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

    cv::Mat gray2rgb(const cv::Mat& gray)
    {
        cv::Mat rgb;
        cv::cvtColor(gray, rgb, CV_GRAY2BGR);

        return rgb;
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

    double Speedometre::GetElapsedSeconds() const
    {
        boost::timer::cpu_times elapsed = m_timer.elapsed();

        return (double) boost::chrono::duration_cast<boost::chrono::seconds>(
            boost::chrono::nanoseconds(elapsed.system + elapsed.user)).count();
    }

    String Speedometre::ToString() const
    {
        std::stringstream ss;
        ss << m_displayName << ": " << std::fixed << std::setprecision(2) << GetSpeed() << " " << m_displayUnit;
        
        return ss.str();
    }
}

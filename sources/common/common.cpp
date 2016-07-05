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
}

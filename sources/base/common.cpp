#include <iomanip>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes/attribute.hpp>
#include <boost/log/attributes/clock.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <seq2map/common.hpp>

namespace fs = boost::filesystem;
namespace logging = boost::log;

using namespace seq2map;

namespace seq2map
{
    Indices makeIndices(size_t start, size_t end)
    {
        Indices idx;
        for (size_t i = start; i < end + 1; i++)
        {
            idx.push_back(i);
        }

        return idx;
    }

    double rad2deg(double radian)
    {
        return radian * 180.0f / CV_PI;
    }

    double deg2rad(double degree)
    {
        return degree * CV_PI / 180.0f;
    }

    int sub2symind(int i, int j, int n)
    {
        assert(i < n && j < n);

        if (i < j) return i * n - (i - 1) * i / 2 + j - i;
        else       return j * n - (j - 1) * j / 2 + i - j;
    }

    cv::Mat symmat(const cv::Mat& A)
    {
        assert(A.rows == A.cols);
        cv::Mat a(1, (A.rows + 1) * A.rows / 2, A.type());
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = i; j < A.cols; j++)
            {
                a.at<double>(sub2symind(i, j, A.rows)) = A.at<double>(i, j);
            }
        }
        return a;
    }

    cv::Mat symmat(const cv::Mat& a, int n)
    {
        assert(a.total() == (n+1) * n / 2);
        cv::Mat A(n, n, a.type());
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                A.at<double>(i, j) = a.at<double>(sub2symind(i, j, n));
            }
        }
        return A;
    }

    bool checkPositiveDefinite(const cv::Mat& A)
    {
        if (A.rows != A.cols || A.channels() != 1) return false;

        for (int i = 0; i < A.rows; i++)
        {
            if (cv::determinant(A.rowRange(0, i + 1).colRange(0, i + 1)) < 0)
            {
                return false;
            }
        }

        return true;
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


    bool mat2raw(const cv::Mat& im, const Path& path)
    {
        if (im.channels() > 1)
        {
            E_ERROR << "multi-channel images not supported (#ch=" << im.channels() << ")";
            return false;
        }

        if (im.dims > 2)
        {
            E_ERROR << "writing of " << im.dims << "-d matrix not supported";
            return false;
        }

        std::ofstream f(path.string().c_str(), std::ios::out | std::ios::binary);

        if (!f.is_open())
        {
            E_ERROR << "error opening output stream";
            return false;
        }

        f.write((char*)im.data, im.elemSize() * im.total());
        f.close();

        return true;
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

        if (K64f.at<double>(1, 0) != 0 ||
            K64f.at<double>(2, 0) != 0 || K64f.at<double>(2, 1) != 0 ||
            K64f.at<double>(2, 2) != 1)
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
        try
        {
            if (dirExists(path)) return true;
            else return fs::create_directories(path);
        }
        catch (std::exception& ex)
        {
            E_ERROR << "error making output directory " << path;
            E_ERROR << ex.what();

            return false;
        }
    }

    size_t filesize(const Path& path)
    {
        return fs::file_size(path);
    }

    Paths enumerateFiles(const Path& root, const String& ext)
    {
        Paths files;
        const fs::directory_iterator endItr;

        for (fs::directory_iterator itr(root); itr != endItr; itr++)
        {
            if (!fs::is_regular(*itr)) continue;

            bool check = ext.empty() || boost::iequals(ext, itr->path().extension().string());

            if (check)
            {
                files.push_back(*itr);
            }
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
        if (!dirExists(root)) return Paths();

        Paths dirs;
        const fs::directory_iterator endItr;

        try
        {
            for (fs::directory_iterator itr(root); itr != endItr; itr++)
            {
                if (!fs::is_directory(*itr)) continue;
                dirs.push_back(*itr);
            }
        }
        catch (std::exception& ex)
        {
            E_ERROR << "error enumerating directories in " << root;
            E_ERROR << ex.what();
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
#if BOOST_VERSION >= 106000
        // since Boost 1.60 we have a nice useful function
        return path.empty() ? path : fs::relative(fullpath(path), fullpath(base));
#else
        // for older Boost we have the code taken from a ticket
        if (path.empty()) return path;

        Path fullPath = fullpath(path);
        Path fullBase = fullpath(base);
        Path relPath;

        fs::path::const_iterator pathItr = fullPath.begin(), pathEnd = fullPath.end();
        fs::path::const_iterator baseItr = fullBase.begin(), baseEnd = fullBase.end();

        // skip the common part
        while (pathItr != pathEnd && baseItr != baseEnd && *pathItr == *baseItr)
        {
            pathItr++;
            baseItr++;
        }

        while (baseItr != baseEnd)
        {
            if (*baseItr != ".") relPath /= "..";
            baseItr++;
        }

        while (pathItr != pathEnd)
        {
            relPath /= *pathItr;
            pathItr++;
        }

        return relPath;
#endif
    }

    bool initLogFile(const Path& path)
    {
        try
        {
            logging::core::get()->add_global_attribute("TimeStamp", logging::attributes::local_clock());
            logging::add_console_log(std::cout, logging::keywords::format = "[%TimeStamp%] %Message%");

            if (!path.empty())
            {
                logging::add_file_log(path, logging::keywords::format = "[%TimeStamp%] %Message%");
            }
        }
        catch (std::exception& ex)
        {
            std::cerr << "error logging to file " << path.string() << std::endl;
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

    String indices2string(const Indices& indices)
    {
        std::stringstream ss;
        Indices::const_iterator idx, last = indices.end();

        for (idx = indices.begin(); idx != last; idx++)
        {
            ss << (*idx) << ((boost::next(idx) != last) ? ", " : "");
        }

        return ss.str();
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
            data[i] = (float)atof(strings[i].c_str());
        }

        return cv::Mat(data).reshape(0, matSize.height).clone();
    }

    bool replace(String& subject, const String& from, const String& to)
    {
        size_t start_pos = subject.find(from);

        if (start_pos == String::npos) return false;
        subject.replace(start_pos, from.length(), to);

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

    cv::Mat interp(const cv::Mat& src, const cv::Mat& sub, int method, bool useGpu)
    {
        if (sub.channels() != 2)
        {
            E_ERROR << "subscript matrix has to have 2 channels";
            return cv::Mat();
        }

        cv::Mat dst;

        if (useGpu && cv::cuda::getCudaEnabledDeviceCount() > 0)
        {
            cv::cuda::GpuMat gpuSrc;
            cv::cuda::GpuMat gpuDst;
            cv::cuda::GpuMat gpuMapX, gpuMapY;

            std::vector<cv::Mat> xy;
            cv::split(sub, xy);

            xy[0].convertTo(xy[0], CV_32F);
            xy[1].convertTo(xy[1], CV_32F);

            gpuSrc.upload(src);
            gpuMapX.upload(xy[0]);
            gpuMapY.upload(xy[1]);

            cv::cuda::remap(gpuSrc, gpuDst, gpuMapX, gpuMapY, method, cv::BORDER_REPLICATE);
            gpuDst.download(dst);
        }
        else
        {
            if (sub.rows < SHRT_MAX && sub.cols < SHRT_MAX)
            {
                cv::remap(src, dst, sub, cv::Mat(), method);
            }
            else
            {
                const int n = static_cast<int>(sub.total());
                const int k = src.channels();
                const int stride = SHRT_MAX - 1;

                cv::Mat map = sub.reshape(1, n);

                dst = cv::Mat(sub.rows, sub.cols, src.type());
                dst = dst.reshape(1, n);

                for (int i = 0; i < n; i += stride)
                {
                    const int i0 = i;
                    const int in = std::min(i + stride, n);
                    cv::remap(src, dst.rowRange(i0, in).reshape(k), map.rowRange(i0, in).reshape(2), cv::Mat(), method, cv::BORDER_REPLICATE);
                }

                dst = dst.reshape(k, sub.rows);
            }
        }

        return dst;
    }

    Time unow()
    {
        return boost::posix_time::microsec_clock::local_time();
    }

    String time2string(const Time& time)
    {
        return boost::posix_time::to_simple_string(time);
    }
}

//==[ PersistentMat ]=========================================================//

const seq2map::String PersistentMat::s_magicString = "CVMAT";

bool PersistentMat::Store(Path& to) const
{
    if (mat.dims > 2)
    {
        E_ERROR << "multi-dimensional matrix not supported";
        return false;
    }

    std::ofstream os(to.string().c_str(), std::ios::out | std::ios::binary);

    if (!os.is_open())
    {
        E_ERROR << "error opening output stream to " << to;
        return false;
    }

    try
    {
        // write header..
        os << s_magicString << " ";
        os << CvDepthToString(mat.depth()) << " ";

        const int m = mat.rows;
        const int n = mat.cols;
        const int k = mat.channels();

        os.write((char*)&m, sizeof m);
        os.write((char*)&n, sizeof n);
        os.write((char*)&k, sizeof k);

        Dump(mat, os); // write data

        os.close(); // finished
    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught: " << ex.what();
        return false;
    }

    return true;
}
    
bool PersistentMat::Restore(const Path& from)
{
    std::ifstream is(from.string().c_str(), std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        E_ERROR << "error opening input stream from " << from;
        return false;
    }

    try
    {
        char magic[5];
        is.read((char*)&magic, sizeof magic);

        if (!boost::equal(magic, s_magicString))
        {
            E_ERROR << "magic string not found";
            return false;
        }

        String depthString;
        std::getline(is, depthString, ' ');

        int depth = StringToCvDepth(depthString);

        if (depth == CV_USRTYPE1)
        {
            E_ERROR << "invalid depth";
            return false;
        }

        int m, n, k;
        is.read((char*)&m, sizeof m);
        is.read((char*)&n, sizeof n);
        is.read((char*)&k, sizeof k);

        mat = cv::Mat(m, n, CV_MAKE_TYPE(depth, k));

        Dump(is, mat);

        is.close();
    }
    catch (std::exception& ex)
    {
        E_ERROR << "exception caught: " << ex.what();
        return false;
    }

    return true;
}

String PersistentMat::CvDepthToString(int depth)
{
    switch (depth)
    {
    case CV_8U:       return "8U";  break;
    case CV_8S:       return "8S";  break;
    case CV_16U:      return "16U"; break;
    case CV_16S:      return "16S"; break;
    case CV_32S:      return "32S"; break;
    case CV_32F:      return "32F"; break;
    case CV_64F:      return "64F"; break;
    case CV_USRTYPE1: return "USR"; break;
    }

    E_WARNING << "unknown matrix depth " << depth;

    return CvDepthToString(CV_USRTYPE1);
}

int PersistentMat::StringToCvDepth(const String& depth)
{
    if      (depth == "8U")   return CV_8U;
    else if (depth == "8S")   return CV_8S;
    else if (depth == "16U")  return CV_16U;       
    else if (depth == "16S")  return CV_16S;  
    else if (depth == "32S")  return CV_32S;  
    else if (depth == "32F")  return CV_32F;  
    else if (depth == "64F")  return CV_64F; 
    else if (depth == "USR")  return CV_USRTYPE1;  

    E_WARNING << "unknown depth string \"" << depth << "\"";

    return StringToCvDepth("USR");
}

void PersistentMat::Dump(const cv::Mat& mat, std::ostream& os)
{
    if (mat.isContinuous())
    {
        os.write((char*)mat.data, mat.elemSize() * mat.total());
    }
    else
    {
        const size_t bytesPerRow = mat.elemSize() * static_cast<size_t>(mat.cols);
        for (int i = 0; i < mat.rows; i++)
        {
            os.write(mat.ptr<char>(i), bytesPerRow);
        }
    }
}

void PersistentMat::Dump(std::istream& is, cv::Mat& mat)
{
    if (mat.isContinuous())
    {
        is.read((char*)mat.data, mat.elemSize() * mat.total());
    }
    else
    {
        const size_t bytesPerRow = mat.elemSize() * static_cast<size_t>(mat.cols);
        for (int i = 0; i < mat.rows; i++)
        {
            is.read(mat.ptr<char>(i), bytesPerRow);
        }
    }
}

//==[ Speedometre ]===========================================================//

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

void Speedometre::Update(size_t amount)
{
    if (!m_activated || m_timer.is_stopped())
    {
        E_WARNING << "the operation takes no effect as the timer is not ticking!!";
        return;
    }

    m_accumulated += amount;
    m_freq++;
}

void Speedometre::Stop(size_t amount)
{
    Update(amount);
    m_timer.stop();
}

void Speedometre::Reset()
{
    m_activated = false;
    m_accumulated = 0;
    m_freq = 0;
    m_timer.stop();
}

double Speedometre::GetElapsedSeconds() const
{
    boost::timer::cpu_times elapsed = m_timer.elapsed();

    return (double) boost::chrono::duration_cast<boost::chrono::milliseconds>(
        boost::chrono::nanoseconds(elapsed.wall)).count() / 1000.0f;
}

String Speedometre::ToString() const
{
    std::stringstream ss;
    ss << m_displayName << ": " << std::fixed << std::setprecision(2) << GetSpeed() << " " << m_displayUnit;

    return ss.str();
}

ColourMap::ColourMap(size_t colours)
{
    m_cmap = cv::Mat(static_cast<int>(colours), 1, CV_64FC4);

    const int q0 = static_cast<int>((double)colours * 0.00f);
    const int q1 = static_cast<int>((double)colours * 0.25f);
    const int q2 = static_cast<int>((double)colours * 0.50f);
    const int q3 = static_cast<int>((double)colours * 0.75f);
    const int q4 = static_cast<int>((double)colours * 1.00f);

    m_cmap.at<cv::Scalar>(0) = cv::Scalar(148, 0, 211);

    for (int i = 1; i < static_cast<int>(colours); i++)
    {
        const double v = static_cast<double>(i);

        m_cmap.at<cv::Scalar>(i) = cv::Scalar(
            i < q2 ? 0 : (i < q3 ? 255 * (v - q2) / (q3 - q2) : 255), 
            i < q1 ? 255 * (v - q0) / (q1 - q0) : (i < q3 ? 255 : 255 * (q4 - v) / (q4 - q3)),
            i < q1 ? 255 : (i < q2 ? 255 * (q2 - v) / (q2 - q1) : 0),
            0
        );
    }
}

cv::Scalar ColourMap::GetColour(double val, double min, double max)
{
    const int cidx = static_cast<int>((double)m_cmap.rows * (val - min) / (max - min));
    return m_cmap.at<cv::Scalar>(cidx < 0 ? 0 : (cidx < m_cmap.rows ? cidx : m_cmap.rows - 1));
};

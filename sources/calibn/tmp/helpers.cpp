#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <calibn/helpers.hpp>

using namespace cv;

std::vector<String> split(const String &s, char delim)
{
	std::vector<String> elems;
	split(s, delim, elems);
	return elems;
}

std::vector<String>& split(const String &s, char delim, Strings &elems)
{
	std::stringstream ss(s);
	String item;

	while (getline(ss, item, delim))
	{
		elems.push_back(item);
	}

	return elems;
}

bool replace(String& str, const String& from, const String& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == String::npos) return false;
    str.replace(start_pos, from.length(), to);

    return true;
}

String toString(int i)
{
	std::stringstream ss;
	ss << i;

	return ss.str();
}

String toString(const Mat& X)
{
	Stringstream ss;
    ss << "[";
    for (int i = 0; i < X.rows; i++)
    {
        for (int j = 0; j < X.cols; j++)
        {
			ss << setw(2) << setprecision(5) << X.at<double>(i,j) << (j < X.cols - 1 ? ", " : "; ");
        }
    }
    ss << "]";

	return ss.str();
}

String toString(const Colour& colour)
{
	std::vector<size_t> x;
	x.push_back(colour.Red);
	x.push_back(colour.Green);
	x.push_back(colour.Blue);
	Stringstream ss; ss << '#';
	for(std::vector<size_t>::const_iterator i = x.begin(); i != x.end(); i++) ss << uppercase << hex << setfill('0') << setw(2) << *i;

	return ss.str();
}

bool fileExists(const String& path)
{
	return boost::filesystem::exists(path);
}

/*
String fullPath(String path, String filename)
{
	boost::filesystem::path p(path);
	p /= filename;

	return p.String();
}
*/
bool mkdir(const String& path)
{
	return boost::filesystem::create_directories(path);
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

double rmse(const Mat& e)
{
	assert(e.rows > 0 && e.cols == 1); // e must be a column std::vector

	Mat ete = e.t() * e;
	double e2 = ete.at<double>(0,0);
	return sqrt(e2 / e.rows);
}

void drawLabel(Mat& im, const String& label, const Point& pt, int fontFace, double fontScale, const Scalar& frontColour, const Scalar& backColour)
{
    int baseline = 0;

    Size text = getTextSize(label, fontFace, fontScale, 1.0, &baseline);
    rectangle(im, pt + Point(0, baseline), pt + Point(text.width, -text.height), backColour, FILLED);
    putText(im, label, pt, fontFace, fontScale, frontColour, 1, 8);
}

void drawEpipolarLines(Mat& im, size_t lines, const Scalar& colour)
{
	int step = (int)((double)im.rows / (double)lines);
	for (int y = 0; y < im.rows; y+= step)
	{
		line(im, Point(0,y), Point(im.cols,y), colour);
	}
}

void drawCrosshair(Mat& im, const Point2f& pt, int markerSize, const Scalar& colour)
{
	line(im, Point2f(pt.x - markerSize, pt.y), Point2f(pt.x + markerSize, pt.y), colour, 1);
	line(im, Point2f(pt.x, pt.y - markerSize), Point2f(pt.x, pt.y + markerSize), colour, 1);
}

void drawCrosshair(Mat& im, const std::vector<Point2f>& pts, int markerSize, const Scalar& colour)
{
	for (std::vector<Point2f>::const_iterator pt = pts.begin(); pt != pts.end(); pt++)
	{
		drawCrosshair(im, *pt, markerSize, colour);
	}
}

void drawLineStripes(Mat& im, const std::vector<Point2f>& pts, const Scalar& colour, bool closed)
{
	for (size_t i = 0; i < pts.size(); i++)
	{
		bool lastPts = !(i < pts.size() - 1);

		if (lastPts && !closed)
		{
			return;
		}

		Point2f p1 = pts[i];
		Point2f p2 = !lastPts ? pts[i+1] : pts[0];
		line(im, p1, p2, colour);
	}
}

std::vector<Colour> makeHsvColourMap(size_t n)
{
	std::vector<Colour> colours(n);
	Mat hsv(n, 1, CV_32FC3), rgb;

	for (size_t i = 0; i < n; i++)
	{
		hsv.at<Vec3f>(i,0) = Vec3f(360.f * i / n, 1.0f, 1.0f); // Vec3f(H,S,V)
	}

	cvtColor(hsv, rgb, COLOR_HSV2RGB);

	for (size_t i = 0; i < n; i++)
	{
		Vec3f c = rgb.at<Vec3f>(i,0);
		colours[i] = Colour(c(0), c(1), c(2));
	}

	return colours;
}

String now()
{
	time_t t;
	time(&t);
	char buff[80];
	struct tm* timeinfo = localtime (&t);
	strftime (buff, 80, "%c",timeinfo);

	return String(buff);
}

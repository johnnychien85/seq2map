#include <seq2map/common.hpp>

using namespace std;
using namespace cv;
using namespace seq2map;

bool parseArgs(int, char*[], string&, string&, string&);
bool readParams(const string&, Mat&, Mat&);
bool checkPaths(const string&, const string&);

int main(int argc, char* argv[])
{
	string params, in, out;
	Mat map1, map2;

	if (!parseArgs(argc, argv, params, in, out) ||
		!readParams(params, map1, map2) ||
		!checkPaths(in, out)) return -1;

	try
	{
		namespace fs = boost::filesystem;
		fs::path inDir(in), outDir(out);
		fs::directory_iterator endItr;

		for (fs::directory_iterator itr(inDir) ; itr != endItr ; itr++)
		{
			if (!fs::is_regular_file(itr->status())) continue;
			fs::path srcPath(itr->path());
			fs::path dstPath(outDir / srcPath.filename());

			Mat im = imread(srcPath.string(), -1);
			Mat rect;

			if (im.empty()) continue;

			E_INFO << srcPath.string() << " -> " << dstPath.string();

			remap(im, rect, map1, map2, INTER_CUBIC, BORDER_CONSTANT, Scalar());
			imwrite(dstPath.string(), rect);
		}
	}
	catch (exception& ex)
	{
		E_ERROR << "error rectifying image";
        E_ERROR << ex.what();

		return EXIT_FAILURE;
	}

	return 0;
}

bool parseArgs(int argc, char* argv[], string& params, string& in, string& out)
{
	if (argc != 4)
	{
		cout << "Usage: " << argv[0] << " <params.{xml|yml}> <source_dir> <dest_dir>" << endl;
		return false;
	}

	params = string(argv[1]);
	in = string(argv[2]);
	out = string(argv[3]);

	return true;
}

bool readParams(const string& path, Mat& map1, Mat& map2)
{
	FileStorage fs(path, FileStorage::READ);
	Mat K, D, R_rect, P_rect;
	Size S_rect;

	if (!fs.isOpened())
	{
		cerr << "Error reading profile " << path << endl;
		return false;
	}

	try
	{
        fs["K"] >> K;
        fs["D"] >> D;
		fs["S_rect"] >> S_rect;
		fs["R_rect"] >> R_rect;
		fs["P_rect"] >> P_rect;

		// sanity check
		if (S_rect.area() == 0 || K.empty() || D.empty() || R_rect.empty() || P_rect.empty())
		{
			E_ERROR << "some of parameters including K, D, R_rect, P_rect and S_rect are missing";
			return false;
		}
	}
	catch (std::exception& ex)
	{
		E_ERROR << "error parsing camera parameters from " << path;
		E_ERROR << ex.what();

		return false;
	}

	initUndistortRectifyMap(K, D, R_rect, P_rect, S_rect, CV_16SC2, map1, map2);

	return true;
}

bool checkPaths(const string& in, const string& out)
{
	namespace fs = boost::filesystem;
	fs::path inDir(in), outDir(out);

	bool inDirOkay = fs::exists(inDir) && fs::is_directory(inDir);
	bool outDirOkay = fs::exists(outDir) && fs::is_directory(outDir);

	if (inDirOkay && !outDirOkay)
	{
		outDirOkay = fs::create_directory(outDir);
		E_INFO << outDir.string() << " created" << endl;
	}

	return inDirOkay && outDirOkay;
}

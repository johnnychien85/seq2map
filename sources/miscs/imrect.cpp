#include <opencv2/opencv.hpp>
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

			cout << srcPath.string() << " -> " << dstPath.string() << endl;

			remap(im, rect, map1, map2, INTER_CUBIC, BORDER_CONSTANT, Scalar());
			imwrite(dstPath.string(), rect);
		}
	}
	catch (exception& ex)
	{
		cerr << "Error rectifying image:" << endl;
		cerr << ex.what() << endl;
		return -1;
	}

	return 0;
}

bool parseArgs(int argc, char* argv[], string& params, string& in, string& out)
{
	if (argc != 4)
	{
		cout << "Usage: " << argv[0] << " <calib_params.xml> <source_dir> <dest_dir>" << endl;
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
	Mat K, D, R, P;
	Size imageSize;

	if (!fs.isOpened())
	{
		cerr << "Error reading profile " << path << endl;
		return false;
	}

	try
	{
		fs["imageSize"] >> imageSize;
		fs["K"] >> K;
		fs["D"] >> D;
		fs["R"] >> R;
		fs["P"] >> P;

		// sanity check
		if (imageSize.area() == 0 || K.empty() || D.empty() || R.empty() || P.empty())
		{
			throw new exception("One of more parameters including imageSize, K, D, R, and P are missing");
		}

		//if (K.rows != 3 && K.cols != 3 && K.type != CV_32F) {
		//}

	}
	catch (exception& ex)
	{
		cerr << "Error parsing profile " << path << ":" << endl;
		cerr << ex.what() << endl;
		return false;
	}

	initUndistortRectifyMap(K, D, R, P, imageSize, CV_16SC2, map1, map2);

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
		cout << outDir.string() << " created" << endl;
	}

	return inDirOkay && outDirOkay;
}
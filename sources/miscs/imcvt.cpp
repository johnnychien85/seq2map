#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <common.hpp>

using namespace std;
using namespace cv;
using namespace seq2map;

bool parseArgs(int, char*[], string&, string&, string&, string&, int&, int&, bool&, bool&, bool&, double&);
bool checkPaths(const string&, const string&);
bool convertImage(Mat& im, int ibbp, int obbp, bool grey);

int main(int argc, char* argv[])
{
	string in, out, pattern, ext;
	int ibpp, obpp;
	bool flipX, flipY, greyscale;
	double ratio;

	initLogFile();
	if (!parseArgs(argc, argv, in, out, pattern, ext, ibpp, obpp, flipX, flipY, greyscale, ratio)) return -1;

	// make the flip code for cv::flip
	int flipCode = (flipX && flipY) ? -1 : (flipX ? 0 : 1);
	bool flipping = flipX || flipY;

	//...
	bool resampling = ratio > 0.0f && ratio != 1.0f;


	if (!checkPaths(in, out))
	{
		cerr << "error checking directories" << endl;
		return -1;
	}

	try
	{
		Path  srcDir(in), dstDir(out);
		Paths srcFiles = enumerateFiles(srcDir);

		int i = 0;

		BOOST_FOREACH(Path srcPath, srcFiles)
		{
			Path dstPath(dstDir / srcPath.filename());
			Mat im = imread(srcPath.string(), -1); // read an image

			if (im.empty())
			{
				E_INFO << "skipped unreadable image file " << srcPath.string();
				continue;
			}

			// file name rewriting
			if (!pattern.empty())
			{
				char buff[128];
				sprintf(buff, pattern.c_str(), i);
				string rename = string(buff) + srcPath.extension().string();
				dstPath.remove_leaf() /= rename;
			}

			// file extension rewriting
			if (!ext.empty())
			{
				dstPath.replace_extension(ext);
			}

			// do depth & colour conversion first
			if (!convertImage(im, ibpp, obpp, greyscale))
			{
				E_ERROR << "error converting " << srcPath.string();
				continue;
			}

			// down-sampling / up-sampling
			if (resampling)
			{
				resize(im, im, Size(), ratio, ratio);
			}

			// flipping
			if (flipping)
			{
				flip(im, im, flipCode);
			}

			imwrite(dstPath.string(), im);
			i++;

			E_INFO << srcPath.string() << " -> " << dstPath.string();
		}
	}
	catch (exception ex)
	{
		E_FATAL << ex.what() << endl;
		return -1;
	}

	return 0;
}

bool parseArgs(int argc, char* argv[], string& in, string& out, string& pattern, string& ext, int& ibpp, int& obpp, bool& flipX, bool& flipY, bool& grey, double& ratio)
{
	namespace po = boost::program_options;
	po::options_description o("options");
	o.add_options()
		("help,h",	"Show this help message and exit.")
		("format,f",po::value<string>(&pattern)->default_value(""), "C format string (e.g. \"%08d\") of output file name pattern that will be passed to sprintf directly with an integer index as the only argument.")
		("ext,e",	po::value<string>(&ext)->default_value(""),		"Extension of output image file. By default the input image extensions are used.")
		("ibpp",	po::value<int>(&ibpp)->default_value(-1),		"The colour depth of input sequence, specified to rescale images to the desired output depth. The value will be determined as either 8 or 16 automatically if not set.")
		("obpp",	po::value<int>(&obpp)->default_value(-1),		"The colour depth of ouput sequence. Default is to use input colour depth.")
		("ratio,r",	po::value<double>(&ratio)->default_value(1.0f),	"Positive sampling ratio. Default sets 1.0 to disable down-sampling.")
		("flipX",	po::value<bool>(&flipX)->default_value(false)->zero_tokens(),	"Flip image vertically.")
		("flipY",	po::value<bool>(&flipY)->default_value(false)->zero_tokens(),	"Flip image horizontally.")
		("grey,g",	po::value<bool>(&grey)->default_value(false)->zero_tokens(),	"Convert to greyscale image.");

	po::options_description h("hiddens");
	h.add_options()
		("in",		po::value<string>(&in),							"input sequence folder")
		("out",		po::value<string>(&out),						"output sequence folder");

	// args "in" and "out" are positional
	po::positional_options_description p;
	p.add("in",1).add("out",1);

	// all options
	bool okay = true;
	po::options_description a("all");
	a.add(o).add(h);

	po::variables_map vm;
	try
	{
		po::store(po::command_line_parser(argc,argv).options(a).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help"))
		{
			okay = false;
		}
		else
		{
			if (in.empty())
			{
				cerr << "<source_dir> is missing" << endl;
				okay &= false;
			}

			if (out.empty())
			{
				cerr << "<dest_dir> is missing" << endl;
				okay &= false;
			}
			
			if (ibpp > 16)
			{
				cerr << "ibpp=" << ibpp << " higher than 16-bit not supported" << endl;
				okay &= false;
			}

			if (obpp > 0 && obpp != 8 && obpp != 16)
			{
				cerr << "obpp=" << obpp << " must be either 8-bit or 16-bit" << endl;
				okay &= false;
			}

			if (ratio <= 0 || ratio > 1)
			{
				cerr << "ratio=" << ratio << " is out of range (0,1]" << endl;
				okay &= false;
			}
		}
	}
	catch (po::error& pe)
	{
		cerr << "Error parsing arguments: " << pe.what() << endl;
		okay = false;
	}

	if (!okay)
	{
		cout << "Usage: " << argv[0] << " <source_dir> <dest_dir> [options]" << endl;
		cout << o << endl;
	}

	return okay;
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

bool convertImage(Mat& im, int ibpp, int obpp, bool grey)
{
	int bpp;
	switch (im.depth())
	{
	case CV_8U: bpp = 8; break;
	case CV_16U: bpp = 16; break;
	case CV_32S: bpp = 32; break;
	default:
		cerr << "Unsupported depth " << im.depth() << endl;
		return false;
		break;
	}

	int ddepth = (obpp > 0) ? obpp : bpp;
	int dtype = (ddepth == 16) ? CV_16U : CV_8U;

	if (ibpp > bpp)
	{
		cerr << "Depth rescaling not possible because specified bbp " << ibpp << " > image bbp " << bpp << "!" << endl;
		return false;
	}

	// convert scale
	double alpha = pow(2, (double)(ddepth - (ibpp > 0 ? ibpp : bpp)));
	im.convertTo(im, dtype, alpha);

	// convert color as well, if required
	if (grey && im.channels() == 3)
	{
		cvtColor(im, im, CV_BGR2GRAY);
	}

	return true;
}
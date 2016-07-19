#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <seq2map/disparity.hpp>

using namespace std;
using namespace cv;
using namespace seq2map;

struct Args
{
	Path leftPath, rightPath, outPath, paramPath;
	string disparityExt;
};

bool parseArgs(int, char*[], Args&, StereoMatcherAdaptor**);
bool checkPaths(const Path&, const Path&, const Path&);

int main(int argc, char* argv[])
{
	Args args;
    StereoMatcherAdaptor* matcher = NULL;

	if (!parseArgs(argc, argv, args, &matcher)) return -1;
	if (!checkPaths(args.leftPath, args.rightPath, args.outPath)) return -1;

	initLogFile(args.outPath / "disparity.log");

	try
	{
		FileStorage f(args.paramPath.string(), FileStorage::WRITE);
		matcher->WriteParams(f);
		f.release();

		Paths leftFiles = enumerateFiles(args.leftPath, string());
		uint16_t normFactor = matcher->GetNormalisationFactor();
		int i = 0;

		BOOST_FOREACH(const Path& leftFile, leftFiles)
		{
			Path rightFile(args.rightPath / leftFile.filename());
			Path outFile(args.outPath / leftFile.filename());

			outFile.replace_extension(args.disparityExt);

			// read an image
			Mat imLeft = imread(leftFile.string(), IMREAD_GRAYSCALE);
			Mat imRight = imread(rightFile.string(), IMREAD_GRAYSCALE);

			if (imLeft.empty())
			{
				E_INFO << "Skipped unreadable file " << leftFile.string();
				continue;
			}

			if (imRight.empty())
			{
				E_INFO << "Skipped unreadable file " << rightFile.string();
				continue;
			}

			Mat D = matcher->Match(imLeft, imRight);
			D.convertTo(D, CV_16U);
			imwrite(outFile.string(), normFactor * D);

			E_INFO << "Processed " << leftFile.string() << " -> " << outFile.string();
		}
	}
	catch (exception& ex)
	{
		E_FATAL << ex.what();
		return -1;
	}

	delete matcher;
	return 0;
}

bool parseArgs(int argc, char* argv[], Args& args, StereoMatcherAdaptor** matcher)
{
	string matcherName, params, left, right, out;
	int blockSize = 0, dmax = 0;

	namespace po = boost::program_options;
	po::options_description o("Options");
	o.add_options()
		("help,h",	"Show this help message and exit.")
		("matcher,m",po::value<string>(&matcherName)->default_value("SGM"), "Stereo matcher. It must be one of \"SGM\", \"BM\".")
		("dmax,d",	po::value<int>(&dmax)->default_value(0),				"Maximum disparity value. In case of SGM ths value must be divisible by 16")
		("block,b",	po::value<int>(&blockSize)->default_value(7),			"Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.")
		("params,p",po::value<string>(&params)->default_value("matcher.yml"),"Parameter file name to be written in the <dest_dir> folder.")
		("ext,e",	po::value<string>(&args.disparityExt)->default_value("png"), "Extension of output disparity file.")
		;

	po::options_description h("hidden");
	h.add_options()
		("left",	po::value<string>(&left),	"input left sequence folder")
		("right",	po::value<string>(&right),	"input right sequence folder")
		("out",		po::value<string>(&out),	"output sequence folder");

	// args "left", "right" and "out" are positional
	po::positional_options_description p;
	p.add("left",1).add("right",1).add("out",1);

	// SGM-specific parameters
	int p1, p2, d12max, pfcap, uratio, spcsize, spcrange;
	bool fulldp;

	po::options_description o_sgm("SGM Options");
	o_sgm.add_options()
		("p1",		po::value<int>(&p1)->default_value(0),			"The first parameter controlling the disparity smoothness.")
		("p2",		po::value<int>(&p2)->default_value(0),			"The second parameter controlling the disparity smoothness.")
		("d12max",	po::value<int>(&d12max)->default_value(1),		"Maximum allowed difference (in integer pixel units) in the left-right disparity check.")
		("pfcap",	po::value<int>(&pfcap)->default_value(63),		"Truncation value for the prefiltered image pixels.")
		("uratio",	po::value<int>(&uratio)->default_value(10),		"Margin in percentage by which the best (minimum) computed cost function value should \"win\" the second best value to consider the found match correct.")
		("spcsize",	po::value<int>(&spcsize)->default_value(100),	"Maximum size of smooth disparity regions to consider their noise speckles and invalidate.")
		("spcrange",po::value<int>(&spcrange)->default_value(32),	"Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.")
		("fulldp",	po::value<bool>(&fulldp)->default_value(false),	"Set it to true to run the full-scale two-pass dynamic programming algorithm.")
		;

	// all options
	bool okay = true;
	po::options_description a("all");
	a.add(o).add(o_sgm).add(h);

	po::variables_map vm;
	try
	{
		po::store(po::command_line_parser(argc,argv).options(a).positional(p).run(), vm);
		po::notify(vm);
	}
	catch (po::error& pe)
	{
		E_ERROR << "Error parsing arguments: " << pe.what();
		okay = false;
	}

	if (okay && vm.count("help"))
	{
		okay = false;
	}

	if (okay)
	{
		if (left.empty())
		{
			E_ERROR << "<left_dir> is missing";
			okay = false;
		}
		else
		{
			args.leftPath = left;
		}

		if (right.empty())
		{
			E_ERROR << "<right_dir> is missing";
			okay = false;
		}
		else
		{
			args.rightPath = right;
		}

		if (out.empty())
		{
			E_ERROR << "<dest_dir> is missing";
			okay = false;
		}
		else
		{
			args.outPath = out;
		}

		//
		// Matcher factory...
		//
		if (boost::iequals(matcherName, "SGM"))
		{
			if (dmax == 0 || (dmax % 16) != 0)
			{
				E_ERROR << "dmax=" << dmax << " must be a nonzero positive integer divisible by 16";
				okay = false;
			}
				
			if (p1 == 0) p1 = 8 * blockSize * blockSize;
			if (p2 == 0) p2 = 32 * blockSize * blockSize;

			if (okay)
			{
				*matcher = new SemiGlobalMatcher(dmax, blockSize, p1, p2, d12max, pfcap, uratio, spcsize, spcrange, fulldp);
			}
		}
		else if (boost::iequals(matcherName, "BM"))
		{
			if (dmax == 0)
			{
				E_ERROR << "dmax=" << dmax << " must be a nonzero positive integer";
				okay = false;
			}

			if (okay)
			{
				*matcher = new BlockMatcher(dmax, blockSize);
			}
		}
		else
		{
			E_ERROR << "unknown matcher=" << matcherName;
			okay = false;
		}
	}

	if (!okay)
	{
		cout << "Usage: " << argv[0] << " <left_dir> <right_dir> <dest_dir> [options]" << endl;
		cout << o << endl;
		cout << o_sgm << endl;

		return false;
	}

	args.paramPath = args.outPath / params;
	
	return true;
}

bool checkPaths(const Path& leftPath, const Path& rightPath, const Path& outPath)
{
	bool okay = true;

	if (!dirExists(leftPath))
	{
		E_ERROR << "<left_dir> not found: " << leftPath.string();
		okay = false;
	}

	if (!dirExists(rightPath))
	{
		E_ERROR << "<right_dir> not found: " << rightPath.string();
		okay = false;
	}

	if (okay && !makeOutDir(outPath))
	{
		E_ERROR << "error creating output directory: " << outPath.string();
		okay = false;
	}

	return okay;
}

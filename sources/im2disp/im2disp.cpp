#include <seq2map/sequence.hpp>

using namespace seq2map;
namespace po = boost::program_options;

class MyApp : public App
{
public:
    MyApp(int argc, char* argv[]) : App(argc, argv) {}

protected:
    virtual void SetOptions(Options&, Options&, Positional&);
    virtual void ShowHelp(const Options&) const;
    virtual bool ProcessUnknownArgs(const Strings& args);
    virtual bool Init();
    virtual bool Execute();

private:
    String m_priPath;
    String m_secPath;
    String m_outPath;
    String m_matcherName;
    String m_geometry;
    String m_extension;
    String m_index;
    int    m_priCamIdx;
    int    m_secCamIdx;
    StereoMatcherFactory::BasePtr m_matcher;
    ImageStore     m_priImageStore;
    ImageStore     m_secImageStore;
    DisparityStore m_dispStore;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Disparity map computation." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << std::endl;
    std::cout << "  " << m_exec.string() << " [options] <primary_image_dir> <secondary_image_dir> <disparity_out_dir>" << std::endl;
    std::cout << "  " << m_exec.string() << " [options] --pri <primary_camera_idx> --sec <secondary_camera_idx> <sequence_in_dir>" << std::endl;
    std::cout << std::endl;
    std::cout << o << std::endl;

    if (m_matcherName.empty())
    {
        std::cout << "Please use -h with -m to list matcher-specific options." << std::endl;
        return;
    }

    StereoMatcherFactory::BasePtr matcher = StereoMatcherFactory::GetInstance().Create(m_matcherName);

    if (matcher)
    {
        std::cout << matcher->GetOptions() << std::endl;
    }
    else
    {
        E_ERROR << "unknown stereo matcher \"" << m_matcherName << "\"";
    }

}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    const StereoMatcherFactory& factory = StereoMatcherFactory::GetInstance();

    String matcherList = makeNameList(factory. GetRegisteredKeys());
    String matcherDesc = "Descriptor extractor name, must be one of " + matcherList;

    // some essential parameters
    o.add_options()
        ("matcher,m", po::value<String>(&m_matcherName)->default_value(         ""), matcherDesc.c_str())
        ("pri",       po::value<int>   (&m_priCamIdx  )->default_value(         -1), "Index of the primary camera for IN-SEQ mode.")
        ("sec",       po::value<int>   (&m_secCamIdx  )->default_value(         -1), "Index of the secondary camera for IN-SEQ mode.")
    //  ("geometry",  po::value<String>(&m_geometry   )->default_value(       "LR"), "Configuration of stereo cameras. This option is automatically determined for IN-SEQ mode.")
        ("ext,e",     po::value<String>(&m_extension  )->default_value(     ".png"), "The extension name of the output disparity files. Must be a valid image extension supported by OpenCV.")
        ("cam,c",     po::value<String>(&m_index      )->default_value("index.yml"), "Select camera from a sequence database to enable IN-SEQ mode.");

    // three positional arguments - input and output directories
    h.add_options()
        ("in0",       po::value<String>(&m_priPath    )->default_value(         ""), "Input folder containing images of the primary camera. Alternatively, this can be set to a sequence database to enable IN-SEQ mode.")
        ("in1",       po::value<String>(&m_secPath    )->default_value(         ""), "Input folder containing images or the seconadry camera. This option is ignored for IN-SEQ mode.")
        ("out",       po::value<String>(&m_outPath    )->default_value(         ""), "Output folder for the computed disparity images. This option is ignored for IN-SEQ mode.");

    p.add("in0", 1).add("in1", 1).add("out", 1);
}

bool MyApp::ProcessUnknownArgs(const Strings& args)
{
    if (m_matcherName.empty())
    {
        E_FATAL << "missing matcher name";
        return false;
    }

    m_matcher = StereoMatcherFactory::GetInstance().Create(m_matcherName);

    if (!m_matcher)
    {
        E_FATAL << "error creating stereo matcher \"" << m_matcherName << "\"";
        return false;
    }

    try
    {
        Parameterised::Options opts = m_matcher->GetOptions();

        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(args).options(opts).run();

        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (std::exception& ex)
    {
        E_FATAL << "error parsing matcher-specific arguments: " << ex.what();
        std::cout << "Try to use -h with -m to see the supported options for specific stereo matcher" << std::endl;

        return false;
    }

    m_matcher->ApplyParams();

    return true;
}

bool MyApp::Init()
{
    bool inSeqMode = m_priCamIdx >= 0 && m_secCamIdx >= 0;
    Sequence seq;

    // set image store for input
    if (inSeqMode)
    {
        size_t priCamIdx = static_cast<size_t>(m_priCamIdx);
        size_t secCamIdx = static_cast<size_t>(m_secCamIdx);
        const Path seqPath = m_priPath;

        E_INFO << "IN-SEQ mode selected, trying to use cameras " << priCamIdx << " (primary) and " << secCamIdx << " (secondary) from " << seqPath;

        if (!seq.Restore(seqPath))
        {
            E_ERROR << "IN-SEQ mode selected but unable to load sequence database from " << seqPath;
            return false;
        }

        bool found = false;

        BOOST_FOREACH (const RectifiedStereo& pair, seq.GetRectifiedStereo())
        {
            found = pair.GetPrimaryCamera()  ->GetIndex() == priCamIdx &&
                    pair.GetSecondaryCamera()->GetIndex() == secCamIdx;

            if (found)
            {
                m_priImageStore = pair.GetPrimaryCamera()  ->GetImageStore();
                m_secImageStore = pair.GetSecondaryCamera()->GetImageStore();

                break;
            }
        }

        if (!found)
        {
            E_ERROR << "cannot find stereo pair (" << m_priCamIdx << "," << m_secCamIdx << ")";
            return false;
        }
    }
    else
    {
        E_INFO << "listing files in " << m_priPath << "..";
        m_priImageStore.FromExistingFiles(m_priPath);
        E_INFO << "found " << m_priImageStore.GetItems() << " item(s)";

        E_INFO << "listing files in " << m_secPath << "..";
        m_secImageStore.FromExistingFiles(m_secPath);
        E_INFO << "found " << m_secImageStore.GetItems() << " item(s)";

        if (m_priImageStore.GetItems() != m_secImageStore.GetItems())
        {
            E_ERROR << "the numbers of items do not match";
            return false;
        }
    }

    E_INFO << m_priImageStore.GetItems() << " items will be processed";

    // set disparity store for output
    if (inSeqMode)
    {
        Path dispStoreRoot = seq.GetDisparityStoreRoot();
        m_outPath = "";

        for (size_t i = 0; i < 65535; i++)
        {
            std::stringstream ss;
            ss << std::setw(8) << std::setfill('0') << i;

            const Path newStorePath = dispStoreRoot / ss.str();

            if (!dirExists(newStorePath))
            {
                m_outPath = newStorePath.string();
                break;
            }
        }

        if (m_outPath.empty())
        {
            E_ERROR << "cannot find an available name for folder creation in " << dispStoreRoot;
            return false;
        }
    }
    else if (m_outPath.empty())
    {
        E_ERROR << "missing output folder";
        return false;
    }

    if (!inSeqMode)
    {
        m_priCamIdx = 0;
        m_secCamIdx = 0;
    }

    Path outPath = Path(m_outPath);
    bool success = m_dispStore.Create(m_outPath, m_priCamIdx, m_secCamIdx, m_matcher);

    if (!success)
    {
        E_ERROR << "error creating disparity store " << outPath;
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    try
    {
        Speedometre metre("Disparity", "px/s");
        size_t frames = 0, bytes = 0;
        size_t files = m_priImageStore.GetItems(); // == m_secImageStore.GetItems()

        E_INFO << "feature extraction procedure starts for " << files << " file(s)";
        E_INFO << "primary source:   " << fullpath(m_priImageStore.GetRoot());
        E_INFO << "secondary source: " << fullpath(m_secImageStore.GetRoot());
        E_INFO << "output folder:    " << fullpath(m_dispStore.GetRoot());

        for (size_t i = 0; i < files; i++)
        {
            Path pri(m_priImageStore.GetItemPath(i));
            Path sec(m_secImageStore.GetItemPath(i));
            Path dst(m_dispStore.GetRoot() / m_priImageStore.GetFileNames()[i]);

            dst.replace_extension(m_extension);

            cv::Mat im0 = m_priImageStore[i].im;
            cv::Mat im1 = m_secImageStore[i].im;

            PersistentImage dp;

            if (im0.empty())
            {
                E_INFO << "skipped unreadable primary file " << pri;
                continue;
            }

            if (im1.empty())
            {
                E_INFO << "skipped unreadable secondary file " << sec;
                continue;
            }

            im0 = im0.channels() == 3 ? rgb2gray(im0) : im0;
            im1 = im1.channels() == 3 ? rgb2gray(im1) : im1;

            metre.Start();
            dp.im = m_matcher->Match(im0, im1);
            metre.Stop(dp.im.total());

            if (dp.im.empty())
            {
                E_ERROR << "error computing disparity map";
                continue;
            }

            if (!m_dispStore.Append(dst.filename().string(), dp))
            {
                E_FATAL << "error writing disparity map to " << dst;
                return false;
            }

            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << metre.GetSpeed() << " " << metre.GetUnit();

            frames++;
            bytes += filesize(dst);

            E_INFO << "processed " << pri.filename() << " -> " << dst.filename() << " [" << ss.str() << "]";
        }

        if (!m_index.empty())
        {
            Path to = m_dispStore.GetRoot() / Path(m_index);
            cv::FileStorage fs(to.string(), cv::FileStorage::WRITE);

            if (!m_dispStore.Store(fs))
            {
                E_ERROR << "error storing index to " << to;
                return false;
            }

            E_INFO << "index written to " << to;
        }

        E_INFO << "disparity map(s) from " << frames << " frame(s) computed and stored";
        E_INFO << "file storage:     " << (bytes / 1024.0f / 1024.0f) << " MBytes";
        E_INFO << "computation time: " << metre.GetElapsedSeconds() << " secs";
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop: " << ex.what();
        return false;
    }

    return true;
}
/*
class OpA : public ChainedOp<int>
{
public:
    OpA(int k) : k(k) {}
    virtual int operator() (int& x) { return k * x; }
    int k;

protected:
    virtual Ptr Create() const { return Ptr(new OpA(k)); }
};

class OpB : public ChainedOp<int>
{
public:
    OpB(int a, int b) : a(a), b(b) {}
    virtual int operator() (int& x) { return a * x + b; }
    int a;
    int b;

protected:
    virtual Ptr Create() const { return Ptr(new OpB(a, b)); }
};

class OpC : public ChainedOp<int>
{
public:
    OpC(int a, int b) : a(a), b(b) {}
    virtual int operator() (int& x) { return (x - b) / a; }
    int a;
    int b;

protected:
    virtual Ptr Create() const { return Ptr(new OpC(a, b)); }
};
*/
int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}
//
//struct Args
//{
//	Path leftPath, rightPath, outPath, paramPath;
//	string disparityExt;
//};
//
//bool parseArgs(int, char*[], Args&, StereoMatcherAdaptor**);
//bool checkPaths(const Path&, const Path&, const Path&);
//
//int main(int argc, char* argv[])
//{
//	Args args;
//    StereoMatcherAdaptor* matcher = NULL;
//
//	if (!parseArgs(argc, argv, args, &matcher)) return -1;
//	if (!checkPaths(args.leftPath, args.rightPath, args.outPath)) return -1;
//
//	initLogFile(args.outPath / "disparity.log");
//
//	try
//	{
//		FileStorage f(args.paramPath.string(), FileStorage::WRITE);
//		matcher->WriteParams(f);
//		f.release();
//
//		Paths leftFiles = enumerateFiles(args.leftPath, string());
//		uint16_t normFactor = matcher->GetNormalisationFactor();
//		int i = 0;
//
//		BOOST_FOREACH(const Path& leftFile, leftFiles)
//		{
//			Path rightFile(args.rightPath / leftFile.filename());
//			Path outFile(args.outPath / leftFile.filename());
//
//			outFile.replace_extension(args.disparityExt);
//
//			// read an image
//			Mat imLeft = imread(leftFile.string(), IMREAD_GRAYSCALE);
//			Mat imRight = imread(rightFile.string(), IMREAD_GRAYSCALE);
//
//			if (imLeft.empty())
//			{
//				E_INFO << "Skipped unreadable file " << leftFile.string();
//				continue;
//			}
//
//			if (imRight.empty())
//			{
//				E_INFO << "Skipped unreadable file " << rightFile.string();
//				continue;
//			}
//
//			Mat D = matcher->Match(imLeft, imRight);
//			D.convertTo(D, CV_16U);
//			imwrite(outFile.string(), normFactor * D);
//
//			E_INFO << "Processed " << leftFile.string() << " -> " << outFile.string();
//		}
//	}
//	catch (exception& ex)
//	{
//		E_FATAL << ex.what();
//		return -1;
//	}
//
//	delete matcher;
//	return 0;
//}
//
//bool parseArgs(int argc, char* argv[], Args& args, StereoMatcherAdaptor** matcher)
//{
//	string matcherName, params, left, right, out;
//	int blockSize = 0, dmax = 0;
//
//	namespace po = boost::program_options;
//	po::options_description o("Options");
//	o.add_options()
//		("help,h",	"Show this help message and exit.")
//		("matcher,m",po::value<string>(&matcherName)->default_value("SGM"), "Stereo matcher. It must be one of \"SGM\", \"BM\".")
//		("dmax,d",	po::value<int>(&dmax)->default_value(0),				"Maximum disparity value. In case of SGM ths value must be divisible by 16")
//		("block,b",	po::value<int>(&blockSize)->default_value(7),			"Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.")
//		("params,p",po::value<string>(&params)->default_value("matcher.yml"),"Parameter file name to be written in the <dest_dir> folder.")
//		("ext,e",	po::value<string>(&args.disparityExt)->default_value("png"), "Extension of output disparity file.")
//		;
//
//	po::options_description h("hidden");
//	h.add_options()
//		("left",	po::value<string>(&left),	"input left sequence folder")
//		("right",	po::value<string>(&right),	"input right sequence folder")
//		("out",		po::value<string>(&out),	"output sequence folder");
//
//	// args "left", "right" and "out" are positional
//	po::positional_options_description p;
//	p.add("left",1).add("right",1).add("out",1);
//
//	// SGM-specific parameters
//	int p1, p2, d12max, pfcap, uratio, spcsize, spcrange;
//	bool fulldp;
//
//	po::options_description o_sgm("SGM Options");
//	o_sgm.add_options()
//		("p1",		po::value<int>(&p1)->default_value(0),			"The first parameter controlling the disparity smoothness.")
//		("p2",		po::value<int>(&p2)->default_value(0),			"The second parameter controlling the disparity smoothness.")
//		("d12max",	po::value<int>(&d12max)->default_value(1),		"Maximum allowed difference (in integer pixel units) in the left-right disparity check.")
//		("pfcap",	po::value<int>(&pfcap)->default_value(63),		"Truncation value for the prefiltered image pixels.")
//		("uratio",	po::value<int>(&uratio)->default_value(10),		"Margin in percentage by which the best (minimum) computed cost function value should \"win\" the second best value to consider the found match correct.")
//		("spcsize",	po::value<int>(&spcsize)->default_value(100),	"Maximum size of smooth disparity regions to consider their noise speckles and invalidate.")
//		("spcrange",po::value<int>(&spcrange)->default_value(32),	"Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.")
//		("fulldp",	po::value<bool>(&fulldp)->default_value(false),	"Set it to true to run the full-scale two-pass dynamic programming algorithm.")
//		;
//
//	// all options
//	bool okay = true;
//	po::options_description a("all");
//	a.add(o).add(o_sgm).add(h);
//
//	po::variables_map vm;
//	try
//	{
//		po::store(po::command_line_parser(argc,argv).options(a).positional(p).run(), vm);
//		po::notify(vm);
//	}
//	catch (po::error& pe)
//	{
//		E_ERROR << "Error parsing arguments: " << pe.what();
//		okay = false;
//	}
//
//	if (okay && vm.count("help"))
//	{
//		okay = false;
//	}
//
//	if (okay)
//	{
//		if (left.empty())
//		{
//			E_ERROR << "<left_dir> is missing";
//			okay = false;
//		}
//		else
//		{
//			args.leftPath = left;
//		}
//
//		if (right.empty())
//		{
//			E_ERROR << "<right_dir> is missing";
//			okay = false;
//		}
//		else
//		{
//			args.rightPath = right;
//		}
//
//		if (out.empty())
//		{
//			E_ERROR << "<dest_dir> is missing";
//			okay = false;
//		}
//		else
//		{
//			args.outPath = out;
//		}
//
//		//
//		// Matcher factory...
//		//
//		if (boost::iequals(matcherName, "SGM"))
//		{
//			if (dmax == 0 || (dmax % 16) != 0)
//			{
//				E_ERROR << "dmax=" << dmax << " must be a nonzero positive integer divisible by 16";
//				okay = false;
//			}
//				
//			if (p1 == 0) p1 = 8 * blockSize * blockSize;
//			if (p2 == 0) p2 = 32 * blockSize * blockSize;
//
//			if (okay)
//			{
//				*matcher = new SemiGlobalMatcher(dmax, blockSize, p1, p2, d12max, pfcap, uratio, spcsize, spcrange, fulldp);
//			}
//		}
//		else if (boost::iequals(matcherName, "BM"))
//		{
//			if (dmax == 0)
//			{
//				E_ERROR << "dmax=" << dmax << " must be a nonzero positive integer";
//				okay = false;
//			}
//
//			if (okay)
//			{
//				*matcher = new BlockMatcher(dmax, blockSize);
//			}
//		}
//		else
//		{
//			E_ERROR << "unknown matcher=" << matcherName;
//			okay = false;
//		}
//	}
//
//	if (!okay)
//	{
//		cout << "Usage: " << argv[0] << " <left_dir> <right_dir> <dest_dir> [options]" << endl;
//		cout << o << endl;
//		cout << o_sgm << endl;
//
//		return false;
//	}
//
//	args.paramPath = args.outPath / params;
//	
//	return true;
//}
//
//bool checkPaths(const Path& leftPath, const Path& rightPath, const Path& outPath)
//{
//	bool okay = true;
//
//	if (!dirExists(leftPath))
//	{
//		E_ERROR << "<left_dir> not found: " << leftPath.string();
//		okay = false;
//	}
//
//	if (!dirExists(rightPath))
//	{
//		E_ERROR << "<right_dir> not found: " << rightPath.string();
//		okay = false;
//	}
//
//	if (okay && !makeOutDir(outPath))
//	{
//		E_ERROR << "error creating output directory: " << outPath.string();
//		okay = false;
//	}
//
//	return okay;
//}

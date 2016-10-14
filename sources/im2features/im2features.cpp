#include <iomanip>
#include <seq2map/features.hpp>

using namespace seq2map;
namespace po = boost::program_options;

struct Args
{
    Path srcPath;
    Path dstPath;
    Path dxtorPath;
    String dstExt;
    String detectorName;
    String xtractorName;
    bool help;
};

bool init(int, char*[], Args&, FeatureDetextractorPtr&);
void showSynopsis(char*);

int main(int argc, char* argv[])
{
    Args args;
    FeatureDetextractorPtr dxtor;

    if (!init(argc, argv, args, dxtor)) return args.help ? EXIT_SUCCESS : EXIT_FAILURE;

    try
    {
        Speedometre metre("Image Feature Detection & Extraction", "px/s");

        if (!makeOutDir(args.dstPath))
        {
            E_FATAL << "error creating output directory";
            return -1;
        }

        Paths files = enumerateFiles(args.srcPath);
        size_t frames = 0, features = 0, bytes = 0;

        E_INFO << "feature extraction procedure starts for " << files.size() << " file(s)";
        E_INFO << "source folder set to " << fullpath(args.srcPath);
        E_INFO << "output folder set to " << fullpath(args.dstPath);

        if (!args.dxtorPath.empty())
        {
            cv::FileStorage fs(args.dxtorPath.string(), cv::FileStorage::WRITE);
            if (!dxtor->Store(fs))
            {
                E_ERROR << "error storing parameter to " << args.dxtorPath;
                return -1;
            }

            E_INFO << "parameters stored to " << fullpath(args.dxtorPath);
        }

        BOOST_FOREACH(const Path& file, files)
        {
            Path dstFile(args.dstPath / file.filename());
            dstFile.replace_extension(args.dstExt);

            // read an image
            cv::Mat im = cv::imread(file.string(), cv::IMREAD_GRAYSCALE);

            if (im.empty())
            {
                E_INFO << "skipped unreadable file " << file.string();
                continue;
            }

            metre.Start();
            ImageFeatureSet f = dxtor->DetectAndExtractFeatures(im);
            metre.Stop(im.total());

            if (!f.Store(dstFile))
            {
                E_FATAL << "error writing features to " << dstFile;
                return -1;
            }

            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << metre.GetSpeed() << " " << metre.GetUnit();

            frames++;
            features += f.GetSize();
            bytes += filesize(dstFile);

            E_INFO << "processed " << file.filename() << " -> " << dstFile.filename() << " [" << ss.str() << "]";
        }

        E_INFO << "image feature extraction procedure finished, " << features << " feature(s) detected from " << frames << " frame(s)";
        E_INFO << "file storage: " << (bytes / 1024.0f / 1024.0f) << " MBytes";
        E_INFO << "computation time: " << metre.GetElapsedSeconds() << " secs";
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop: " << ex.what();
        return -1;
    }

    return 0;
}

bool init(int argc, char* argv[], Args& args, FeatureDetextractorPtr& dxtor)
{
    FeatureDetextractorFactory dxtorFactory;
    String srcPath, dstPath, dxtorPath;

    //
    // Prepare messages for argument parsing
    //
    String detectorList = makeNameList(dxtorFactory.GetDetectorFactory().GetRegisteredKeys());
    String xtractorList = makeNameList(dxtorFactory.GetExtractorFactory().GetRegisteredKeys());

    String detectorDesc = "Keypoint detector name, must be one of " + detectorList;
    String xtractorDesc = "Descriptor extractor name, must be one of " + xtractorList;
    xtractorDesc += " When the keypoint detector is not given explicitly, the extractor will be applied first to find image features, if it is capable to do so.";
    xtractorDesc += " In this case, the extractor must be one of " + makeNameList(dxtorFactory.GetRegisteredKeys());

    // some essential parameters
    po::options_description o("General Options");
    o.add_options()
        ("help,h",    "Show this help message and exit. This option can be combined with -k and/or -x to list feature-specific options.")
        ("detect,k",  po::value<String>(&args.detectorName)->default_value(""), detectorDesc.c_str())
        ("extract,x", po::value<String>(&args.xtractorName)->default_value(""), xtractorDesc.c_str())
        ("dxtor",     po::value<String>(&dxtorPath)->default_value("dxtor.yml"), "Path to the feature detector and extractor persistent storage where parameters to be written.")
        ("ext,e",     po::value<String>(&args.dstExt)->default_value(".dat"),    "The extension name of the output feature files.");

    // two positional arguments - input and output directories
    po::options_description h("Hiddens");
    h.add_options()
        ("in",  po::value<String>(&srcPath)->default_value(""), "Input folder containing images")
        ("out", po::value<String>(&dstPath)->default_value(""), "Output folder in where the detected and extracted features are written");

    po::positional_options_description p;
    p.add("in", 1).add("out", 1);

    // do parsing
    Strings unknownToks;
    try
    {
        po::options_description a;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(a.add(o).add(h)).positional(p).allow_unregistered().run();
        po::variables_map vm;

        po::store(parsed, vm);
        po::notify(vm);

        unknownToks = po::collect_unrecognized(parsed.options, po::exclude_positional);
        args.help = vm.count("help") > 0;
    }
    catch (po::error& pe)
    {
        E_FATAL << "error parsing general arguments: " << pe.what();
        showSynopsis(argv[0]);

        return false;
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caugth: " << ex.what();
        return false;
    }

    args.srcPath = srcPath;
    args.dstPath = dstPath;
    args.dxtorPath = dxtorPath;
    args.detectorName = !args.detectorName.empty() ? args.detectorName : args.xtractorName;

    if (args.help)
    {
        if (!args.detectorName.empty())
        {
            FeatureDetectorPtr detector =
                dxtorFactory.GetDetectorFactory().Create(args.detectorName);
            if (!detector)
            {
                E_FATAL << "error creating feature detector for parameter listing";
                std::cout << "Known feature detectors: " << detectorList << std::endl;

                return false;
            }
            o.add(detector->GetOptions(DETECTION_OPTIONS));
        }

        if (!args.xtractorName.empty())
        {
            FeatureExtractorPtr extractor =
                dxtorFactory.GetExtractorFactory().Create(args.xtractorName);
            if (!extractor)
            {
                E_FATAL << "error creating descriptor extractor for parameter listing";
                std::cout << "Known descriptor extractors: " << xtractorList << std::endl;

                return false;
            }
            o.add(extractor->GetOptions(EXTRACTION_OPTIONS));
        }

        std::cout << "Usage: " << argv[0] << " <image_in_dir> <feature_out_dir> [options]" << std::endl;
        std::cout << o << std::endl;

        return false;
    }

    if (srcPath.empty())
    {
        E_FATAL << "input directory path missing";
        showSynopsis(argv[0]);

        return false;
    }

    if (dstPath.empty())
    {
        E_FATAL << "output directory path missing";
        showSynopsis(argv[0]);

        return false;
    }

    if (args.xtractorName.empty())
    {
        E_FATAL << "descriptor extractor name missing";
        showSynopsis(argv[0]);

        return false;
    }

    dxtor = dxtorFactory.Create(args.detectorName, args.xtractorName);

    if (!dxtor)
    {
        E_FATAL << "error creating feature detector-and-extractor object";
        return false;
    }

    // Finally the last mile..
    // parse options for the detector and extractor!
    try
    {
        Parameterised::Options detectorOptions = dxtor->GetOptions(DETECTION_OPTIONS);
        Parameterised::Options xtractorOptions = dxtor->GetOptions(EXTRACTION_OPTIONS);
        Parameterised::Options allOptions;

        allOptions.add(detectorOptions).add(xtractorOptions);

        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(unknownToks).options(allOptions).run();

        // store the parsed values to the detector object's internal data member(s)
        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (std::exception& ex)
    {
        E_FATAL << "error parsing feature-specific arguments: " << ex.what();
        std::cout << "Try to use -h with -k and/or -x to see the supported options for specific feature detectior/extractor" << std::endl;

        return false;
    }

    dxtor->ApplyParams();

    return true;
}

void showSynopsis(char* exec)
{
    std::cout << std::endl;
    std::cout << "Try \"" << exec << " -h\" for usage listing." << std::endl;
}

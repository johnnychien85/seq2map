#include "scanner.hpp"

namespace po = boost::program_options;

struct Args
{
    String scanner;
    String seqNanme;
    Path srcPath;
    Path dstPath;
    Path calPath;
    Path motPath;
    Path featuresPath;
    String dxtorFileName;
    bool help;
};

bool init(int, char*[], const ScannerFactory& factory, Args&);
void scanFeatures(Path path, Sequence& seq);
void showSynopsis(char*);

int main(int argc, char* argv[])
{
    Args args;
    ScannerFactory factory;

    if (!init(argc, argv, factory, args)) return args.help ? EXIT_SUCCESS : EXIT_FAILURE;

    try
    {
        Sequence seq;
        ScannerFactory::BasePtr scanner = factory.Create(args.scanner);

        if (!scanner)
        {
            E_ERROR << "unknown sequence scanner \"" << args.scanner << "\"";
            return -1;
        }

        if (!scanner->Scan(args.srcPath, args.calPath, args.motPath, seq))
        {
            E_ERROR << "error scanning sequence \"" << args.srcPath << "\"";
            return -1;
        }

        seq.SetPath(boost::filesystem::absolute(args.srcPath));
        seq.SetName(args.seqNanme);
        seq.SetGrabber(args.scanner);

        if (!args.featuresPath.empty())
        {
            // try to restore the feature detextractor if its
            //  persistent storage is found
            Path dxtorPath = args.featuresPath / args.dxtorFileName;

            if (fileExists(dxtorPath))
            {
                cv::FileStorage fs(dxtorPath.string(), cv::FileStorage::READ);
                FeatureDetextractorFactory dxtorFactory;
                FeatureDetextractorPtr dxtor = dxtorFactory.Create(fs.root());

                if (dxtor)
                {
                    E_INFO << "the feature detector and extractor succefully restored from " << dxtorPath;
                    seq.SetFeatureDetextractor(dxtor);
                }
            }

            scanFeatures(args.featuresPath, seq);
        }

        if (!seq.Store(args.dstPath))
        {
            E_ERROR << "error writing sequence profile to \"" << args.dstPath << "\"";
            return -1;
        }

        E_INFO << "sequence profile succesfuilly saved to " << args.dstPath;
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop";
        E_FATAL << ex.what();

        return -1;
    }

    return 0;
}

bool init(int argc, char* argv[], const ScannerFactory& factory, Args& args)
{
    String srcPath, calPath, motPath, featuresPath, dstPath;
    String scanners = "Folder scanner name, must be one of " + makeNameList(factory.GetRegisteredKeys());

    po::options_description o("General Options");
    o.add_options()
        ("name,n", po::value<String>(&args.seqNanme), "Name of the sequence.")
        ("type,t", po::value<String>(&args.scanner), scanners.c_str())
        ("cal,c",  po::value<String>(&calPath)->default_value(""), "Path to the calibration file(s).")
        ("mot,m",  po::value<String>(&motPath)->default_value(""), "Optional path to the motion data.")
        ("features,f", po::value<String>(&featuresPath)->default_value(""), "Optional path to the features folder.")
        ("dxtor",  po::value<String>(&args.dxtorFileName)->default_value("dxtor.yml"), "File name of the feature detector-and-extractor parameters. This file has to be located in the feature folder.")
        ("help,h", "Show this help message and exit.");

    po::options_description h("Hiddens");
    h.add_options()
        ("in",  po::value<String>(&srcPath)->default_value(""), "Path to the input sequence")
        ("out", po::value<String>(&dstPath)->default_value(""), "Path to the output profile");

    po::positional_options_description p;
    p.add("in", 1).add("out", 1);

    try
    {
        po::options_description a;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(a.add(o).add(h)).positional(p).run();
        po::variables_map vm;

        po::store(parsed, vm);
        po::notify(vm);

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
    args.calPath = calPath;
    args.motPath = motPath;
    args.dstPath = dstPath;
    args.featuresPath = featuresPath;

    if (args.help)
    {
        std::cout << "Usage: " << argv[0] << " <sequence_dir> <output_profile> [options]" << std::endl;
        std::cout << o << std::endl;

        return false;
    }

    if (srcPath.empty())
    {
        E_FATAL << "input sequence path missing";
        showSynopsis(argv[0]);

        return false;
    }

    if (dstPath.empty())
    {
        E_FATAL << "output sequence profile missing";
        showSynopsis(argv[0]);

        return false;
    }

    if (args.seqNanme.empty())
    {
        args.seqNanme = removeFilenameExt(args.dstPath.filename().string());
    }

    return true;
}

void showSynopsis(char* exec)
{
    std::cout << std::endl;
    std::cout << "Try \"" << exec << " -h\" for usage listing." << std::endl;
}

void scanFeatures(Path path, Sequence& seq)
{
    Cameras& cams = seq.GetCameras();
    
    E_INFO << "scanning image feature files";

    for (size_t i = 0; i < cams.size(); i++)
    {
        Camera& cam = cams.at(i);
        Path featureDirPath = path / cam.GetName();

        if (!dirExists(featureDirPath))
        {
            E_INFO << "skipped because folder \"" << featureDirPath.string() << "\" does not exist";
            continue;
        }

        Paths featureFiles = enumerateFiles(featureDirPath);
        Camera::FeatureStorage& featureStore = cam.GetFeatureStorage();

        featureStore.SetRootPath(featureDirPath);
        featureStore.Allocate(featureFiles.size());

        BOOST_FOREACH(const Path& featureFile, featureFiles)
        {
            featureStore.Add(featureFile.filename().string());
        }

        E_INFO << featureStore.GetSize() << " feature set(s) located for camera " << i;
    }
}
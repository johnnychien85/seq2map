#include "grabber.hpp"
#include "recorder.hpp"
#include "uirenderer.hpp"

using namespace seq2map;

struct Args
{
    size_t bufferSize;
    size_t targetFPS;
    double epsilon;
    String grabber;
    size_t cameras;
    Path   dstPath;
    Path   logPath;
    bool   help;
};

bool parseArgs(int argc, char* argv[], Args& args);
void showSynopsis(char* exec);

int main(int argc, char* argv[])
{
    Args args;

    if (!parseArgs(argc, argv, args))
    {
        return args.help ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    initLogFile(args.logPath);

    // allocate sync buffer
    SyncBuffer::Ptr buffer = SyncBuffer::Ptr(
        new SyncBuffer(args.cameras, 1, args.bufferSize, args.targetFPS, args.epsilon)
    );

    if (!buffer)
    {
        E_ERROR << "error creating synchronisation buffer";
        return EXIT_FAILURE;
    }

    ImageGrabberBuilderFactory factory;
    ImageGrabberBuilder::Ptr builder = factory.Create(args.grabber);
    ImageGrabber::Ptrs grabbers;

    if (!builder)
    {
        E_ERROR << "error creating grabber builder for " << args.grabber;
        return EXIT_FAILURE;
    }

    Strings devices = builder->GetDeviceIdList();
    for (size_t i = 0; i < devices.size(); i++)
    {
        E_INFO << devices[i];
    }

    if (args.cameras > builder->GetDevices())
    {
        E_ERROR << "insufficient resources";
        E_ERROR << "the grabber builder is able to build " << builder->GetDevices() << " grabber(s), however " << args.cameras << " requested";

        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < args.cameras; i++)
    {
        ImageGrabber::Ptr grabber = builder->Build(buffer, i);

        if (!grabber)
        {
            E_ERROR << "error creating grabber " << i;
            return EXIT_FAILURE;
        }

        grabbers.push_back(grabber);
    }

    BufferRecorder recorder(buffer, "seq");

    for (size_t i = 0; i < grabbers.size(); i++)
    {
        if (!grabbers[i]->Start())
        {
            E_ERROR << "error starting grabber " << i;
            return EXIT_FAILURE;
        }
    }

    if (!recorder.Start())
    {
        E_ERROR << "error starting the recorder";
        return EXIT_FAILURE;
    }

    // bring up the main form and loop forever
    MainUI ui(grabbers, recorder, buffer);
    ui.Loop();

    for (size_t i = 0; i < grabbers.size(); i++)
    {
        grabbers[i]->Stop();
    }

    recorder.Stop();

    return EXIT_SUCCESS;
}

bool parseArgs(int argc, char* argv[], Args& args)
{
    ImageGrabberBuilderFactory factory;
    String dstPath, logPath;

    //
    // Prepare messages for argument parsing
    //
    String grabberList = makeNameList(factory.GetRegisteredKeys());
    String grabberDesc = "Image grabber name, must be one of " + grabberList;

    // some essential parameters
    namespace po = boost::program_options;
    po::options_description o("General Options");
    o.add_options()
        ("help,h",    "Show this help message and exit.")
        ("out,o",     po::value<String>(&dstPath)        ->default_value("seq"),   "Path to the folder where the grabbed frames are to be written to.")
        ("grabber,g", po::value<String>(&args.grabber)   ->default_value("DUMMY"), grabberList.c_str())
        ("fps,f",     po::value<size_t>(&args.targetFPS) ->default_value(10),      "Target frame per second")
        ("cam,c",     po::value<size_t>(&args.cameras)   ->default_value(1),       "The number of cammeras to connect to.")
        ("epsilon,e", po::value<double>(&args.epsilon)   ->default_value(0.9),     "The acceptable time difference between the timestamps of the grabbed frame and of the hitting buffer slot.")
        ("buf,b",     po::value<size_t>(&args.bufferSize)->default_value(128),     "The number of slots in the synchronisatio buffer.")
        ("log",       po::value<String>(&logPath)        ->default_value(""),      "Path to the logfile.")
        ;

    po::positional_options_description p;
    p.add("out", 1);

    try
    {
        po::options_description a;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(a.add(o)).positional(p).run();
        po::variables_map vm;

        po::store(parsed, vm);
        po::notify(vm);

        args.help = vm.count("help") > 0;
    }
    catch (po::error& pe)
    {
        E_FATAL << "error parsing arguments: " << pe.what();
        showSynopsis(argv[0]);

        return false;
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caugth: " << ex.what();
        return false;
    }

    args.dstPath = dstPath;
    args.logPath = logPath;

    if (args.help)
    {
        std::cout << "Usage: " << argv[0] << " <out_dir> [options]" << std::endl;
        std::cout << o << std::endl;

        return false;
    }

    if (dstPath.empty())
    {
        E_FATAL << "output directory path missing";
        showSynopsis(argv[0]);

        return false;
    }

    if (args.grabber.empty())
    {
        E_FATAL << "grabber name missing";
        showSynopsis(argv[0]);

        return false;
    }

    return true;
}

void showSynopsis(char* exec)
{
    std::cout << std::endl;
    std::cout << "Try \"" << exec << " -h\" for usage listing." << std::endl;
}

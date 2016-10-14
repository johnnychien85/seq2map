#include "grabber.hpp"
#include "recorder.hpp"
#include "uirenderer.hpp"

#define WINDOW_TITLE    "VGGRABER - 1.0.0"
#define WAIT_DELAY      50
#define KEY_QUIT        'q'
#define KEY_RECORDING   ' '
#define KEY_SNAPSHOT    's'
#define KEY_VIEW_PREV   'c'
#define KEY_VIEW_NEXT   'v'
#define KEY_PLOT_SWITCH 'l'

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

    if (!builder)
    {
        E_ERROR << "error creating grabber builder for " << args.grabber;
        return EXIT_FAILURE;
    }

    if (args.cameras >= builder->GetDevices())
    {
        E_ERROR << "insufficient resources";
        E_ERROR << "the grabber builder is able to build " << builder->GetDevices() << " grabber(s), however " << args.cameras << " requested";

        return EXIT_FAILURE;
    }

    ImageGrabber::Ptrs grabbers;

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

    /*
    DummyImageGrabber cam0(buffer), cam1(buffer);
    BufferRecorder recorder(buffer, "seq");

    // UI stuff
    cv::Mat canvas;
    StereoImageRenderer imageRenderer(cam0, cam1);
    std::vector<BufferWriterStatsRenderer> statsRenderers;
    BufferUsageIndicator usageIndicator(buffer);
    BufferRecorderStatsRenderer recRenderer(recorder);

    statsRenderers.push_back(BufferWriterStatsRenderer(cam0, 1000 / WAIT_DELAY * 3));
    statsRenderers.push_back(BufferWriterStatsRenderer(cam1, 1000 / WAIT_DELAY * 3));
    statsRenderers[0].Rectangle = cv::Rect(32, 32, 320, 90);
    statsRenderers[1].Rectangle = cv::Rect(32, 138, 320, 90);

    size_t viewIdx = 0, numViews = StereoImageRenderer::ListedModes.size();
    imageRenderer.SetMode(StereoImageRenderer::ListedModes[viewIdx]);

    usageIndicator.Rectangle = cv::Rect(-192, -64, 160, 32);
    recRenderer.Origin = cv::Point(-400, 64);

    cv::namedWindow(WINDOW_TITLE, cv::WINDOW_AUTOSIZE);
 
    if (!cam0.Start() || !cam1.Start() || !recorder.Start()) return -1;

    for (int key = 0; key != KEY_QUIT; key = cv::waitKey(WAIT_DELAY))
    {
        switch (key) 
        {
        case KEY_VIEW_PREV:
            imageRenderer.SetMode(StereoImageRenderer::ListedModes[--viewIdx % numViews]);
            break;
        case KEY_VIEW_NEXT:
            imageRenderer.SetMode(StereoImageRenderer::ListedModes[++viewIdx % numViews]);
            break;
        case KEY_RECORDING:
            if (!recorder.IsRecording()) recorder.StartRecording();
            else recorder.StopRecording();
            break;
        case KEY_SNAPSHOT:
            recorder.Snapshot();
            break;
        case KEY_QUIT: //
            break;
        default:
            /////;
        }
        
        if (!imageRenderer.Draw(canvas))
        {
            E_ERROR << "error rendering camera views";
            continue;
        }

        BOOST_FOREACH(BufferWriterStatsRenderer& render, statsRenderers)
        {
            if (render.Draw(canvas)) continue;
            E_ERROR << "error rendering stats";
        }

        if (!usageIndicator.Draw(canvas)) E_ERROR << "error rendering buffer usage indicator";
        if (!recRenderer.Draw(canvas)) E_ERROR << "error rendering recorder stats";

        cv::imshow(WINDOW_TITLE, canvas);
    }

    cam0.Stop();
    cam1.Stop();
    recorder.Stop();
    */
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

#include "args.hpp"

namespace po = boost::program_options;

Args::Args(int argc, char* argv[])
{
    po::options_description general("General Options");
    general.add_options()
        ("in,i",    po::value<String>(&imageList  )->default_value("%1-%2.png"), "Specify either a text file containing image file names or a name pattern. A list file conatains exactly [#cameras] x [#images] lines; the first [#images] lines specify image files of camera #0, followed by next [#images] lines of camera #1 and so on. A valid pattern contains camera index %1 and image index %2 which ranged from 0 to [#cameras-1] and [#images-1] respectively. The string is considered a list file if neither %1 nor %2 present. This option is required to start a new calibration.")
        ("out,o",   po::value<String>(&calPath    )->default_value("caout"),     "Path to calibration output directory. Individual camera parameters will be written to c*.xml.")
        ("save",    po::value<String>(&profilePath)->default_value(""),          "Path to the calibration profile. General options -i, -m, -n and all the chessboard detection options are ignored if the specified profile exists. If this is not desried, plese use -f to override the existing profile.")
        ("cams,m",  po::value<size_t>(&cameras    )->default_value(0),           "Number of cameras to be calibrated. This option is required to start a new calibration.")
        ("views,n", po::value<size_t>(&images     )->default_value(0),           "Number of calibration images to be used. This option is required to start a new calibration.");

    po::options_description target("Chessboard Detection Options");
    target.add_options()
        ("def,d",       po::value<String>(&targetDef     )->default_value(""),   "Definition of calibration target as a ROWSxCOLSxMETRIC[xMETRIC] string (e.g. \"3x4x50\" describes a checkerboard with 3-by-4 inner corners and each black/white square has a size of 50x50 unit2). This option is required to start a new calibration.")
        ("adaptive",    po::value<bool>  (&adaptiveThresh)->default_value(true), "...")
        ("normalise",   po::value<bool>  (&normaliseImage)->default_value(true), "...")
        ("fastcheck",   po::value<bool>  (&fastCheck     )->default_value(true), "...")
        ("subpx-win",   po::value<int>   (&subpxWinSize  )->default_value(11),   "Windows size for sub-pixel refinement on the detected corners. Set to 0 to disable the refinement.")
        ("subpx-iters", po::value<size_t>(&subpxIters    )->default_value(30),   "Maximum number of iterations of sub-pixel refinement.")
        ("subpx-eps",   po::value<double>(&subpxEpsilon  )->default_value(1e-1), "...")
        ;

    po::options_description optim("Optimisation Options");
    optim.add_options()
        ("optim-pair",    po::bool_switch  (&optimPairwise)->default_value(true), "Apply pairwise optimisation before global adjustment using OpenCV's gradient descent implementation.")
        ("optim-all",     po::bool_switch  (&optimGlobal  )->default_value(true), "Enable global bundle adjustment using Levenberg-Marquardt algorithm.")
        ("optim-iters,k", po::value<size_t>(&optimIters   )->default_value(100),  "Maximum number of iterations of non-linear optimisation.")
        ("optim-eps,e",   po::value<double>(&optimEpsilon )->default_value(1e-3), "Maximum difference ratio in parameters and error updates that indicates a convergence through the non-linear optimisation.")
        ("optim-threads", po::value<size_t>(&optimThreads )->default_value(4),    "Number of numerical differentiation threads.");

    po::options_description miscs("Miscs. Options");
    miscs.add_options()
        ("help,h",    "Show this message and exit")
        ("version,v", "")
        ("force,f",   po::bool_switch(&force        )->default_value(false), "")
        ("report",    po::value<String>(&reportPath )->default_value(""),    "Enable reporting and dump the calibration report to the specified folder. Use empty string to suppress reporting.")
        ("logfile",   po::value<String>(&logfilePath)->default_value(""),    "")
        ("gnuplot",   po::value<String>(&gnuplotPath)->default_value(""),    "");

    m_options.add(general).add(target).add(optim).add(miscs);
	m_okay = Parse(argc, argv);
}

bool Args::Parse(int argc, char* argv[])
{
    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(m_options).run(), vm);

        help    = vm.count("help")    > 0;
        force   = vm.count("force")   > 0;
        version = vm.count("version") > 0;

        buildingOptionsSet =
            vm.count("in")        || 
            vm.count("cams")      ||
            vm.count("views")     ||
            vm.count("def")       || 
            vm.count("adaptive")  ||
            vm.count("normalise") ||
            vm.count("fastcheck") ||
            vm.count("subpx");

        if (help || version)
        {
            return false;
        }

        po::notify(vm);
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error parsing arguments: " << ex.what();
        E_ERROR << "Use \"" << argv[0] << " -h\" to see all the options";

        return false;
    }
   
    bool missingRequired =
        imageList.empty() || // -i
        cameras == 0      || // -m
        images == 0       || // -n
        targetDef.empty();   // -d

    if (missingRequired && profilePath.empty())
    {
        E_ERROR << "required option(s) missing";
        E_ERROR << "check if options -i, -m, -n and -d are set properly";
        E_ERROR << "use \"" << argv[0] << " -h\" to see all the options";

        return false;
    }

    return true;
}


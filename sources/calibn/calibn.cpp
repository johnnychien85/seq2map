#include "args.hpp"
#include "calibgraphbuilder.hpp"

void version()
{
    std::cout << "calibn 1.0.0" << std::endl;
}

int main(int argc, char* argv[])
{
    CalibGraph graph;
    Args args(argc, argv);

    if (!args.logfilePath.empty()) initLogFile(args.logfilePath);
    if (!args.IsOkay())
    {
        if (args.version || args.help)
        {
            if (args.help)
            {
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << args.GetOptions();
            }
            else if (args.version) version();           

            return EXIT_SUCCESS;
        }
        else
        {
            E_FATAL << "error parsing arguments";
            E_INFO  << "use \"" << argv[0] << " -h\" to see usage";

            return EXIT_FAILURE;
        }
    }   
   
    // check if data building is required
    bool freshstart = args.force || !fileExists(args.profilePath);

    // start from scratch, using a graph builder to construct an instance of 
    //  the m-camera n-view calibration problem.
    if (freshstart)
    {
        E_INFO << "start building a " << args.cameras << "-camera " << args.images << "-view calibration graph";

        CalibGraphBuilder builder(args.cameras, args.images);
        builder.SetAdaptiveThresholding(args.adaptiveThresh);
        builder.SetImageNormalisation(args.normaliseImage);
        builder.SetFastCheck(args.fastCheck);
        builder.SetSubPixelWinSize(args.subpxWinSize);
        builder.SetSubPixelTerm(args.subpxIters, args.subpxEpsilon);

        if (!builder.SetTargetDef(args.targetDef))
        {
            E_FATAL << "error setting target definition";
            return EXIT_FAILURE;
        }

        E_INFO << "calibration target definition: " << builder.GetTargetDef();

        if (!builder.SetFileList(args.imageList))
        {
            E_FATAL << "error setting file list";
            return EXIT_FAILURE;
        }

        if (!builder.Build(graph))
        {
            E_FATAL << "error building calibration graph";
            return EXIT_FAILURE;
        }

        if (!args.profilePath.empty() && !graph.Store(args.profilePath))
        {
            E_WARNING << "error storing built calibration graph to profile " << args.profilePath;
        }
    }
    else
    {
        if (!graph.Restore(args.profilePath))
        {
            E_FATAL << "error restoring calibration graph from profile " << args.profilePath;
            return EXIT_FAILURE;
        }

        if (args.buildingOptionsSet)
        {
            E_WARNING << "given data building option(s) eclipsed due to existing profile";
        }

        E_INFO << "calibration graph restored from " << args.profilePath;
    }

    //
    // Stage 2: Camera Calibration
    //
    if (!graph.Calibrate(args.optimPairwise))
    {
        E_FATAL << "initial calibration failed";
        return EXIT_FAILURE;
    }

    if (args.optimGlobal && !graph.Optimise(args.optimIters, args.optimEpsilon, args.optimThreads))
    {
        E_FATAL << "global optimisation failed";
        return EXIT_FAILURE;
    }

    // save optimised states of the profile
    if (!args.profilePath.empty() && !graph.Store(args.profilePath))
    {
        E_WARNING << "error storing profile to " << args.profilePath;    
    }

    //graph.WriteMFile("graph.m");

    // let's print something
    graph.Summary();

    // write the calibrated parameters
    if (!graph.WriteParams(args.calPath))
    {
        E_WARNING << "error writing calibrated parameters to " << args.calPath;
    }

    // report generation
    if (!args.reportPath.empty() && !graph.WriteReport(args.reportPath))
    {
        E_WARNING << "error writing calibration report to " << args.reportPath;
    }

    return 0;
}

#ifndef ARGS_HPP
#define ARGS_HPP

#include <seq2map/common.hpp>

using namespace seq2map;

class Args
{
public:
    /* ctor */
    Args(int argc, char* argv[]);

    inline bool IsOkay() const {return m_okay;};
    inline boost::program_options::options_description GetOptions() const { return m_options; }

    /**
     * Required parameters
     */
    String profilePath; // path to the optional input/output calibration data file, in either YML or XML format
    String calPath;     // path to the output parameter file, in either YML or XML format
    String imageList;   // either the path to the image file list or a pattern of file name
    size_t cameras;     // number of cameras
    size_t images;      // number of frames
    String targetDef;   // definition of the target calibration pattern

    /**
     * Parameters for chessboard detection options
     */
    bool   adaptiveThresh; // enable adaptive binarisation using Otsu's thresholding method
    bool   normaliseImage; // perform image normalisation
    bool   fastCheck;      // ...
    int    subpxWinSize;   // window size used to refine the detected corners
    size_t subpxIters;     // subpixel corner refinement stop criterion
    double subpxEpsilon;   // subpixel corner refinement stop criterion
    bool   buildingOptionsSet; // flag to indicate if any parameter involved in the data building process is set

    /**
     * Parameters for nonlinear optimisation options
     */
    bool   optimPairwise; // flag to enable/disable pair-wise local optimisation
    bool   optimGlobal;   // flag to enable/disable graph-wise global optimisation
    size_t optimIters;    // max iteration of nonlinear optimisation
    double optimEpsilon;  // differential stop criterion of the nonlinear optimisation
    size_t optimThreads;  // number of lunched numerical differentiation threads

    /**
     * Report generation and plotting options
     */
    String reportPath;  // optional output path to the report file
    String logfilePath; // optional output path of the log file
    String gnuplotPath; // optional path to the binary of gnuplot

    /**
     * misc. options
     */
    bool force;     // force to restart the whole calibration process even the specified profile exists
    bool version;   // show version and leave
    bool help;      // show help message and leave

protected:
    bool Parse(int argc, char* argv[]);
    bool m_okay;
    boost::program_options::options_description m_options;
};

#endif // ARGS_HPP

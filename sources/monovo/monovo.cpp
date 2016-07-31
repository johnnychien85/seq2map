#include <seq2map/features.hpp>

using namespace seq2map;
namespace po = boost::program_options;

struct Args
{
    Path srcPath;
    Path imgPath;
    bool help;
};

bool init(int, char*[], Args&);
void showSynopsis(char*);

int main(int argc, char* argv[])
{
    Args args;

    if (!init(argc, argv, args)) return args.help ? EXIT_SUCCESS : EXIT_FAILURE;

    try
    {
        Paths files = enumerateFiles(args.srcPath);
        int i = 0;

        cv::Mat I1, I2;
        ImageFeatureSet f1, f2;
        FeatureMatcher matcher;
        FundamentalMatFilter fmatFilter;
        StdevFilter stdevFilter;
        matcher.AddFilter(fmatFilter).AddFilter(stdevFilter);

        BOOST_FOREACH(const Path& file, files)
        {
            if (!f2.Restore(file))
            {
                E_ERROR << "error reading features from " << file.string();
                continue;
            }

            if (!f1.IsEmpty())
            {
                ImageFeatureMap map = matcher.MatchFeatures(f1, f2);

                Path imPath = args.imgPath / file.filename();
                imPath.replace_extension(".png");

                I2 = cv::imread(imPath.string());

                if (!I1.empty() && !I2.empty())
                {
                    cv::Mat im = map.Draw(I1, I2);
                    cv::imshow("matched result", im);
                    cv::waitKey(1);
                }
            }

            E_INFO << file;
            E_INFO << matcher.Report();

            f1 = f2;
            I1 = I2;
        }
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop: " << ex.what();
        return -1;
    }

    return 0;
}

bool init(int argc, char* argv[], Args& args)
{
    String srcPath, imgPath;

    po::options_description o("General Options");
    o.add_options()
        ("help,h", "Show this help message and exit.");

    po::options_description h("Hiddens");
    h.add_options()
        ("in", po::value<String>(&srcPath)->default_value(""), "Input folder containing features")
        ("im", po::value<String>(&imgPath)->default_value(""), "Input folder containing images");

    po::positional_options_description p;
    p.add("in", 1).add("im", 1);

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
    args.imgPath = imgPath;

    if (args.help)
    {
        std::cout << "Usage: " << argv[0] << " <features_in_dir> <images_in_dir> [options]" << std::endl;
        std::cout << o << std::endl;
    }

    if (srcPath.empty())
    {
        E_FATAL << "input feature directory path missing";
        showSynopsis(argv[0]);

        return false;
    }

    if (imgPath.empty())
    {
        E_FATAL << "input image directory path missing";
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

#include <seq2map/sequence.hpp>

namespace po = boost::program_options;

using namespace seq2map;

struct Args
{
    Path seqPath;
    bool show;
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
        Sequence seq;

        if (!seq.Restore(args.seqPath))
        {
            E_ERROR << "error loading sequence profile";
            return -1;
        }

        E_INFO << "sequence succesfully restored from " << args.seqPath;

        const Cameras& cams = seq.GetCameras();
        for (size_t i = 0; i < cams.size(); i++)
        {
            const Camera& cam = cams.at(i);
            size_t frames = cam.GetFrames();

            E_INFO << "checking camera " << i << " [" << cam.GetName() << "]";

            for (size_t j = 0; j < frames; j++)
            {
                Frame frame = cam[j];

                if (frame.im.empty())
                {
                    E_ERROR << "error reading frame " << j;
                    return -1;
                }

                E_INFO << "camera " << i << " frame " << j << "/" << frames << " checked";

                if (!frame.features.IsEmpty())
                {
                    if (frame.im.channels() != 3) frame.im = gray2rgb(frame.im);
                    cv::drawKeypoints(frame.im, frame.features.GetKeyPoints(), frame.im, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
                }

                if (args.show)
                {
                    cv::imshow("Sequence Check", frame.im);
                    cv::waitKey(1);
                }
            }
        }
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop";
        E_FATAL << ex.what();

        return -1;
    }

    E_INFO << "sequence verified without error";
    return 0;
}

bool init(int argc, char* argv[], Args& args)
{
    String seqPath;

    po::options_description o("General Options");
    o.add_options()
        ("show",   "Render sequence images")
        ("help,h", "Show this help message and exit.");

    po::options_description h("Hiddens");
    h.add_options()
        ("in", po::value<String>(&seqPath)->default_value(""), "Path to the input sequence profile");

    po::positional_options_description p;
    p.add("in", 1);

    try
    {
        po::options_description a;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(a.add(o).add(h)).positional(p).run();
        po::variables_map vm;

        po::store(parsed, vm);
        po::notify(vm);

        args.show = vm.count("show") > 0;
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

    args.seqPath = seqPath;

    if (args.help)
    {
        std::cout << "Usage: " << argv[0] << " <sequence_dir> <output_profile> [options]" << std::endl;
        std::cout << o << std::endl;

        return false;
    }

    if (seqPath.empty())
    {
        E_FATAL << "sequence profile path missing";
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

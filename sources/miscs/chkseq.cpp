#include <seq2map/sequence.hpp>

using namespace seq2map;

class MyApp : public App
{
public:
    MyApp(int argc, char* argv[]) : App(argc, argv) {}
protected:
    virtual void SetOptions(Options&, Options&, Positional&);
    virtual void ShowHelp(const Options&) const;
    virtual bool Init();
    virtual bool Execute();

    String m_seqPath;
    bool   m_checkStores;
    bool   m_show;
};

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ("all,a", po::bool_switch(&m_checkStores)->default_value(false), "Run a thorough check")
        ("show",  po::bool_switch(&m_show)->default_value(false),        "Render and show images, used with -a");

    h.add_options()
        ("in",    po::value<String>(&m_seqPath)->default_value(""), "Path to the input sequence database folder");

    p.add("in", 1);
}

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Check integrity of sequence." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <sequence_dir>" << std::endl;
    std::cout << o;
}

bool MyApp::Init()
{
    if (m_seqPath.empty())
    {
        E_ERROR << "sequence path not specified";
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    Sequence seq;
    Path from = m_seqPath;

    if (!seq.Restore(from))
    {
        E_ERROR << "error loading sequence profile";
        return EXIT_FAILURE;
    }

    E_INFO << "sequence succesfully restored from " << from;

    // show sequence summary
    std::cout << "Database:        " << seq.GetRootPath().string() << std::endl;
    std::cout << "Source:          " << seq.GetRawPath().string()  << std::endl;
    std::cout << "Vehicle:         " << seq.GetVehicle()           << std::endl;
    std::cout << "Grabber:         " << seq.GetGrabber()           << std::endl;
    std::cout << "Frames:          " << seq.GetFrames()            << std::endl;
    std::cout << "Cameras:         " << seq.GetCameras().size()    << std::endl;

    BOOST_FOREACH (const Camera& cam, seq.GetCameras())
    {
        std::cout << " Cam " << cam.GetIndex() << std::endl;
        std::cout << "  Name:          " << cam.GetName()      << std::endl;
        std::cout << "  Model:         " << cam.GetModel()     << std::endl;
        std::cout << "  Projection:    " << (cam.GetIntrinsics() ? cam.GetIntrinsics()->GetModelName() : "UNKNOWN!!") << std::endl;
        std::cout << "  Image size:    " << cam.GetImageSize() << std::endl;;
        std::cout << "  Extrinsics:    " << mat2string(cam.GetExtrinsics().GetTransformMatrix()) << std::endl;
        std::cout << "  Frames:        " << cam.GetFrames() << std::endl;;;
        std::cout << "  Feature store: " << cam.GetFeatureStores().size() << std::endl;;

        size_t i = 0;

        BOOST_FOREACH (const FeatureStore& fs, cam.GetFeatureStores())
        {
            std::cout << "   Store " << (i++) << std::endl;
            std::cout << "    Index:       " << fs.GetIndex() << std::endl;
            std::cout << "    Root:        " << fs.GetRoot()  << std::endl;
            std::cout << "    Items:       " << fs.GetItems() << std::endl;
            std::cout << "    Keypoint:    " << (fs.GetFeatureDetextractor() ? fs.GetFeatureDetextractor()->GetKeypointName()   : "UNKNOWN!!") << std::endl;
            std::cout << "    Descriptor:  " << (fs.GetFeatureDetextractor() ? fs.GetFeatureDetextractor()->GetDescriptorName() : "UNKNOWN!!") << std::endl;
        }
    }

    std::cout << "Stereo pairs: " << seq.GetRectifiedStereo().size() << std::endl;
    BOOST_FOREACH (const RectifiedStereo& stereo, seq.GetRectifiedStereo())
    {
        std::cout << " Pair " << stereo.ToString() << " ]" << std::endl;
    }
    std::cout << std::endl;

    if (!m_checkStores)
    {
        return true;
    }

    std::cout << "Initiating store check.." << std::endl;

    BOOST_FOREACH (const Camera& cam, seq.GetCameras())
    {
        std::cout << "Checking camera " << cam.GetIndex() << "..";

        for (size_t i = 0; i < cam.GetFrames(); i++)
        {
            cv::Mat im = cam.GetImageStore()[i].im;

            if (im.empty())
            {
                std::cout << "..FAILED" << std::endl;
                std::cout << "error reading frame " << i << std::endl;
                return false;
            }

            BOOST_FOREACH (const FeatureStore& fs, cam.GetFeatureStores())
            {
                ImageFeatureSet fset;

                if (!fs.Retrieve(i, fset))
                {
                    std::cout << "..FAILED" << std::endl;
                    std::cout << "error reading feature set " << i << " from feature store " << fs.GetIndex() << std::endl;
                    return false;
                }

                if (m_show)
                {
                    cv::drawKeypoints(im, fset.GetKeyPoints(), im, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
                }
            }

            if ((i % (cam.GetFrames() / 10)) == 0)
            {
                std::cout << ".." << i;
            }

            if (m_show)
            {
                cv::imshow("chkseq", im);
                cv::waitKey(1);
            }
        }

        std::cout << "..DONE" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Sequence verified without error." << std::endl;

    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

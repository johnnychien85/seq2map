#include <seq2map/app.hpp>
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

    std::cout << "== SEQUENCE ====================================================================" << std::endl;

    // show sequence summary
    std::cout << " Database:         " << seq.GetRootPath().string()      << std::endl;
    std::cout << " Source:           " << seq.GetRawPath().string()       << std::endl;
    std::cout << " Vehicle:          " << seq.GetVehicle()                << std::endl;
    std::cout << " Grabber:          " << seq.GetGrabber()                << std::endl;
    std::cout << " Frames:           " << seq.GetFrames()                 << std::endl;
    std::cout << " Cameras:          " << seq.GetCameras().size()         << std::endl;
    std::cout << " Stereo Pairs:     " << seq.GetStereoPairs().size()     << std::endl;
    std::cout << " Feature Stores:   " << seq.GetFeatureStores().size()   << std::endl;
    std::cout << " Disparity Stores: " << seq.GetDisparityStores().size() << std::endl;

    std::cout << std::endl;
    std::cout << "== CAMERAS =====================================================================" << std::endl;

    BOOST_FOREACH (const Camera::Map::value_type& pair, seq.GetCameras())
    {
        if (!pair.second)
        {
            std::cout << " Cam " << pair.first << " missing!!" << std::endl;
            continue;
        }

        const Camera& cam = *pair.second;
        ProjectionModel::ConstOwn intrinsics = cam.GetIntrinsics();
        cv::Mat pose = cam.GetExtrinsics().GetTransformMatrix();

        std::cout << " Cam " << cam.GetIndex() << std::endl;
        std::cout << "  Name:            " << cam.GetName()      << std::endl;
        std::cout << "  Model:           " << cam.GetModel()     << std::endl;
        std::cout << "  Projection:      " << (intrinsics ? intrinsics->GetModelName() : "UNKNOWN") << std::endl;
        std::cout << "  Image size:      " << cam.GetImageSize() << std::endl;;
        std::cout << "  Extrinsics:      " << mat2string(pose)   << std::endl;
        std::cout << "  Frames:          " << cam.GetFrames()    << std::endl;;;
    }

    std::cout << std::endl;
    std::cout << "== STEREO PAIRS ================================================================" << std::endl;

    BOOST_FOREACH (const RectifiedStereo::Set::value_type& pair, seq.GetStereoPairs())
    {
        std::cout << " Pair " << (pair ? pair->ToString() : "missing!!") << std::endl;

        std::cout << "  Configuration:   ";
        switch (pair->GetConfiguration())
        {
        case RectifiedStereo::LEFT_RIGHT:   std::cout << "Left-right";   break;
        case RectifiedStereo::TOP_BOTTOM:   std::cout << "Top-bottom";   break;
        case RectifiedStereo::BACK_FORWARD: std::cout << "Back-forward"; break;
        case RectifiedStereo::UNKNOWN:      std::cout << "Unknown";      break;
        }
        std::cout << std::endl;

        std::cout << "  Baseline:        " << pair->GetBaseline() << " world units" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "== FEATURE STORES ==============================================================" << std::endl;

    BOOST_FOREACH (const FeatureStore::Map::value_type& pair, seq.GetFeatureStores())
    {
        if (!pair.second)
        {
            std::cout << " Store " << pair.first << " missing!!" << std::endl;
            continue;
        }
        
        const FeatureStore& store = *pair.second;
        Camera::ConstOwn camera = store.GetCamera();
        FeatureDetextractor::ConstOwn dxtor = store.GetFeatureDetextractor();

        std::cout << " Store " << store.GetIndex() << std::endl;

        if (camera) std::cout << "  Camera:          " << camera->GetIndex() << std::endl;
        else        std::cout << "  Camera:          " << "UNKNOWN"          << std::endl;

        std::cout << "  Root:            " << store.GetRoot()  << std::endl;
        std::cout << "  Items:           " << store.GetItems() << std::endl;
        std::cout << "  Keypoint:        " << (dxtor ? dxtor->GetKeypointName()   : "UNKNOWN") << std::endl;
        std::cout << "  Descriptor:      " << (dxtor ? dxtor->GetDescriptorName() : "UNKNOWN") << std::endl;
    }

    std::cout << std::endl;
    std::cout << "== DISPARITY MAP STORES ========================================================" << std::endl;

    BOOST_FOREACH (const DisparityStore::Map::value_type& pair, seq.GetDisparityStores())
    {
        if (!pair.second)
        {
            std::cout << " Store " << pair.first << " missing!!" << std::endl;
            continue;
        }

        const DisparityStore& store = *pair.second;
        RectifiedStereo::ConstOwn stereo = store.GetStereoPair();
        StereoMatcher::ConstOwn  matcher = store.GetMatcher();

        std::cout << " Store " << store.GetIndex() << std::endl;

        if (stereo) std::cout << "  Stereo pair:     " << stereo->ToString() << std::endl;
        else        std::cout << "  Stereo pair:     " << "UNKNOWN"          << std::endl;

        std::cout << "  Root:            " << store.GetRoot()  << std::endl;
        std::cout << "  Items:           " << store.GetItems() << std::endl;
        std::cout << "  Matcher:         " << (matcher ? matcher->GetMatcherName() : "UNKNOWN") << std::endl;
    }

    std::cout << std::endl;

    if (!m_checkStores)
    {
        return true;
    }

    std::cout << "Initiating store check.." << std::endl;

    BOOST_FOREACH (const Camera::Map::value_type& pair, seq.GetCameras())
    {
        if (!pair.second) continue;

        const Camera& cam = *pair.second;
        Progress progress(cam.GetFrames());

        std::cout << "Checking camera " << cam.GetIndex() << "..";

        for (size_t i = 0; i < cam.GetFrames(); i++)
        {
            cv::Mat im = cam.GetImageStore()[i].im;

            if (im.empty())
            {
                std::cout << "..FAILED" << std::endl;
                std::cout << "Error reading frame " << i << std::endl;

                return false;
            }

            if (m_show)
            {
                cv::imshow("chkseq", im);
                cv::waitKey(1);
            }

            if (progress.IsMilestone(i))
            {
                std::cout << ".." << i;
            }
        }

        std::cout << "..DONE" << std::endl;
    }

    BOOST_FOREACH (const FeatureStore::Map::value_type& pair, seq.GetFeatureStores())
    {
        if (!pair.second) continue;

        const FeatureStore& store = *pair.second;
        Camera::ConstOwn camera = store.GetCamera();
        Progress progress(store.GetItems());

        std::cout << "Checking feature store " << store.GetIndex() << "..";

        for (size_t i = 0; i < store.GetItems(); i++)
        {
            ImageFeatureSet fset;
            cv::Mat im = (m_show && camera) ? camera->GetImageStore()[i].im : cv::Mat();

            if (!store.Retrieve(i, fset))
            {
                std::cout << "..FAILED" << std::endl;
                std::cout << "Error reading feature set " << i << " from feature store" << std::endl;

                return false;
            }

            if (!im.empty())
            {
                cv::drawKeypoints(im, fset.GetKeyPoints(), im, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
                cv::imshow("chkseq", im);
                cv::waitKey(1);
            }

            if (progress.IsMilestone(i))
            {
                std::cout << ".." << i;
            }
        }

        std::cout << "..DONE" << std::endl;
    }

    BOOST_FOREACH (const DisparityStore::Map::value_type& pair, seq.GetDisparityStores())
    {
        if (!pair.second)
        {
            std::cout << " Store " << pair.first << " missing!!" << std::endl;
            continue;
        }

        const DisparityStore& store = *pair.second;
        Progress progress(store.GetItems());

        std::cout << "Checking disparity store " << store.GetIndex() << "..";

        for (size_t i = 0; i < store.GetItems(); i++)
        {
            cv::Mat im = store[i].im;

            if (im.empty())
            {
                std::cout << "..FAILED" << std::endl;
                std::cout << "Error reading frame " << i << std::endl;

                return false;
            }

            if (m_show)
            {
                cv::imshow("chkseq", im);
                cv::waitKey(1);
            }

            if (progress.IsMilestone(i))
            {
                std::cout << ".." << i;
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

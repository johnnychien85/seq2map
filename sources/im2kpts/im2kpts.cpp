#include <iomanip>
#include <seq2map/sequence.hpp>

using namespace seq2map;
namespace po = boost::program_options;

class MyApp : public App
{
public:
    MyApp(int argc, char* argv[]) : App(argc, argv) {}

protected:
    virtual void SetOptions(Options&, Options&, Positional&);
    virtual void ShowHelp(const Options&) const;
    virtual bool ProcessUnknownArgs(const Strings& args);
    virtual bool Init();
    virtual bool Execute();

private:
    String m_seqPath;
    String m_outPath;
    String m_detector;
    String m_xtractor;
    String m_extension;
    String m_index;
    int    m_camIdx;
    FeatureDetextractorFactory::BasePtr m_dxtor;    
    ImageStore   m_imageStore;
    FeatureStore m_featureStore;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Image feature detection and extraction." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << std::endl;
    std::cout << "  " << m_exec.string() << " [options] <sequence_in_dir> <feature_out_dir>"  << std::endl;
    std::cout << "  " << m_exec.string() << " [options] --cam <camera_idx> <sequence_in_dir>" << std::endl;
    std::cout << std::endl;
    std::cout << o << std::endl;

    if (m_detector.empty() && m_xtractor.empty())
    {
        std::cout << "Please use -h with -k and/or -x to list feature-specific options." << std::endl;
        return;
    }

    const FeatureDetextractorFactory& dxtorFactory = FeatureDetextractorFactory::GetInstance();

    if (!m_detector.empty())
    {
        FeatureDetectorPtr detector = dxtorFactory.GetDetectorFactory().Create(m_detector);

        if (detector)
        {
            std::cout << detector->GetOptions(DETECTION_OPTIONS) << std::endl;
        }
        else
        {
            E_ERROR << "unknown feature detector \"" << m_detector << "\"";
        }
        
    }

    if (!m_xtractor.empty())
    {
        FeatureExtractorPtr xtractor = dxtorFactory.GetExtractorFactory().Create(m_xtractor);

        if (xtractor)
        {
            std::cout << xtractor->GetOptions(EXTRACTION_OPTIONS) << std::endl;
        }
        else
        {
            E_ERROR << "unknown descriptor extractor \"" << m_xtractor << "\"";
        }
    }
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    //
    // Prepare messages for argument parsing
    //
    const FeatureDetextractorFactory& dxtorFactory = FeatureDetextractorFactory::GetInstance();

    String detectorList = makeNameList(dxtorFactory.GetDetectorFactory(). GetRegisteredKeys());
    String xtractorList = makeNameList(dxtorFactory.GetExtractorFactory().GetRegisteredKeys());

    String detectorDesc = "Keypoint detector name, must be one of " + detectorList;
    String xtractorDesc = "Descriptor extractor name, must be one of " + xtractorList;

    xtractorDesc += " When the keypoint detector is not given explicitly, the extractor will be applied first to find image features, if it is capable to do so.";
    xtractorDesc += " In this case, the extractor must be one of " + makeNameList(dxtorFactory.GetRegisteredKeys());

    // some essential parameters
    o.add_options()
        ("detect,k",  po::value<String>(&m_detector )->default_value(         ""), detectorDesc.c_str())
        ("extract,x", po::value<String>(&m_xtractor )->default_value(         ""), xtractorDesc.c_str())
        ("index",     po::value<String>(&m_index    )->default_value("index.yml"), "Path to where the index of generated feature files to be written. Set to empty to disable index generation. This option is ignored for IN-SEQ mode.")
        ("out",       po::value<String>(&m_outPath  )->default_value(         ""), "Output folder in where the detected and extracted features are written. This option is ignored for IN-SEQ mode.")
        ("cam,c",     po::value<int>   (&m_camIdx   )->default_value(         -1), "Select camera from a sequence database to enable IN-SEQ mode.")
        ("ext,e",     po::value<String>(&m_extension)->default_value(     ".dat"), "The extension name of the output feature files.");

    // two positional arguments - input and output directories
    h.add_options()
        ("in",        po::value<String>(&m_seqPath  )->default_value(         ""), "Input folder containing images or sequence index file with ");

    p.add("in", 1);
}

bool MyApp::ProcessUnknownArgs(const Strings& args)
{
    if (m_xtractor.empty())
    {
        E_FATAL << "missing descriptor extractor name";
        return false;
    }

    m_detector = m_detector.empty() ? m_xtractor : m_detector;
    m_dxtor = FeatureDetextractorFactory::GetInstance().Create(m_detector, m_xtractor);

    if (!m_dxtor)
    {
        E_FATAL << "error creating feature detector-and-extractor object";
        return false;
    }

    // Finally the last mile..
    // parse options for the detector and extractor!
    try
    {
        Parameterised::Options detectorOptions = m_dxtor->GetOptions(DETECTION_OPTIONS);
        Parameterised::Options xtractorOptions = m_dxtor->GetOptions(EXTRACTION_OPTIONS);
        Parameterised::Options opts;

        opts.add(detectorOptions).add(xtractorOptions);

        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(args).options(opts).run();

        // store the parsed values to the detector object's internal data member(s)
        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (std::exception& ex)
    {
        E_FATAL << "error parsing feature-specific arguments: " << ex.what();
        std::cout << "Try to use -h with -k and/or -x to see the supported options for specific feature detectior/extractor" << std::endl;

        return false;
    }

    m_dxtor->ApplyParams();

    return true;
}

bool MyApp::Init()
{
    bool inSeqMode = m_camIdx >= 0;
    Sequence seq;

    // set image store for input
    if (inSeqMode)
    {
        size_t cam = static_cast<size_t>(m_camIdx);

        E_INFO << "IN-SEQ mode selected, trying to use camera " << cam << " from " << m_seqPath;

        if (!seq.Restore(m_seqPath))
        {
            E_ERROR << "IN-SEQ mode selected but unable to load sequence database from " << m_seqPath;
            return false;
        }

        if (cam >= seq.GetCameras().size())
        {
            E_ERROR << "camera index out of bound (idx=" << cam << ",size=" << seq.GetCameras().size() << ")";
            return false;
        }

        m_imageStore = seq.GetCamera(cam).GetImageStore();
    }
    else
    {
        E_INFO << "listing files in " << m_seqPath;
        m_imageStore.FromExistingFiles(m_seqPath);
    }

    if (m_imageStore.GetItems() == 0)
    {
        E_INFO << "no input image files detected, aborting..";
        return false;
    }

    E_INFO << m_imageStore.GetItems() << " frame(s) will be processed";

    // set feature store for output
    if (inSeqMode)
    {
        Path featureStoreRoot = seq.GetFeatureStoreRoot();
        m_outPath = "";

        for (size_t i = 0; i < 65535; i++)
        {
            std::stringstream ss;
            ss << std::setw(8) << std::setfill('0') << i;

            const Path newStorePath = featureStoreRoot / ss.str();

            if (!dirExists(newStorePath))
            {
                m_outPath = newStorePath.string();
                break;
            }
        }

        if (m_outPath.empty())
        {
            E_ERROR << "cannot find an available folder name in " << featureStoreRoot;
            return false;
        }
    }
    else if (m_outPath.empty())
    {
        E_ERROR << "missing output folder";
        return false;
    }

    Path outPath = Path(m_outPath);

    if (!m_featureStore.Create(m_outPath, inSeqMode ? static_cast<size_t>(m_camIdx) : 0, m_dxtor))
    {
        E_ERROR << "error creating feature store " << outPath;
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    try
    {
        Speedometre metre("Image Feature Detection & Extraction", "px/s");
        size_t frames = 0, features = 0, bytes = 0;

        E_INFO << "feature extraction procedure starts for " << m_imageStore.GetItems() << " file(s)";
        E_INFO << "source folder set to " << fullpath(m_imageStore.GetRoot());
        E_INFO << "output folder set to " << fullpath(m_featureStore.GetRoot());

        for (size_t i = 0; i < m_imageStore.GetItems(); i++)
        {
            Path src(m_imageStore.GetItemPath(i));
            Path dst(m_featureStore.GetItemPath(i));

            dst.replace_extension(m_extension);

            cv::Mat im = m_imageStore[i].im;

            if (im.empty())
            {
                E_INFO << "skipped unreadable file " << src;
                continue;
            }

            im = im.channels() == 3 ? rgb2gray(im) : im;

            metre.Start();
            ImageFeatureSet f = m_dxtor->DetectAndExtractFeatures(im);
            metre.Stop(im.total());

            if (!m_featureStore.Append(dst.filename().string(), f))
            {
                E_FATAL << "error writing features to " << dst;
                return false;
            }

            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << metre.GetSpeed() << " " << metre.GetUnit();

            frames++;
            features += f.GetSize();
            bytes += filesize(dst);

            E_INFO << "processed " << src.filename() << " -> " << dst.filename() << " [" << ss.str() << "]";
        }

        if (!m_index.empty())
        {
            Path to = m_featureStore.GetRoot() / Path(m_index);
            cv::FileStorage fs(to.string(), cv::FileStorage::WRITE);

            if (!m_featureStore.Store(fs))
            {
                E_ERROR << "error storing index to " << to;
                return false;
            }

            E_INFO << "index written to " << to;
        }

        E_INFO << "image feature extraction procedure finished, " << features << " feature(s) detected from " << frames << " frame(s)";
        E_INFO << "file storage:     " << (bytes / 1024.0f / 1024.0f) << " MBytes";
        E_INFO << "computation time: " << metre.GetElapsedSeconds() << " secs";
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop: " << ex.what();
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

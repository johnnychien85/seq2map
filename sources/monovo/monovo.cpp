#include <boost/algorithm/string/join.hpp>
#include <seq2map/app.hpp>
#include <seq2map/mapping.hpp>

using namespace seq2map;
namespace po = boost::program_options;

class MyApp : public App
{
public:
    MyApp(int argc, char* argv[]) : App(argc, argv) {}

protected:
    virtual void SetOptions(Options&, Options&, Positional&);
    virtual void ShowHelp(const Options&) const;
    virtual bool Init();
    virtual bool Execute();

private:
    struct AlignmentOptions
    {
        AlignmentOptions()
        : forwardProj (true ),
          backwardProj(false),
          epipolar    (false),
          photometric (false),
          rigid       (false)
        {}

        bool forwardProj;
        bool backwardProj;
        bool epipolar;
        bool photometric;
        bool rigid;
    };

    bool ParseAlignOptions(const String& flag, AlignmentOptions& options);
    bool ParseFlowOptions(const String& flow, bool& forward, bool& backward);

    String m_seqPath;
    String m_outPath;
    size_t m_kptsStoreId;
    int    m_dispStoreId;

    Sequence m_seq;
    FeatureStore::ConstOwn   m_featureStore;
    DisparityStore::ConstOwn m_disparityStore;

    String m_alignString;
    String m_flowString;
    AlignmentOptions m_alignment;
    size_t m_ransacIter;
    double m_ransacConf;

    bool m_forwardFlow;
    bool m_backwardFlow;
    bool m_blockMatching;
    int m_flowLevels;
    int m_blockSize;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Ego-motion estimation using monocular vision." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <sequence_database>" << std::endl;
    std::cout << o << std::endl;
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ("feature-store,f",   po::value<size_t>(&m_kptsStoreId)->default_value(0),   "Source feature store")
        ("disparity-store,d", po::value<int>   (&m_dispStoreId)->default_value(-1),  "Optional disparity store, set to -1 to disable use of disparity maps.")
        ("align-model,a",     po::value<String>(&m_alignString)->default_value(""),  "A string containning flags of alignment models to be enabled. \"B\" for backward projection, \"P\" for photometric, \"R\" for rigid, and \"E\" for epipolar alignment.")
        ("flow",              po::value<String>(&m_flowString)->default_value(""),   "Optical flow computation option for missed features. Valid strings are \"FORWARD\", \"BACKWARD\" and \"BIDIRECTION\"")
        ("ransac-iter",       po::value<size_t>(&m_ransacIter)->default_value(100),  "Max number of iterations for the RANSAC outlier rejection process.")
        ("ransac-conf",       po::value<double>(&m_ransacConf)->default_value(0.5f), "Expected probability that all the drawn samples are inliers during the RANSAC process.")
        ("block-matching",    po::bool_switch  (&m_blockMatching)->default_value(false),  "Enable block matching along epipolar lines to recover missing features.")
    ;

    h.add_options()
        ("seq", po::value<String>(&m_seqPath)->default_value(""), "Path to the input sequence.")
        ("out", po::value<String>(&m_outPath)->default_value(""), "Path to the output motion.");

    p.add("seq", 1).add("out", 1);
}

bool MyApp::Init()
{
    if (m_seqPath.empty())
    {
        E_ERROR << "missing input path";
        return false;
    }

    if (m_outPath.empty())
    {
        E_ERROR << "missing output path";
        return false;
    }

    if (!m_seq.Restore(m_seqPath))
    {
        E_ERROR << "error restoring sequence from " << m_seqPath;
        return false;
    }

    if (!ParseAlignOptions(m_alignString, m_alignment))
    {
        E_ERROR << "error parsing alignment flag \"" << m_alignString << "\"";
        return false;
    }

    if (!ParseFlowOptions(m_flowString, m_forwardFlow, m_backwardFlow))
    {
        E_ERROR << "error parsing optical flow option \"" << m_flowString << "\"";
        return false;
    }

    FeatureStore::ConstOwn   F = m_seq.GetFeatureStore(m_kptsStoreId);
    DisparityStore::ConstOwn D = m_dispStoreId > -1 ? m_seq.GetDisparityStore(static_cast<size_t>(m_dispStoreId)) : DisparityStore::ConstOwn();

    Camera::ConstOwn cam = F ? F->GetCamera() : Camera::ConstOwn();
    RectifiedStereo::ConstOwn stereo = D ? D->GetStereoPair() : RectifiedStereo::ConstOwn();

    if (!F)
    {
        E_ERROR << "missing feature store " << m_kptsStoreId;
        return false;
    }

    if (!cam)
    {
        E_ERROR << "missing camera";
        return false;
    }

    if (m_dispStoreId > -1 && !D)
    {
        E_ERROR << "missing disparity store";
        return false;
    }
    else if (!stereo)
    {
        E_ERROR << "missing stereo camera";
        return false;
    }
    else if (stereo->GetPrimaryCamera() != cam)
    {
        E_ERROR << "the primary camera of " << stereo->ToString() << " has to be cam " << cam->GetIndex();
        return false;
    }

    if (!D)
    {
        E_INFO << "no disparity store available, the estimated motion will have no absolute scale";
    }

    m_featureStore = F;
    m_disparityStore = D;

    // randomise the RANSAC process
    std::srand(static_cast<unsigned int>(std::time(0)));

    return true;
}

bool MyApp::Execute()
{
    FeatureStore::ConstOwn   F = m_featureStore;
    DisparityStore::ConstOwn D = m_disparityStore;

    Map map;
    FeatureTracker tracker(
        FeatureTracker::FramedStore(F, 0, D), // tracking from frame k
        FeatureTracker::FramedStore(F, 1, D)  // to frame k+1
    );

    Strings models;
    if (m_alignment.forwardProj)  models.push_back("FORWARD-PROJECTION");
    if (m_alignment.backwardProj) models.push_back("BACKWARD-PROJECTION");
    if (m_alignment.rigid)        models.push_back("RIGID");
    if (m_alignment.photometric)  models.push_back("PHOTOMETRIC");
    if (m_alignment.epipolar)     models.push_back("EPIPOLAR");

    E_INFO << "starting monocular visual odometry from sequence " << m_seqPath;
    E_INFO << "enabled alignment model: " << boost::algorithm::join(models, ", ");
    E_INFO << "optical flow direction:  " << m_flowString;

    if (m_alignment.forwardProj ) tracker.outlierRejection.scheme |= FeatureTracker::FORWARD_PROJ_ALIGN;
    if (m_alignment.backwardProj) tracker.outlierRejection.scheme |= FeatureTracker::BACKWARD_PROJ_ALIGN;
    if (m_alignment.rigid       ) tracker.outlierRejection.scheme |= FeatureTracker::RIGID_ALIGN;
    if (m_alignment.photometric ) tracker.outlierRejection.scheme |= FeatureTracker::PHOTOMETRIC_ALIGN;
    if (m_alignment.epipolar    ) tracker.outlierRejection.scheme |= FeatureTracker::EPIPOLAR_ALIGN;

    tracker.outlierRejection.maxIterations = m_ransacIter;
    tracker.outlierRejection.minInlierRatio = 0.4f;
    tracker.outlierRejection.confidence = m_ransacConf;
    tracker.outlierRejection.epipolarEps = 999;

    if (m_forwardFlow)   tracker.inlierInjection.scheme |= FeatureTracker::FORWARD_FLOW;
    if (m_backwardFlow)  tracker.inlierInjection.scheme |= FeatureTracker::BACKWARD_FLOW;
    if (m_blockMatching) tracker.inlierInjection.scheme |= FeatureTracker::EPIPOLAR_SEARCH;
    tracker.inlierInjection.blockSize = 5;
    tracker.inlierInjection.levels = 3;
    tracker.inlierInjection.bidirectionalEps = 1;
    tracker.inlierInjection.epipolarEps = 1800;
    tracker.inlierInjection.extractDescriptor = false;

    tracker.structureScheme = FeatureTracker::MIDPOINT_TRIANGULATION;

    Motion mot;
    Speedometre metre;

    mot.Update(EuclideanTransform::Identity);

    for (size_t t = 0; t < m_seq.GetFrames() - 1; t++)
    {
        metre.Start();
        bool success = tracker(map, t);
        metre.Stop(1);

        E_INFO << "Frame " << t << " -> " << (t+1) << ": "
            << tracker.stats.spawned  << " spawned, "
            << tracker.stats.tracked  << " tracked, "
            << tracker.stats.injected << " injected, "
            << tracker.stats.removed  << " removed, "
            << tracker.stats.joined   << " joined";
        E_INFO << "Matcher: " << tracker.matcher.Report();

        BOOST_FOREACH (FeatureTracker::Stats::ObjectiveStats::value_type pair, tracker.stats.objectives)
        {
            E_INFO << "Outlier model " << std::setw(2) << pair.first << ": "
                << std::setw(5) << pair.second.inliers << " / " << std::setw(5) << pair.second.population
                << " (" << pair.second.secs << " secs)";
        }

        //tracker.stats

        if (!success)
        {
            E_ERROR << "error tracking frame " << t << " -> " << (t+1);
            return false;
        }

        E_INFO << mat2string(tracker.stats.motion.pose.GetRotation().ToVector(), "R");
        E_INFO << mat2string(tracker.stats.motion.pose.GetTranslation(), "t");

        std::stringstream ss;
        ss << "M(:,:," << (t + 2) << ")";
        //of << mat2string(map.GetFrame(t + 1).poseEstimate.pose.GetTransformMatrix(true), ss.str()) << std::endl;
        //of.flush();

        mot.Update(tracker.stats.motion.pose);
    }

    E_INFO << "runtime : " << metre.GetSpeed() << " fps";

    if (!mot.Store(Path(m_outPath)))
    {
        E_ERROR << "error writing motion to \"" << m_outPath << "\"";
    }

    return true;
}

bool MyApp::ParseAlignOptions(const String& flag, AlignmentOptions& options)
{
    for (size_t i = 0; i < flag.length(); i++)
    {
        switch (flag[i])
        {
        case 'B': options.backwardProj = true; break;
        case 'P': options.photometric  = true; break;
        case 'R': options.rigid        = true; break;
        case 'E': options.epipolar     = true; break;
        default:
            E_ERROR << "unknown model \"" << flag[i] << "\"";
            return false;
        }
    }

    return true;
}

bool MyApp::ParseFlowOptions(const String& flow, bool& forward, bool& backward)
{
    if (flow.empty())
    {
        forward = backward = false;
        return true;
    }

    if (flow.compare("FORWARD") == 0)
    {
        forward = true;
        backward = false;

        return true;
    }

    if (flow.compare("BACKWARD") == 0)
    {
        forward = false;
        backward = true;

        return true;
    }

    if (flow.compare("BIDIRECTION") == 0)
    {
        forward = backward = true;
        return true;
    }

    E_ERROR << "unknown flow option \"" << flow << "\"";
    return false;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

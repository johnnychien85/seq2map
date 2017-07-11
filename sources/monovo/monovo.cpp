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

    String m_seqPath;
    String m_outPath;
    size_t m_kptsStoreId;
    int    m_dispStoreId;

    Sequence m_seq;
    FeatureStore::ConstOwn   m_featureStore;
    DisparityStore::ConstOwn m_disparityStore;

    AlignmentOptions m_alignment;
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
        ("feature-store,f",   po::value<size_t>(&m_kptsStoreId)->default_value(0),  "source feature store")
        ("disparity-store,d", po::value<int>   (&m_dispStoreId)->default_value(-1), "optional disparity store, set to -1 to disable use of disparity maps.")
        ("back-projection",   po::bool_switch  (&m_alignment.backwardProj)->default_value(false), "enable backward-projection ego-motion estimation in addition to the forward-projection model.")
        ("epipolar",          po::bool_switch  (&m_alignment.epipolar    )->default_value(false), "enable epipolar objective for ego-motion estimation.")
        ("rigid",             po::bool_switch  (&m_alignment.rigid       )->default_value(false), "enable rigid alignment model for ego-motion estimation.")
        ("photometric",       po::bool_switch  (&m_alignment.photometric )->default_value(false), "enable photometric alignment model for ego-motion estimation.");

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

    if (m_alignment.forwardProj ) tracker.outlierRejection.scheme |= FeatureTracker::FORWARD_PROJ_ALIGN;
    if (m_alignment.backwardProj) tracker.outlierRejection.scheme |= FeatureTracker::BACKWARD_PROJ_ALIGN;
    if (m_alignment.rigid       ) tracker.outlierRejection.scheme |= FeatureTracker::RIGID_ALIGN;
    if (m_alignment.photometric ) tracker.outlierRejection.scheme |= FeatureTracker::PHOTOMETRIC_ALIGN;
    if (m_alignment.epipolar    ) tracker.outlierRejection.scheme |= FeatureTracker::EPIPOLAR_ALIGN;

    tracker.outlierRejection.maxIterations = 50;
    tracker.outlierRejection.minInlierRatio = 0.5f;
    tracker.outlierRejection.confidence = 0.5f;
    tracker.outlierRejection.epipolarEps = 999;

    tracker.inlierInjection.scheme |= FeatureTracker::FORWARD_FLOW;
    tracker.inlierInjection.scheme |= FeatureTracker::BACKWARD_FLOW;
    tracker.inlierInjection.scheme |= FeatureTracker::EPIPOLAR_SEARCH;
    tracker.inlierInjection.levels = 3;
    tracker.inlierInjection.blockSize = 5;
    tracker.inlierInjection.searchRange = 7;
    tracker.inlierInjection.extractDescriptor = true;

    Motion mot;
    Speedometre metre;

    mot.Update(EuclideanTransform::Identity);

    for (size_t t = 0; t < m_seq.GetFrames() - 1; t++)
    {
        metre.Start();
        bool success = tracker(map, t);
        metre.Stop(1);

        E_INFO << "Frame " << t << " -> " << (t+1) << ": "
            << tracker.stats.spawned << " spawned, "
            << tracker.stats.tracked << " tracked, "
            << tracker.stats.removed << " removed, "
            << tracker.stats.joined  << " joined";
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

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

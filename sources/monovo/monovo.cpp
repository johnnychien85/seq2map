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
    String m_seqPath;
    String m_outPath;
    size_t m_kptsStoreId;
    int    m_dispStoreId;

    Sequence m_seq;
    FeatureStore::ConstOwn   m_featureStore;
    DisparityStore::ConstOwn m_disparityStore;
    int m_start;
    int m_until;
    int m_seed;

    FeatureTracker m_tracker;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Ego-motion estimation using monocular vision." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <sequence_database> <output_pose_txt>" << std::endl;
    std::cout << o << std::endl;
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ("feature-store,f",   po::value<size_t>(&m_kptsStoreId  )->default_value(    0), "Source feature store")
        ("disparity-store,d", po::value<int>   (&m_dispStoreId  )->default_value(   -1), "Optional disparity store, set to -1 to disable using disparity maps.")
        ("start",             po::value<int>   (&m_start        )->default_value(    0), "Start frame.")
        ("until",             po::value<int>   (&m_until        )->default_value(   -1), "Last frame. Set to negative number to go through the whole sequence.")
        ("seed",              po::value<int>   (&m_seed         )->default_value(   -1), "Seed for random number generation. Set to a negative number to use system time as seed.")
    ;

    o.add(m_tracker.GetOptions());

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

    // randomise the RANSAC process
    std::srand(static_cast<unsigned int>(m_seed < 0 ? std::time(0) : m_seed));

    return true;
}

bool MyApp::Execute()
{
    E_INFO << "starting monocular visual odometry from sequence " << m_seqPath;

    FeatureStore::ConstOwn   F = m_featureStore;
    DisparityStore::ConstOwn D = m_disparityStore;

    Map map;
    Source& src = map.AddSource(F, D);

    m_tracker.ApplyParams();
    m_tracker.matcher.SetMaxRatio(0.8f);
    m_tracker.matcher.SetUniqueness(true);
    m_tracker.matcher.SetSymmetric(false);
    m_tracker.inlierInjection.extractDescriptor = false;

    Motion mot;
    Speedometre metre;
    FeatureTracker::Stats& stats = m_tracker.stats;

    mot.Update(EuclideanTransform::Identity);

    m_until = m_until > 0 && m_until > m_start ? m_until : m_seq.GetFrames() - 1;
    map.GetFrame(m_start).pose.valid = true; // set the starting frame as the reference frame

    size_t kf = INVALID_INDEX;

    for (size_t t = m_start; t < m_until; t++)
    {
        const size_t ti = t;
        const size_t tj = t + 1;

        metre.Start();
        bool success = m_tracker(map, src, map.GetFrame(ti), src, map.GetFrame(tj));
        metre.Stop(1);

        E_INFO << "Frame " << ti << " -> " << tj << ": "
            << stats.spawned  << " spawned, "
            << stats.tracked  << " tracked, "
            << stats.injected << " injected, "
            << stats.removed  << " removed, "
            << stats.joined   << " joined, "
            << map.GetLandmarks() << " accumulated";

        E_INFO << "Matcher: " << m_tracker.matcher.Report();

        BOOST_FOREACH (FeatureTracker::Stats::ObjectiveStats::value_type pair, stats.objectives)
        {
            E_INFO << "Outlier model " << std::setw(2) << pair.first << ": "
                << std::setw(5) << pair.second.inliers << " / " << std::setw(5) << pair.second.population
                << " (" << pair.second.secs << " secs)";
        }

        if (!success)
        {
            E_ERROR << "error tracking frame " << ti << " -> " << tj;
            mot.Store(Path("failed.txt"));
            return false;
        }

        E_INFO << mat2string(stats.motion.pose.GetRotation().ToVector(), "R");
        E_INFO << mat2string(stats.motion.pose.GetTranslation(), "t");

        const double covis = kf == INVALID_INDEX ? 0.0f : map.GetFrame(kf).GetCovisibility(map.GetFrame(tj));

        if (kf == INVALID_INDEX || covis < 0.25f)
        {
            E_INFO << "added frame " << ti << " as a keyframe";
            kf = ti;
        }
        else
        {
            E_INFO << "covis(" << kf << "," << ti << ") = " << std::setprecision(2) << (100.0f * covis) << "%";
        }

        mot.Update(stats.motion.pose);
    }

    // covisibility matrix
    // cv::Mat cvs = cv::Mat::zeros(m_until, m_until, CV_64F);
    // for (size_t ti = m_start; ti != m_until; ti++)
    // {
    //    for (size_t tj = ti + 1; tj < m_until; tj++)
    //    {
    //        cvs.at<double>(ti, tj) = map.GetFrame(ti).GetCovisibility(map.GetFrame(tj));
    //    }
    // }
    // PersistentMat(cvs).Store(Path("cvs.bin"));

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

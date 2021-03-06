#include <seq2map/app.hpp>
#include <seq2map/mapping.hpp>

using namespace seq2map;

class TrackingPath : public Persistent<cv::FileStorage, cv::FileNode>
{
public:
    struct Node
    {
        Node() : source(INVALID_INDEX), frame(INVALID_INDEX) {}

        size_t source;
        size_t frame;
    };

    /**
     * Null constructor
     */
    TrackingPath() {}

    /**
     * Explicit constructor
     */
    TrackingPath(size_t s0, size_t t0, size_t s1, size_t t1);

    /**
     * Copy constructor for std::vector::push_back
     */
    TrackingPath(const TrackingPath& tr)
    : TrackingPath(tr.m_source.source, tr.m_source.frame, tr.m_target.source, tr.m_target.frame) { tracker = tr.tracker; }

    /**
     *
     */
    bool operator() (Map& map, size_t t, size_t& ti, size_t& tj);

    /**
     *
     */
    bool InScope(size_t t, size_t n) const { return IsOkay() && m_source.frame + t < n && m_target.frame + t < n; }

    /**
     *
     */
    inline bool IsOkay() const { return m_okay; }

    /**
     *
     */
    inline bool IsSynched() const { return m_source.frame == m_target.frame; }

    //
    // Persistence
    //
    virtual bool Store(cv::FileStorage& fn) const;
    virtual bool Restore(const cv::FileNode& fs);

    FeatureTracker tracker;

private:
    bool Check();

    bool m_okay;
    Node m_source;
    Node m_target;
};

class Mapper : public Persistent<Path>
{
public:
    struct SourceDef
    {
        SourceDef(size_t kpts = INVALID_INDEX, size_t disp = INVALID_INDEX)
        : kptsStore(kpts), dispStore(disp) {}

        SourceDef(const SourceDef& def) : kptsStore(def.kptsStore), dispStore(def.dispStore) {}

        size_t kptsStore;
        size_t dispStore;
    };

    bool operator() (Map& map, size_t t, size_t n);

    //
    // Persistence
    //
    bool Mapper::Store(Path& to) const;
    bool Mapper::Restore(const Path& from);

    std::vector<SourceDef> sources;
    std::vector<TrackingPath> tracking;
};

class MyApp : public App
{
public:
    MyApp(int argc, char* argv[]) : App(argc, argv) {}

protected:
    virtual void SetOptions(Options&, Options&, Positional&);
    virtual void ShowHelp(const Options&) const;
    virtual bool Init();
    virtual bool Execute();

    String   seqPath;
    String   outPath;
    String   mapperPath;
    size_t   start;
    size_t   until;
    Sequence seq;
    Map      map;
    Mapper   mapper;
    double   minKfCovis;
};

TrackingPath::TrackingPath(size_t s0, size_t t0, size_t s1, size_t t1)
{
    m_source.source = s0;
    m_source.frame  = t0;
    m_target.source = s1;
    m_target.frame  = t1;

    Check();
}

bool TrackingPath::Check()
{
    m_okay = m_source.source != INVALID_INDEX && m_source.frame  != INVALID_INDEX &&
             m_target.source != INVALID_INDEX && m_target.frame  != INVALID_INDEX &&
             (m_source.frame != m_target.frame || m_source.source != m_target.source);

    return m_okay;
}

bool TrackingPath::operator() (Map& map, size_t t, size_t& ti, size_t& tj)
{
    return m_okay && tracker(map,
        map.GetSource(m_source.source), map.GetFrame(ti = m_source.frame + t),
        map.GetSource(m_target.source), map.GetFrame(tj = m_target.frame + t)
    );
}

bool TrackingPath::Store(cv::FileStorage& fs) const
{
    fs << "source" << "{";
    {
        fs << "source" << m_source.source;
        fs << "frame"  << m_source.frame;
    }
    fs << "}";

    fs << "target" << "{";
    {
        fs << "source" << m_target.source;
        fs << "frame " << m_target.frame;
    }
    fs << "}";

    fs << "tracker" << "{";
    tracker.WriteParams(fs);
    fs << "}";

    return true;
}

bool TrackingPath::Restore(const cv::FileNode& fn)
{
    if (!tracker.ReadParams(fn["tracker"]))
    {
        E_ERROR << "error reading tracker parameters";
        return false;
    }

    tracker.rendering = false;

    fn["source"]["source"] >> m_source.source;
    fn["source"]["frame"]  >> m_source.frame;
    fn["target"]["source"] >> m_target.source;
    fn["target"]["frame"]  >> m_target.frame;

    return Check();
}

bool Mapper::operator() (Map& map, size_t t, size_t n)
{
    for (size_t i = 0; i < tracking.size(); i++)
    {
        TrackingPath& tr = tracking[i];
        const FeatureTracker::Stats& stats = tr.tracker.stats;
        size_t ti, tj;
        
        if (tr.InScope(t, n) && !tr(map, t, ti, tj)) return false;

        /*
        E_INFO 
            << stats.spawned  << " spawned, "
            << stats.tracked  << " tracked, "
            << stats.injected << " injected, "
            << stats.removed  << " removed, "
            << stats.joined   << " joined, "
            << map.GetLandmarks() << " accumulated";
        */
        //E_INFO << "Matcher: " << tr.tracker.matcher.Report();
    }

    return true;

    // if (i != tracking.size() - 1) continue;
    // pose-only bundle adjustment
    // if (!stats.fresh && !tr.IsSynched() && (tj == optimised.GetIndex() || ti == optimised.GetIndex()))
    Frame& optimised = map.GetFrame(t+1);
    if (tracking.size() > 1 && optimised.size() > 0)
    {
        typedef std::map<size_t, GeometricMapping::WorldToImageBuilder> BuilderMap;
        typedef std::map<size_t, Landmark::Ptrs> LandmarkMap;

        BuilderMap  builders;
        LandmarkMap lmks;

        for (Frame::const_iterator hit = optimised.cbegin(); hit != optimised.cend(); hit++)
        {
            Landmark& lmk = hit.GetContainer<0, Landmark>();
            Frame&    frm = hit.GetContainer<1, Frame   >();
            Source&   src = hit.GetContainer<2, Source  >();

            if (lmk.size() == 1 || lmk.position.z == 0) continue; // skip one-time observation

            // E_INFO << "lmk=" << lmk.GetIndex() << ",frm=" << frm.GetIndex() << ",src=" << src.GetIndex() << ",idx=" << src.store->GetIndex() << ",pos=" << lmk.position << ",proj=" << hit->proj;
            builders[src.GetIndex()].Add(lmk.position, hit->proj, lmk.GetIndex());
            lmks[src.GetIndex()].push_back(&lmk);
        }

        ConsensusPoseEstimator estimator;
        PoseEstimator::Estimate estimate;
        //PoseEstimator::ConstOwn solver = PoseEstimator::ConstOwn(new DummyPoseEstimator(optimised.pose.pose));
        PoseEstimator::ConstOwn solver;
        GeometricMapping solverData;

        for (BuilderMap::const_iterator itr = builders.begin(); itr != builders.end(); itr++)
        {
            ProjectionModel::ConstOwn proj = map.GetSource(itr->first).store->GetCamera()->GetPosedProjection();
            AlignmentObjective::Own obj = AlignmentObjective::Own(new ProjectionObjective(proj));

            GeometricMapping data = itr->second.Build();
            data.metric = map.GetStructure(lmks[itr->first]).metric->Reduce();

            if (!obj->SetData(data))
            {
                E_ERROR << "error setting 3D-to-2D perspective constraint(s) for source " << itr->first;
                return false;
            }

            if (!solver)
            {
                solver = PoseEstimator::ConstOwn(new PerspevtivePoseEstimator(proj));
                solverData = data;
            }

            estimator.AddSelector(obj->GetSelector(5.0f));
        }

        estimator.SetStrategy(ConsensusPoseEstimator::RANSAC);
        estimator.SetMaxIterations(10);
        estimator.SetMinInlierRatio(0.1f);
        estimator.SetConfidence(0.99f);
        estimator.SetSolver(solver);
        estimator.SetVerbose(true);
        estimator.EnableOptimisation();

        std::vector<IndexList> survived, eliminated;

        if (!estimator(solverData, estimate, survived, eliminated))
        {
            E_WARNING << "error optimising frame pose of frame " << optimised.GetIndex();

            PersistentMat(solverData.src.mat).Store(Path("src.bin"));
            PersistentMat(solverData.dst.mat).Store(Path("dst.bin"));
            PersistentMat(cv::Mat(solverData.indices)).Store(Path("idx.bin"));
            PersistentMat(optimised.pose.pose.GetTransformMatrix()).Store(Path("mot.bin"));

            map.Store(Path("dump"));

            return false;
        }
        else
        {
            optimised.pose = estimate;
        }
    }


    return true;
}

bool Mapper::Store(Path& to) const
{
    cv::FileStorage fs(to.string(), cv::FileStorage::WRITE);

    fs << "sources" << "[";
    BOOST_FOREACH (const SourceDef& src, sources)
    {
        fs << "{";
        fs << "keyPoints" << src.kptsStore;
        fs << "disparity" << src.dispStore;
        fs << "}";
    }
    fs << "]";

    fs << "tracking" << "[";
    BOOST_FOREACH (const TrackingPath& tr, tracking)
    {
        fs << "{";
        tr.Store(fs);
        fs << "}";
    }
    fs << "]";

    return true;
}

bool Mapper::Restore(const Path& from)
{
    sources.clear();
    tracking.clear();

    try
    {
        cv::FileStorage fs(from.string(), cv::FileStorage::READ);

        if (!fs.isOpened())
        {
            E_ERROR << "error reading " << from;
            return false;
        }

        cv::FileNode sourcesNode = fs["sources"];

        for (cv::FileNodeIterator itr = sourcesNode.begin(); itr != sourcesNode.end(); itr++)
        {
            SourceDef def;
            (*itr)["keyPoints"] >> def.kptsStore;
            (*itr)["disparity"] >> def.dispStore;

            sources.push_back(def);
        }

        cv::FileNode trackingNode = fs["tracking"];

        for (cv::FileNodeIterator itr = trackingNode.begin(); itr != trackingNode.end(); itr++)
        {
            TrackingPath tr;

            if (!tr.Restore(*itr))
            {
                E_ERROR << "error restoring tracking node";
                return false;
            }

            tracking.push_back(tr);
        }
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error restoring from " << from;
        E_ERROR << ex.what();

        return false;
    }

    return true;
}

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Egomotion estimation and 3D mapping." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <sequence_database> <output_map>" << std::endl;
    std::cout << o << std::endl;
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ("mapper,m", po::value<String>(&mapperPath)->default_value(  ""), "...")
        ("start",    po::value<size_t>(&start     )->default_value(   0), "Start frame.")
        ("until",    po::value<size_t>(&until     )->default_value(   0), "Last frame. Set to zero to go through the whole sequence.")
        ("kf-covis", po::value<double>(&minKfCovis)->default_value(0.3f), "Minimum covisibility to consider a frame keyframe.")
        ;

    h.add_options()
        ("seq", po::value<String>(&seqPath)->default_value(""), "Path to the input sequence.")
        ("out", po::value<String>(&outPath)->default_value(""), "Path to the output map directory.");

    p.add("seq", 1).add("out", 1);
}

bool MyApp::Init()
{
    if (seqPath.empty())
    {
        E_ERROR << "missing input path";
        return false;
    }

    if (outPath.empty())
    {
        E_ERROR << "missing output path";
        return false;
    }

    if (mapperPath.empty())
    {
        E_ERROR << "missing mapper profile";
        return false;
    }

    if (!seq.Restore(seqPath))
    {
        E_ERROR << "error restoring sequence from " << seqPath;
        return false;
    }

    if (!mapper.Restore(mapperPath))
    {
        E_ERROR << "error restoring mapper from " << mapperPath;
        return false;
    }

    if (start >= seq.GetFrames() || until >= seq.GetFrames())
    {
        E_ERROR << "sequence range [" << start << "," << until << ") out of bound";
        E_ERROR << "the sequence has " << seq.GetFrames() << " frame(s)";

        return false;
    }

    until = until > 0 ? until : seq.GetFrames();

    if (start > until)
    {
        E_ERROR << "starting frame " << start << " exceeds the boundary of " << until;
        return false;
    }

    // initialise the map
    map.SetSequencePath(seqPath);
    map.GetFrame(start).pose.valid = true; // set the starting frame as the reference frame

    for (size_t i = 0; i < mapper.sources.size(); i++)
    {
        const Mapper::SourceDef& def = mapper.sources[i];
        Source& src = map.GetSource(i);

        src.store = seq.GetFeatureStore(def.kptsStore);
        src.dpm   = def.dispStore != INVALID_INDEX ? seq.GetDisparityStore(def.dispStore) : DisparityStore::Own();

        if (!src.store)
        {
            E_ERROR << "required feature store " << def.kptsStore << " by source " << i << " is missing";
            return false;
        }

        if (!src.dpm && def.dispStore != INVALID_INDEX)
        {
            E_ERROR << "required disparity store " << def.dispStore << " by source " << i << " is missing";
            return false;
        }
    }

    return true;
}

bool MyApp::Execute()
{
    E_INFO << "starting mapping sequence " << seqPath;
    E_INFO << "starting from frame: " << start << " to " << until;

    Speedometre metre;

    for (size_t t = start; t < until; t++)
    {
        metre.Start();
        if (!mapper(map, t, until))
        {
            E_ERROR << "error mapping frame " << t;
            return false;
        }
        metre.Stop(1);

        const double fps = metre.GetFrequency();
        E_INFO << "proccesed frame " << t << " at "
            << std::fixed << std::setprecision(2) << fps << " fps, "
            << std::fixed << std::setprecision(2) << ((double)(until - t) / fps) << "s left";

        const Frame& ti = map.GetFrame(t);
        const Frame& tj = map.GetFrame(t + 1);
        const size_t kf = map.GetLastKeyframe();
        const double covis = kf == INVALID_INDEX ? 0.0f : map.GetFrame(kf).GetCovisibility(ti);

        if (kf == INVALID_INDEX || covis < minKfCovis)
        {
            E_INFO << "added frame " << ti.GetIndex() << " as a keyframe";
            map.AddKeyframe(ti.GetIndex());
        }
    }

    if (!map.Store(Path(outPath)))
    {
        E_ERROR << "error saving map to \"" << outPath << "\"";
        return false;
    }

    E_INFO << "map saved to " << fullpath(outPath);

    if (mapperPath.empty())
    {
        return true;
    }

    Motion mot;
    mot.Update(EuclideanTransform::Identity);

    for (size_t t = start; t < until - 1; t++)
    {
        const Frame& ti = map.GetFrame(t);
        const Frame& tj = map.GetFrame(t + 1);
        
        if (!ti.pose.valid || !tj.pose.valid)
        {
            E_ERROR << "motion from " << ti.GetIndex() << " to " << tj.GetIndex() << " not solved!";
            mot.Store(Path(outPath) / "mot.failed.txt");

            return false;
        }

        mot.Update(ti.pose.pose.GetInverse() >> tj.pose.pose);
    }

    Path motPath = Path(outPath) / "mot.txt";
    mot.Store(motPath);

    E_INFO << "motion saved to " << motPath;

    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

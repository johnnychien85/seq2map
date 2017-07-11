#include <seq2map/mapping.hpp>

using namespace seq2map;

class Mapper
{
public:
    struct Capability
    {
        bool motion; // ego-motion estimation
        bool metric; // metric reconstruction
        bool dense;  // dense or semi-dense reconstruction
    };

    virtual Capability GetCapability() const = 0;
    virtual bool SLAM(Map& map, size_t t0, size_t tn) = 0;
};

/**
* Implementation of a generalised multi-frame feature integration algorithm
* based on the paper "Visual Odometry by Multi-frame Feature Integration"
*/
class MultiFrameFeatureIntegration : public Mapper
{
public:
    virtual Capability GetCapability() const;
    virtual bool SLAM(Map& map, size_t t0, size_t tn);

    bool AddTracking(const FeatureTracker& tracking);

private:
    std::vector<FeatureTracker> m_tracking;
    std::vector<DisparityStore::ConstOwn> m_dispStores;
};

// class OrbSLAM : public Mapper {
// ...
// };
//
// class LargeScaleDenseSLAM : public Mapper {
// ...
// };

//==[ MultiFrameFeatureIntegration ]==========================================//

bool MultiFrameFeatureIntegration::AddTracking(const FeatureTracker& matching)
{
    if (!matching.IsOkay())
    {
        E_WARNING << "invalid matching " << matching.ToString();
        return false;
    }

    BOOST_FOREACH (const FeatureTracker& m, m_tracking)
    {
        bool duplicated = (m.src == matching.src) && (m.dst == matching.dst);

        if (duplicated)
        {
            E_WARNING << "duplicated matching " << matching.ToString();
            return false;
        }
    }

    m_tracking.push_back(matching);
    return true;
}

Mapper::Capability MultiFrameFeatureIntegration::GetCapability() const
{
    Capability capability;

    capability.motion = false;
    capability.metric = m_dispStores.size() > 0;
    capability.dense = false;

    BOOST_FOREACH (const FeatureTracker& m, m_tracking)
    {
        capability.motion |= !m.IsSynchronised();
        capability.metric |= m.IsCrossed();
    }

    return capability;
}

bool MultiFrameFeatureIntegration::SLAM(Map& map, size_t t0, size_t tn)
{
    for (size_t t = t0; t < tn; t++)
    {
        BOOST_FOREACH (FeatureTracker& f, m_tracking)
        {
            if (f.InRange(t, tn) && !f(map, t))
            {
                E_ERROR << "error matching " << f.ToString();
                return false;
            }
        }


    }

    return true;
}


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
    String m_mapperName;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Ego-motion estimation and 3D mapping." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <sequence_database>" << std::endl;
    std::cout << o << std::endl;

}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ("mapper,m", po::value<String>(&m_mapperName)->default_value(""), "...");

    h.add_options()
        ("seq", po::value<String>(&m_seqPath)->default_value(""), "Path to the input sequence.");

    p.add("seq", 1);
}

bool MyApp::Init()
{
    if (m_seqPath.empty())
    {
        E_ERROR << "missing input path";
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    Sequence seq;

    if (!seq.Restore(m_seqPath))
    {
        E_ERROR << "error restoring sequence from " << m_seqPath;
        return false;
    }

    MultiFrameFeatureIntegration mapper;
    Map map;

    FeatureStore::ConstOwn f0 = seq.GetFeatureStore(0);
    FeatureStore::ConstOwn f1 = seq.GetFeatureStore(1);

    FeatureTracker::FramedStore f00(f0, 0), f01(f0, 1), f10(f1, 0), f11(f1, 1);

    mapper.AddTracking(FeatureTracker(f00, f01));
    mapper.AddTracking(FeatureTracker(f01, f10));
    mapper.AddTracking(FeatureTracker(f10, f11));
    mapper.AddTracking(FeatureTracker(f11, f00));
    
    if (!mapper.SLAM(map, 0, seq.GetFrames()))
    {
        E_ERROR << "error mapping sequence " << m_seqPath;
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

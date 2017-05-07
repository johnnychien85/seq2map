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

    if (seq.GetCameras().size() == 0 || 
        seq.GetCamera(0).GetFeatureStores().size() == 0)
    {
        return false;
    }

    const FeatureStore& f = seq.GetCamera(0).GetFeatureStore(0);
    FeatureMatcher matcher(true, true, 0.6f, false);

    for (size_t t = 0; t < seq.GetFrames() - 1; t++)
    {
        size_t ti = t;
        size_t tj = t + 1;

        ImageFeatureMap map = matcher.MatchFeatures(f[ti], f[tj]);
        E_INFO << ti << "->" << tj << ": " << matcher.Report();
    }

    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

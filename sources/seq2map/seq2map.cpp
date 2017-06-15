#include <seq2map/mapping.hpp>

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

    MultiFrameFeatureIntegration mapper;
    Map map;

    FeatureStore::ConstOwn f0 = seq.GetFeatureStore(0);
    FeatureStore::ConstOwn f1 = seq.GetFeatureStore(1);

    FeatureMatching::FramedStore f00(f0, 0), f01(f0, 1), f10(f1, 0), f11(f1, 1);

    mapper.AddMatching(FeatureMatching(f00, f01));
    mapper.AddMatching(FeatureMatching(f01, f10));
    mapper.AddMatching(FeatureMatching(f10, f11));
    mapper.AddMatching(FeatureMatching(f11, f00));
    
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

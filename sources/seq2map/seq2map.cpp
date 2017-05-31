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
    spamap::Map3<int> map;
    spamap::Map<int> map2;

    const size_t n = 1e5;
    Speedometre m0, m1;

    for (size_t i = 0; i < n; i++)
    {
        size_t i0 = (size_t)(std::rand() % 100);
        size_t i1 = (size_t)(std::rand() % 100);
        size_t i2 = (size_t)(std::rand() % 100);
        int v = std::rand() % 100;

        {
            AutoSpeedometreMeasure measure(m0, 1);
            map.Insert(i0, i1, i2, v);
        }
        {
            AutoSpeedometreMeasure measure(m1, 1);
            map2.Insert(i0, i1, v);
        }

        E_INFO << m0.ToString() << "/" << m1.ToString();
    }

    Sequence seq;
    /*
    if (!seq.Restore(m_seqPath))
    {
        E_ERROR << "error restoring sequence from " << m_seqPath;
        return false;
    }

    MultiFrameFeatureIntegration mapper;
    Map map;

    mapper.SetMergePolicy(MultiFrameFeatureIntegration::REJECT);
    mapper.AddPathway(0, 0, 0, 1);
    mapper.AddPathway(0, 1, 1, 1);
    mapper.AddPathway(1, 1, 1, 0);
    mapper.AddPathway(1, 0, 0, 0);
    //mapper.AddPathway(1, 1, 0, 1);
    //mapper.AddPathway(0, 1, 0, 1);
    //mapper.AddPathway(1, 0, 0, 1);
    
    if (!mapper.SLAM(seq, map))
    {
        E_ERROR << "error mapping sequence " << m_seqPath;
        return false;
    }
    */
    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

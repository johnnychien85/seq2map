#include <seq2map/app.hpp>
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

    String srcPath;
    String dstPath;
    Map src;
    Map dst;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Multi-sequence integration." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <source_map> <target_map>" << std::endl;
    std::cout << o << std::endl;
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ;

    h.add_options()
        ("src", po::value<String>(&srcPath)->default_value(""), "Path to the source map.")
        ("dst", po::value<String>(&dstPath)->default_value(""), "Path to the target map.");

    p.add("src", 1).add("dst", 1);
}

bool MyApp::Init()
{
    if (srcPath.empty())
    {
        E_ERROR << "missing path to source map";
        return false;
    }

    if (dstPath.empty())
    {
        E_ERROR << "missing path to target map";
        return false;
    }

    if (!src.Restore(Path(srcPath)))
    {
        E_ERROR << "error restoring source map from \"" << srcPath << "\"";
        return false;
    }

    if (!dst.Restore(Path(dstPath)))
    {
        E_ERROR << "error restoring target map from \"" << dstPath << "\"";
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    E_INFO << "GOOD!";
    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

#include "scanner.hpp"

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
    String m_rawPath;
    String m_outPath;
    String m_seqName;
    String m_builderName;
    SeqBuilderFactory m_factory;
    SeqBuilderFactory::BasePtr m_builder;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Make a new sequence database from raw data folder." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <sequence_in_dir> <database_out_dir>" << std::endl;
    std::cout << o << std::endl;

    if (m_builderName.empty())
    {
        std::cout << "Please use -h with -b to see sequence builder specific options.";
        return;
    }

    SeqBuilderFactory::BasePtr builder = m_factory.Create(m_builderName);

    if (!builder)
    {
        E_ERROR << "unknown sequence builder \"" << m_builderName << "\"";
        return;
    }

    std::cout << builder->GetOptions();
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;
    String builders = "Sequence builder, must be one of " + makeNameList(m_factory.GetRegisteredKeys());

    o.add_options()
        ("name,n",    po::value<String>(&m_seqName    )->default_value(""), "Name of the sequence.")
        ("builder,b", po::value<String>(&m_builderName)->default_value(""), builders.c_str());

    h.add_options()
        ("raw", po::value<String>(&m_rawPath)->default_value(""), "Path to the input sequence.")
        ("out", po::value<String>(&m_outPath)->default_value(""), "Path to the output folder.");

    p.add("raw", 1).add("out", 1);
}

bool MyApp::ProcessUnknownArgs(const Strings& args)
{
    namespace po = boost::program_options;

    if (m_builderName.empty())
    {
        E_ERROR << "missing builder name";
        return false;
    }

    m_builder = m_factory.Create(m_builderName);

    if (!m_builder)
    {
        E_ERROR << "unknown sequence builder \"" << m_builderName << "\"";
        return false;
    }

    try
    {
        Options opts = m_builder->GetOptions();

        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(args).options(opts).run();
        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error parsing builder-specific arguments: " << ex.what();
        std::cout << "Please use -h with -b to see supported options.";

        return false;
    }

    m_builder->ApplyParams();
    return true;
}

bool MyApp::Init()
{
    assert(m_builder);

    if (m_rawPath.empty())
    {
        E_ERROR << "missing input path";
        return false;
    }

    if (m_outPath.empty())
    {
        E_ERROR << "missing output path";
        return false;
    }

    if (m_seqName.empty())
    {
        E_ERROR << "sequence name not specified";
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    Sequence seq;

    Path seqDirName  = Path(m_seqName);
    Path seqFullPath = m_outPath / seqDirName;
    Path rawFullPath = fullpath(m_rawPath);

    if (!makeOutDir(seqFullPath))
    {
        E_ERROR << "error creating output dir " << seqFullPath;
        return false;
    }

    if (!m_builder->Build(rawFullPath, m_seqName, m_builderName, seq))
    {
        E_ERROR << "error building sequence " << rawFullPath;
        return false;
    }

    if (!seq.Store(seqFullPath))
    {
        E_ERROR << "error storing sequence to " << seqFullPath;
        return false;
    }

    E_INFO << "sequence succesfuilly built in " << seqFullPath;
    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

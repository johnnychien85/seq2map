#include <boost/log/core.hpp> // for log level setting
#include <boost/log/expressions.hpp>
#include <boost/algorithm/string.hpp>
#include <seq2map/app.hpp>

namespace logging = boost::log;

using namespace seq2map;

//==[ App ]===================================================================//

App::App(int argc, char* argv[])
: m_parser(boost::program_options::command_line_parser(argc, argv)),
    m_exec(argc > 0 ? String(argv[0]) : "")
{
    // make the default path to the log file
    m_logfile = m_exec;
    m_logfile.replace_extension("log");
}

int App::Run()
{
    namespace po = boost::program_options;

    Options o("General options"), h("hidden");
    Positional p;
    String logfile;
    String loglevel;
    Strings unknownArgs;

    o.add_options()
        ("help,h",    po::bool_switch  (&m_help  )->default_value(false),              "Show this help message and exit.")
        ("log-file",  po::value<String>(&logfile )->default_value(m_logfile.string()), "Path to the log file.")
        ("log-level", po::value<String>(&loglevel)->default_value("warning"),          "Log level, can be one of \"trace\", \"debug\", \"info\", \"warning\", \"error\", and \"fatal\".");

    SetOptions(o, h, p);

    try
    {
        Options a; // all options
        po::parsed_options parsed = m_parser.options(a.add(o).add(h)).positional(p).allow_unregistered().run();
        po::variables_map vm;

        po::store(parsed, vm);
        po::notify(vm);

        unknownArgs = po::collect_unrecognized(parsed.options, po::exclude_positional);
    }
    catch (po::error& pe)
    {
        E_FATAL << "error parsing general arguments: " << pe.what();
        return EXIT_FAILURE;
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caugth: " << ex.what();
        return EXIT_FAILURE;
    }

    if (m_help)
    {
        ShowHelp(o);
        return EXIT_SUCCESS;
    }

    if (!SetLogLevel(boost::to_lower_copy(loglevel)))
    {
        E_ERROR << "error setting log level to \"" << loglevel << "\"";
    }

    if (!initLogFile(m_logfile = logfile))
    {
        E_WARNING << "error writing to log file " << m_logfile;
    }

#ifdef NDEBUG
    try
    {
#endif
        if (!ProcessUnknownArgs(unknownArgs) || !Init())
        {
            ShowSynopsis();
            return EXIT_FAILURE;
        }

        return Execute() ? EXIT_SUCCESS : EXIT_FAILURE;
#ifdef NDEBUG
    }
    catch (std::exception& ex)
    {
        E_FATAL << "unhandled exception caught";
        E_FATAL << ex.what();

        return EXIT_FAILURE;
    }
#endif
}

bool App::SetLogLevel(const String& level)
{
    boost::log::trivial::severity_level severity;
    std::istringstream(level) >> severity;

    switch (severity)
    {
    case boost::log::trivial::severity_level::trace:
    case boost::log::trivial::severity_level::debug:
    case boost::log::trivial::severity_level::info:
    case boost::log::trivial::severity_level::warning:
    case boost::log::trivial::severity_level::error:
    case boost::log::trivial::severity_level::fatal:

        boost::log::core::get()->set_filter(boost::log::trivial::severity >= severity);
        E_INFO << "log level set to " << severity;

        return true;
    }

    E_WARNING << "unknown level \"" << level << "\"";
    return false;
}

bool App::ProcessUnknownArgs(const Strings& args)
{
    if (!args.empty())
    {
        E_ERROR << "unknown argument(s) detected: " << makeNameList(args);
        return false;
    }

    return true;
}

void App::ShowSynopsis() const
{
    std::cout << std::endl;
    std::cout << "Try \"" << m_exec.string() << " -h\" for usage listing." << std::endl;
}

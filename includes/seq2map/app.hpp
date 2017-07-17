#ifndef APP_HPP
#define APP_HPP
#include <seq2map/common.hpp>

namespace seq2map
{
    /**
     * Application class to provide an unified interface for seq2map utilities.
     */
    class App
    {
    public:
        int Run();

    protected:
        typedef boost::program_options::options_description Options;
        typedef boost::program_options::positional_options_description Positional;

        /* ctor */ App(int argc, char* argv[]);
        /* dtor */ virtual ~App() {}
        virtual void SetOptions(Options& general, Options& hidden, Positional& positional) = 0;
        bool SetLogLevel(const String& level);
        virtual void ShowSynopsis() const;
        virtual void ShowHelp(const Options& options) const = 0;
        virtual bool ProcessUnknownArgs(const Strings& args);
        virtual bool Init() = 0;
        virtual bool Execute() = 0;

        const Path m_exec;

    private:
        boost::program_options::command_line_parser m_parser;
        bool m_help;
        Path m_logfile;
    };
}

#endif //APP_HPP

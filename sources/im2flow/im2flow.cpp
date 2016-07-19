#include <seq2map/features.hpp>

using namespace seq2map;
namespace po = boost::program_options;

struct Args
{
    bool help;
};

bool init(int, char*[], Args&);
void showSynopsis(char*);
String makeNameList(Strings names);

int main(int argc, char* argv[])
{
    Args args;

    if (!init(argc, argv, args)) return args.help ? EXIT_SUCCESS : EXIT_FAILURE;

    try
    {
        // ...
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop: " << ex.what();
        return -1;
    }

    return 0;
}


bool init(int argc, char* argv[], Args& args)
{
    return false;
}

void showSynopsis(char* exec)
{
    std::cout << std::endl;
    std::cout << "Try \"" << exec << " -h\" for usage listing." << std::endl;
}

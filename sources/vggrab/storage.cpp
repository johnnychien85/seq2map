#include "storage.hpp"

using namespace seq2map;

VOStorage::VOStorage(const Path& root, size_t numCams)
    : _root(root), _numCams(numCams),
    _calibDirName("cal"), _rawDirName("raw"), _rectDirName("rect"),
    _disparityDirName("disparity"), _featuresDirName("features")
{
    if (root.empty())
    {
        assert(numCams == 0);
        return;
    }

    if (Load(root)) return;
    else E_INFO << "unable to load " << root << ", trying to create a new sequence";

    if (Create(root, numCams)) return;
    else E_ERROR << "error creating " << root;
}

bool VOStorage::Create(const Path& root, size_t numCams)
{
    assert(numCams > 0);

    Path rawDir = root / _rawDirName;
    _rawDirPaths.clear();
    _rawImageBaseNames.clear();

    // Raw image directory
    if (!makeOutDir(rawDir))
    {
        E_ERROR << "error creating raw data dir " << rawDir;
        return false;
    }

    for (size_t i = 0; i < numCams; i++)
    {
        std::stringstream ss; ss << "c" << i;
        Path camDir = rawDir / ss.str();

        if (!makeOutDir(camDir))
        {
            E_ERROR << "error creating raw data dir " << camDir;
            return false;
        }

        _rawDirPaths.push_back(camDir);
    }

    // DONE..
    _numCams = numCams;
    _numFrames = 0;

    E_INFO << root << " initialised";

    return true;
}

bool VOStorage::Load(const Path& root)
{
    Path rawDir = root / _rawDirName;
    if (!dirExists(rawDir)) return false;

    _rawDirPaths = enumerateDirs(rawDir);
    _numCams = _rawDirPaths.size();

    size_t i = 0;

    BOOST_FOREACH (const Path& camDir, _rawDirPaths)
    {
        Paths files = enumerateFiles(camDir);

        if (i == 0) _numFrames = files.size();
        else if (_numFrames != files.size()) return false;

        i++;
    }

    E_INFO << "sequence " << root << " loaded, " << _numFrames << " frame(s) found)";

    return true;
}

bool VOStorage::AddFrame(const String& filename, const std::vector<cv::Mat>& images)
{
    assert(images.size() == _numCams);

    for (size_t i = 0; i < _numCams; i++)
    {
        Path path = _rawDirPaths[i] / filename;

        if (path.extension().empty())
        { // use PGM / PPM by default
            path.replace_extension(images[i].channels() == 3 ? ".ppm" : ".pgm");
        }

        try
        {
            cv::imwrite(path.string(), images[i]);
        }
        catch (std::exception& ex)
        {
            E_ERROR << "error writing " << path.string();
            E_ERROR << ex.what();

            return false;
        }

        if (i == _numCams - 1)
        {
            _rawImageBaseNames.push_back(path.filename());
            _numFrames++;
        }

    }

    return true;
}

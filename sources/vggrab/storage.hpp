#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <seq2map/common.hpp>

using namespace seq2map;

class VOStorage
{
public:
    VOStorage(const Path& root = "", size_t numCams = 0);

    bool Create(const Path& root, size_t numCams);
    bool Load(const Path& root);
    bool AddFrame(const std::string& filename, const std::vector<cv::Mat>& images);
    inline size_t GetNumOfFrames() const {return _numFrames;}
    inline bool IsOkay() const {return _numCams > 0;}

protected:
    Path _root;
    size_t _numCams;
    size_t _numFrames;

    const std::string _calibDirName;
    const std::string _rawDirName;
    const std::string _rectDirName;
    const std::string _disparityDirName;
    const std::string _featuresDirName;

    Paths _rawDirPaths;
    Paths _rawImageBaseNames;
};

#endif //STORAGE_HPP

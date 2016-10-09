#ifndef CALIBPROFILE_HPP
#define CALIBPROFILE_HPP
#include <seq2map/common.hpp>
//#include <calibn/calibcam.hpp>
//#include <calibn/calibgraph.hpp>

using namespace seq2map;

class CalibProfile : public Persistent<Path>
{
public:
    /* ctor */ CalibProfile() : m_numCameras(0), m_numImages(0) {}
	bool Create(size_t cams, size_t imgs, const String& imageList, const String& def);
	bool Build(bool adaptiveThresh, bool normaliseImage, bool fastCheck, size_t subpxWinSize);
	bool Calibrate(bool pairwiseOptim);
	bool Optimise(size_t iter, double eps, size_t threads);
    bool WriteReport(const Path& reportPath);
    bool WriteParams(const Path& calPath);
    void Summary() const;
    virtual bool Store(Path& path) const;
    virtual bool Restore(const Path& path);

protected:
    class ImageFileList
    {
    public:
        inline void SetRoot(const Path& root) {m_root = root;}
        inline Path operator() (size_t cam, size_t img) const {return m_root / m_list[cam][img];}
        bool FromPattern(const String& pattern, size_t cams, size_t imgs);
        bool FromFile(const Path& listFile, size_t cams, size_t imgs);
        inline bool IsOkay() const {return !m_list.empty();}
        bool CheckImageFiles() const;

        static bool CheckPattern(const String& pattern);

    protected:
        Path                 m_root;
        std::vector<Strings> m_list;
    };

    inline bool	ReadyToBuild()     const {return m_numCameras > 0 && m_numImages > 0 && !m_imageFileName.empty();}
    inline bool	ReadyToCalibrate() const {return !m_cams.empty();}

    ImageFileList           m_imageFiles;
	//CalibPattern			m_pattern;

private:
	size_t					m_numCameras;
	size_t					m_numImages;
	//CalibCameras			m_cams;
	//CalibGraph              m_graph;
};

#endif //CALIBPROFILE_HPP

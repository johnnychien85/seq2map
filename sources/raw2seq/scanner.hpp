#ifndef SCANNER_HPP
#define SCANNER_HPP
#include <seq2map/sequence.hpp>

using namespace seq2map;

/* class Scanner : public Parameterised
{
public:
    virtual void WriteParams(cv::FileStorage& fs) const = 0;
    virtual bool ReadParams(const cv::FileNode& fn) = 0;
    virtual void ApplyParams() = 0;
    virtual Options GetOptions(int flag = 0) = 0;
    virtual bool Scan(const Path& from, Sequence& seq) = 0;
}; */

class KittiOdometryBuilder : public Sequence::Builder
{
public:
    virtual void WriteParams(cv::FileStorage& fs) const;
    virtual bool ReadParams(const cv::FileNode& fn);
    virtual void ApplyParams() {}
    virtual Options GetOptions(int flag = 0);

protected:
    static size_t ParseIndex(const String& varname, size_t offset = 1);
    virtual bool BuildCamera(const Path& from, Cameras& cams, RectifiedStereoPairs& stereo) const;

    friend class KittiRawDataBuilder;

private:
    /**
     * KITTI calib.txt parser
     */
    struct Calib
    {
    public:
        Calib(const Path& from);
        std::vector<cv::Mat> P;
        cv::Mat Tr;
        inline bool IsOkay() const { return m_okay; }
    private:
        bool m_okay;
    };

    String m_posePath; // path to pose file
};

class KittiRawDataBuilder : public Sequence::Builder
{
public:
    virtual void WriteParams(cv::FileStorage& fs) const;
    virtual bool ReadParams(const cv::FileNode& fn);
    virtual void ApplyParams() {}
    virtual Options GetOptions(int flag = 0);

protected:
    virtual bool BuildCamera(const Path& from, Cameras& cams, RectifiedStereoPairs& stereo) const;

private:
    /**
     * KITTI calib_cam_to_cam.txt parser
     */
    struct CalibCam2Cam
    {
    public:
        struct Entry
        {
            cv::Mat S;
            cv::Mat K;
            cv::Mat D;
            cv::Mat R;
            cv::Mat T;
            cv::Mat S_rect;
            cv::Mat R_rect;
            cv::Mat P_rect;
        };

        CalibCam2Cam(const Path& from);
        inline bool IsOkay() const { return m_okay; }

        std::vector<Entry> data;

    private:
        bool m_okay;
    };

    /**
     * KITTI extrinsics file (e.g. calib_imu_to_velo.txt) parser
     */
    struct CalibRigid
    {
    public:
        cv::Mat R;
        cv::Mat T;
        EuclideanTransform tform;

        CalibRigid(const Path& from);
        inline bool IsOkay() const { return m_okay; }

    private:
        bool m_okay;
    };

    bool m_rectified;
    String m_cam2cam; // path to the calib_cam_to_cam.txt file
    String m_imu2lid; // path to the calib_imu_to_velo.txt file
    String m_lid2cam; // path to the calib_velo_to_cam.txt file
};

class SeqBuilderFactory : public Factory<String, Sequence::Builder>
{
public:
    SeqBuilderFactory();
};
#endif // SCANNER_HPP

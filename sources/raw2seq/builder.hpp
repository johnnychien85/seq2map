#ifndef BUILDER_HPP
#define BUILDER_HPP
#include <seq2map/sequence.hpp>

using namespace seq2map;

/**
 * Class to parse VO/SLAM datasets from KITTI Benchmark Suite, KIT.
 * Source: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
 */
class KittiOdometryBuilder : public Sequence::Builder
{
public:
    virtual void WriteParams(cv::FileStorage& fs) const;
    virtual bool ReadParams(const cv::FileNode& fn);
    virtual void ApplyParams() {}
    virtual Options GetOptions(int flag = 0);

protected:
    static size_t ParseIndex(const String& varname, size_t offset = 1);

    virtual String GetVehicleName(const Path& from) const { return "KITTI CAR"; }
    virtual bool BuildCamera(const Path& from, Camera::Map& cams, RectifiedStereo::Set& stereo) const;

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

/**
 * Class to parse raw data from KITTI Benchmark Suite, KIT.
 * Source: http://www.cvlibs.net/datasets/kitti/raw_data.php
 */
class KittiRawDataBuilder : public Sequence::Builder
{
public:
    virtual void WriteParams(cv::FileStorage& fs) const;
    virtual bool ReadParams(const cv::FileNode& fn);
    virtual void ApplyParams() {}
    virtual Options GetOptions(int flag = 0);

protected:
    virtual String GetVehicleName(const Path& from) const { return "KITTI CAR"; }
    virtual bool BuildCamera(const Path& from, Camera::Map& cams, RectifiedStereo::Set& stereo) const;

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

/**
 * Class to parse aerial visual-intertial datasets from Autonomous Systems Lab, ETH.
 * Source: http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
 */
class EurocMavBuilder : public Sequence::Builder
{
public:
    virtual void WriteParams(cv::FileStorage& fs) const {}
    virtual bool ReadParams(const cv::FileNode& fn) { return true; }
    virtual void ApplyParams() {}
    virtual Options GetOptions(int flag = 0) { return Options(); }

protected:
    virtual String GetVehicleName(const Path& from) const;
    virtual bool BuildCamera(const Path& from, Camera::Map& cams, RectifiedStereo::Set& stereo) const;

    static bool ReadConfig(const Path& fromm, cv::FileStorage& to);
};

class SeqBuilderFactory : public Factory<String, Sequence::Builder>
{
public:
    SeqBuilderFactory();
};
#endif // BUILDER_HPP

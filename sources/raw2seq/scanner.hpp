#ifndef SCANNER_HPP
#define SCANNER_HPP
#include <seq2map/sequence.hpp>

using namespace seq2map;

class Scanner
{
public:
    virtual bool Scan(const Path& seqPath, const Path& calPath, const Path& motPath, Sequence& seq) = 0;
};

class KittiOdometryScanner : public Scanner
{
public:
    virtual bool Scan(const Path& seqPath, const Path& calPath, const Path& motPath, Sequence& seq);
protected:
    static size_t ParseIndex(const String& varname, size_t offset = 1);
    friend class KittiRawDataScanner;
private:
    /**
     * KITTI's calib.txt parser
     */
    struct Calib
    {
        Calib(const Path& calPath);
        std::vector<cv::Mat> P;
        cv::Mat Tr;
    };
};

class KittiRawDataScanner : public Scanner
{
public:
    virtual bool Scan(const Path& seqPath, const Path& calPath, const Path& motPath, Sequence& seq);
private:
    /**
    * KITTI's calib_cam_to_cam.txt parser
    */
    struct Calib
    {
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

        Calib(const Path& calPath);
        std::vector<Entry> data;
    };
};

class ScannerFactory : public Factory<String, Scanner>
{
public:
    ScannerFactory();
};
#endif // SCANNER_HPP

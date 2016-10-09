#ifndef CALIBCAM_HPP
#define CALIBCAM_HPP

#include <calibn/calibdata.hpp>
#include <cassert>

class CalibVectorisable
{
public:
    virtual std::vector<double>	ToVector() const = 0;
    virtual bool			FromVector(const std::vector<double>&) = 0;
    virtual size_t			GetSize() const = 0;
};

class CalibIntrinsics : CalibVectorisable
{
public:
    /* ctor */				CalibIntrinsics(const cv::Mat& K, const cv::Mat& D) : CameraMatrix(K), DistortionMatrix(D) {assert(K.size() == cv::Size(3,3) && D.size() == cv::Size(5,1));}
    /* ctor */				CalibIntrinsics() : CameraMatrix(cv::Mat::zeros(3,3,CV_64F)), DistortionMatrix(cv::Mat::zeros(1,5,CV_64F)) {}
    virtual std::vector<double>	ToVector() const;
    virtual bool			FromVector(const std::vector<double>& vec);
    virtual size_t			GetSize() const {return NumParameters;}
    bool					Write(cv::FileStorage& fn) const;

    cv::Mat					CameraMatrix;
    cv::Mat					DistortionMatrix;

    static size_t			NumParameters;
};

class CalibExtrinsics : CalibVectorisable
{
public:
    /* ctor */				CalibExtrinsics(const cv::Mat& rvec, const cv::Mat& tvec) : Rotation(rvec), Translation(tvec) {}
    /* ctor */				CalibExtrinsics() : Rotation(cv::Mat::eye(3,3,CV_64F)), Translation(cv::Mat::zeros(3,1,CV_64F)) {}
    virtual std::vector<double>	ToVector() const;
    virtual bool			FromVector(const std::vector<double>& vec);
    bool					IsOkay() const {return !Rotation.empty() && !Translation.empty();}
    virtual size_t			GetSize() const {return NumParameters;}
    CalibExtrinsics			GetInverse() const {return CalibExtrinsics(Rotation.t(), -Rotation.t()*Translation);}
    cv::Mat					GetMatrix4x4() const;
    CalibExtrinsics			Concatenate(const CalibExtrinsics& E) const {return CalibExtrinsics(E.Rotation * Rotation, E.Rotation * Translation + E.Translation);}
    CalibExtrinsics&		operator +=(const CalibExtrinsics& E) {return (*this = Concatenate(E));}
    void					Inverse() {*this = GetInverse();}
    bool					Write(cv::FileStorage& fn) const;

    cv::Mat					Rotation;
    cv::Mat					Translation;
    static size_t			NumParameters;
};

typedef std::vector<CalibIntrinsics> CalibIntrinsicsList;
typedef std::vector<CalibExtrinsics> CalibExtrinsicsList;
typedef std::vector<CalibData> CalibDataSet;

class CalibBundleParams : CalibVectorisable
{
public:
    /* ctor */              CalibBundleParams() : m_rectOkay(false) {}
    /* ctor */				CalibBundleParams(size_t numCams, size_t refCamIdx) {Create(numCams, refCamIdx);}
    void                    Create(size_t numCams, size_t refCamIdx);
    virtual std::vector<double>	ToVector() const;
    virtual bool			FromVector(const std::vector<double>& vec);
    virtual size_t			GetSize() const {return Intrinsics.size() * CalibIntrinsics::NumParameters + (Extrinsics.size() - 1 + ImagePoses.size()) * CalibExtrinsics::NumParameters;}
    bool					Store(cv::FileStorage& fn) const;
    bool                    Store(Path& path) const;
    
    ImagePointList			Project(const ObjectPointList& pts3d, size_t camIdx, size_t imgIdx) const;
    ImagePointList			ProjectUndistort(const ObjectPointList& pts3d, size_t camIdx, size_t imgIdx) const;
    ImagePointList			UndistortPoints(const ImagePointList& pts2d, size_t camIdx) const;

    void					InitUndistortRectifyMaps(size_t cam0, size_t cam1, const cv::Size& imageSize);
    cv::Mat					Rectify(size_t camIdx, const cv::Mat& im) const;
    inline bool				RectifyInitialised() const {return m_rectOkay;}
    inline cv::Mat          GetRectR(size_t camIdx) const {return m_rectR[camIdx];}
    inline cv::Mat          GetRectP(size_t camIdx) const {return m_rectP[camIdx];}

    CalibIntrinsicsList		Intrinsics;
    CalibExtrinsicsList		Extrinsics; // extrinsics parameters of cam0, cam1,.. with respect to cam0
    CalibExtrinsicsList		ImagePoses; // poses of calibration data

    static CalibBundleParams NullParams;

protected:
    void					InitMonocularUndistortMap(const cv::Size& imageSize);
    void					InitCollinearUndistortMaps(size_t cam0, size_t cam1, const cv::Size& imageSize);

    bool					m_rectOkay;
    size_t					m_numCams;
    size_t					m_refCamIdx;
    std::vector<cv::Mat>	m_rectMaps1;
    std::vector<cv::Mat>	m_rectMaps2;
    std::vector<cv::Mat>	m_rectR;
    std::vector<cv::Mat>	m_rectP;
};

class Calibratable
{
public:
    /* ctor */			Calibratable() : m_rpe(-1) {};
    /* dtor */ virtual	~Calibratable(){}
    inline double		GetRpe() const {return m_rpe;}
    CalibIndexList		GetImageIndices() const {return m_imageIndcies;}
    inline bool			IsCalibrated() const {return m_rpe >= 0;}
    virtual size_t		GetSize() const = 0;
    virtual double		Calibrate() = 0;

    static cv::TermCriteria	OptimTermCriteria;

protected:
    double				m_rpe;
    CalibIndexList		m_imageIndcies;
    cv::Size			m_imageSize;
};

class CalibCam : public CalibIndexed, public Calibratable
{
public:
    /* ctor */			CalibCam(size_t index =  CalibIndexed::UndefinedIndex) : CalibIndexed(index) {}
    /* ctor */			CalibCam(const cv::FileNode& fn);

    bool				Write(cv::FileStorage& fn) const;
    bool				AddCalibData(CalibData& data);
    virtual	size_t		GetSize() const {return m_data.size();}
    virtual double		Calibrate();

    inline CalibData	GetData(size_t img) const {return m_data[img];}
    inline CalibExtrinsics GetImagePose(size_t img) const {return m_extrinsics[img];}
    inline CalibExtrinsicsList GetImagePoses() const {return m_extrinsics;}
    inline CalibIntrinsics GetIntrinsics() const {return m_intrinsics;}

    friend class		CalibPair;

protected:
    CalibDataSet		m_data;
    CalibIntrinsics		m_intrinsics;
    CalibExtrinsicsList	m_extrinsics;
};

typedef std::vector<CalibCam> CalibCameras;

#endif // CALIBCAM_HPP

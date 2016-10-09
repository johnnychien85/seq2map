#ifndef CALIBPAIR_HPP
#define CALIBPAIR_HPP

#include <calibn/calibcam.hpp>

class CalibPair : public Calibratable
{
public:
	/* ctor */				CalibPair(CalibCam* cam1, CalibCam* cam2);
	virtual size_t			GetSize() const {return m_imageIndcies.size();}
	virtual double			Calibrate();
	bool					IsOkay() const {return GetSize() >= 3;} // needs at least 3 common calibration images
	bool					IsEnabled() const {return m_enabled;}
	CalibCam*				GetCamera1() const {return m_cam1;}
	CalibCam*				GetCamera2() const {return m_cam2;}
	CalibExtrinsics			GetExtrinsics(bool inverse = false) const {return inverse ? m_extrinsics.GetInverse() : m_extrinsics;}
	void					SetEnabled(bool enabled) {m_enabled = enabled;}

protected:
	CalibCam*				m_cam1;
	CalibCam*				m_cam2;
	bool					m_enabled;
	std::vector<ObjectPointList> m_objectPoints;
	std::vector<ImagePointList>	 m_imagePoints1;
	std::vector<ImagePointList>	 m_imagePoints2;
	CalibExtrinsics			m_extrinsics;
	cv::Mat					m_fundamentalMatrix;
	cv::Mat					m_essentialMatrix;
};

typedef std::vector<CalibPair>  CalibPairs;
typedef std::vector<CalibPair*> CalibPairPtrs;

#endif // CALIBPAIR_HPP

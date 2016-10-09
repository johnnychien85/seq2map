#ifndef CALIBREPORT_HPP
#define CALIBREPORT_HPP

#include <fstream>
#include <calibn/calibpair.hpp>
#include <calibn/caliboptim.hpp>

class CalibReport
{
public:
	/* ctor */			CalibReport(const String& reportPath, const String& inputPath);
	/* dtor */ virtual	~CalibReport();
	void				WriteVisibilityGraph(const CalibPairs& pairs, const CalibCameras& cams, const CalibListOfIndexList& seqs);
	void				WriteParams(const CalibBundleParams& params, const CalibBundleParams& sigmas);
	void				WriteOptimisationPlot(const CalibOptimState& state);
	void				WriteRpePlots(const std::vector<std::vector<ImagePointList> >& pts2d);
	void				WriteCoverageImages(const CalibCameras& cams);
	void				WriteFusionImage(const CalibCameras& cams, CalibBundleParams& params);
	void				WriteHessianMatrix(const cv::Mat& H);
	void				WriteRpeImages(const CalibCameras& cams, CalibBundleParams& params);

protected:
	void				WriteHeader();
	void				WriteFooter();

	String			    m_inputPath;
	String			    m_reportPath;
	String		    	m_outImgPath;
	String			    m_outPlotPath;
	std::ofstream		m_stream;

	int					m_fontFace;
	double				m_fontScale;
	cv::Scalar			m_frontColour;
	cv::Scalar			m_backColour;
	cv::Scalar			m_colour1;
	cv::Scalar			m_colour2;
	int					m_markerSize;

	static String		OutImgDirName;
	static String		OutPlotDirName;
};

#endif

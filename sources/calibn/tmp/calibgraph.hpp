#ifndef CALIBGRAPH_HPP
#define CALIBGRAPH_HPP

#include <calibn/calibpair.hpp>
#include <calibn/caliboptim.hpp>
#include <calibn/calibreport.hpp>

class CalibGraph : public CalibOptim
{
public:
	bool                    Create(const CalibCameras& cams, size_t startingCam);
	bool					IsOkay() {return !m_seqs.empty();}
	bool					Initialise(bool pairOptim);
	CalibBundleParams		GetBundleParams() const {return m_params;}
	CalibPairPtrs			GetPairsByCam(size_t cam);

	static CalibPair*		GetPair(CalibPairs& pairs, size_t i, size_t j, size_t cams) {return &pairs[GetPairIndex(i,j,cams)];}
	static const CalibPair*	GetPair(const CalibPairs& pairs, size_t i, size_t j, size_t cams)  {return &pairs[GetPairIndex(i,j,cams)];}
	
protected:
	CalibPair*				GetPair(size_t i, size_t j) {return GetPair(m_pairs, i, j, m_numCams);}
	virtual cv::Mat			MakeIntialGuess() {return cv::Mat(m_params.ToVector()).clone();}
	virtual cv::Mat		    Evaluate(const cv::Mat& x, const Mask& mask) const;
	virtual void			Finalise(const CalibOptimState& state);
	void					BuildRpeData(const cv::Mat& y, std::vector<std::vector<ImagePointList> >& rpeData) const;

	static CalibExtrinsicsList DeriveImagePoses(CalibPairs& pairs, CalibExtrinsicsList& extrinsics);
	static size_t			GetPairIndex(size_t i, size_t j, size_t cams);

	CalibCameras			m_cams;
	size_t					m_startingCamIdx;
	size_t					m_numCams;
	CalibPairs				m_pairs;
	CalibListOfIndexList	m_seqs; // path starting at cam0, cam1,.. to cam0
	CalibBundleParams		m_params;
	//CalibReport&			m_report;
};


#endif // CALIBGRAPH_HPP

#ifndef CALIBDATA_HPP
#define CALIBDATA_HPP

#include <opencv2/opencv.hpp>
#include <calibn/helpers.hpp>

typedef std::vector<cv::Point3f> ObjectPointList;
typedef std::vector<cv::Point2f> ImagePointList;

class CalibPattern
{
public:
	/* ctor */ CalibPattern() {}
	/* ctor */ CalibPattern(cv::Size patternSize, cv::Size2f patternMetric) : PatternSize(patternSize), PatternMetric(patternMetric) {}
    bool FromString(const String& def);
    String ToString() const; 
    bool IsOkay() const { return PatternSize.width > 0 && PatternSize.height > 0 && PatternMetric.width > 0 && PatternMetric.height > 0; }

	cv::Size   PatternSize;
	cv::Size2f PatternMetric;
};

class CalibIndexed
{
public:
	/* ctor */				CalibIndexed(size_t index = UndefinedIndex) : m_index(index) {}
	size_t					GetIndex() const {return m_index;}
	//template<class T> void	Remove(vector<T> list, size_t index);
protected:
	void					SetIndex(size_t index);
	static size_t			UndefinedIndex;
private:
	size_t					m_index;
};

typedef std::vector<size_t> CalibIndexList;
typedef std::vector<CalibIndexList> CalibListOfIndexList;

class CalibData : public CalibIndexed
{
public:
    /* ctor */              CalibData() {}
	/* ctor */				CalibData(const cv::FileNode& fn);
	/* ctor */				//CalibData(size_t index, const cv::Mat& im, const CalibPattern& pattern, std::string filename);
	/* dtor */				virtual ~CalibData() {};

    bool                    FromImage(size_t index, const Path& imagePath, const CalibPattern& pattern);
	inline bool				IsOkay() const  {return m_okay;};
	bool					Store(cv::FileStorage& fs) const;
	bool                    Restore(const cv::FileNode& fn);
	
	inline std::string		GetFileName() const {return m_filename;}
	inline cv::Size			GetImageSize() const {return m_imageSize;}
	inline ImagePointList	GetImagePoints() const {return m_imagePoints;}
	inline ImagePointList	GetCornerPoints() const {return m_cornerPoints;}
	inline ObjectPointList	GetObjectPoints() const {return m_objectPoints;}

	static int				detectionFlags;
	static cv::Size			subPixelWinSize;
	static cv::TermCriteria	subPixelTermCriteria;

private:
	bool					m_okay;
	size_t					m_numPoints;
	String  				m_filename;
	cv::Size				m_imageSize;
	ImagePointList			m_imagePoints;
	ImagePointList			m_cornerPoints;
	ObjectPointList			m_objectPoints;
};

#endif

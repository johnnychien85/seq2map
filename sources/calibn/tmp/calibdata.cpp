#include <stdio.h>
#include <calibn/calibdata.hpp>

size_t	CalibIndexed::UndefinedIndex = (size_t) -1;
int CalibData::detectionFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE;
cv::Size CalibData::subPixelWinSize(11,11);
cv::TermCriteria CalibData::subPixelTermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);

bool CalibPattern::FromString(const String& def)
{
    Strings toks = split(def, 'x');

    if (toks.size() != 3 && toks.size() != 4)
    {
        return false;
    }

    bool success = true;

    success &= sscanf(toks[0].c_str(), "%d", &PatternSize.height)   == 1;
    success &= sscanf(toks[1].c_str(), "%d", &PatternSize.width)    == 1;
    success &= sscanf(toks[2].c_str(), "%f", &PatternMetric.height) == 1;

    if (toks.size() == 3)
    {
        PatternMetric.width = PatternMetric.height;
    }
    else
    {
        success &= sscanf(toks[2].c_str(), "%f", &PatternMetric.width) == 1;
    }

    return success;
}

void CalibIndexed::SetIndex(size_t index)
{
	assert(m_index == UndefinedIndex); // modifying a valid index is not allowed
	m_index = index;
}

bool CalibData::FromImage(size_t index, const Path& imagePath, const CalibPattern& pattern)
{
    cv::Mat im = cv::imread(imagePath.string(), cv::IMREAD_GRAYSCALE);

    if (im.empty())
    {
        E_ERROR << "error reading image from " << imagePath;
        return false;
    }

    SetIndex(index);

	m_okay = false;
	m_filename = imagePath.string();
	m_imageSize = im.size();

	// try to find the corners on the chessboard
	bool found = cv::findChessboardCorners(im, pattern.PatternSize, m_imagePoints, detectionFlags);

	if (!found)
	{
		m_imagePoints.clear();
		return false;
	}

	// refine found corners
	if (subPixelWinSize.height > 0 && subPixelWinSize.width > 0)
	{
		cv::cornerSubPix(im, m_imagePoints, subPixelWinSize, cv::Size(-1,-1), subPixelTermCriteria);
	}

	// how many calibration points?
	m_numPoints = m_imagePoints.size();

	// generating object points
	for (int i = 0; i < pattern.PatternSize.height; i++)
	{
		for (int j = 0; j < pattern.PatternSize.width; j++)
		{
			m_objectPoints.push_back(cv::Point3f(
				(float)((j+1) * pattern.PatternMetric.width),  // X
				(float)((i+1) * pattern.PatternMetric.height), // Y
				0 // Z = 0 (since we have a planar target)
			));

			if ((i == 0 /**************************/ && /*************************/ j == 0) ||
				(i == pattern.PatternSize.height - 1 && /*************************/ j == 0) ||
				(i == 0 /**************************/ && j == pattern.PatternSize.width - 1) ||
				(i == pattern.PatternSize.height - 1 && j == pattern.PatternSize.width - 1))
			{
				size_t idx = m_objectPoints.size() - 1;
				m_cornerPoints.push_back(m_imagePoints[idx]);
			}
		}
	}

	// swap last two corners so they are arranged clockwise
	assert(m_cornerPoints.size() == 4);
	std::iter_swap(m_cornerPoints.end() - 2, m_cornerPoints.end() - 1);

	m_okay = true;
}

bool CalibData::Store(cv::FileStorage& fs) const
{
	try
	{
		fs << "{:";
		fs << "index" << (int) GetIndex();
		fs << "fileName" << m_filename;
		fs << "imageSize" << m_imageSize;
		fs << "imagePoints" << m_imagePoints;
		fs << "cornerPoints" << m_cornerPoints;
		fs << "objectPoints" << m_objectPoints;
		fs << "}";
	}
	catch (std::exception& ex)
	{
	    E_ERROR << "error storing properties";
	    E_ERROR << ex.what();

		return false;
	}

	return true;
}

bool CalibData::Restore(const cv::FileNode& fn)
{
	try
	{
		int index;

		fn["index"] >> index;
		fn["fileName"] >> m_filename;
		fn["imageSize"] >> m_imageSize;
		fn["imagePoints"] >> m_imagePoints;
		fn["cornerPoints"] >> m_cornerPoints;
		fn["objectPoints"] >> m_objectPoints;

		SetIndex((size_t)index);
		
		m_okay = !m_imagePoints.empty() && !m_objectPoints.empty();
	}
	catch (std::exception& ex)
	{
	    E_ERROR << "error restoring properties";
	    E_ERROR << ex.what();

		m_okay = false;
	}

	return m_okay;
}


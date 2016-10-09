#include <calibn/calibpair.hpp>

using namespace cv;

CalibPair::CalibPair(CalibCam* cam1, CalibCam* cam2) : m_enabled(false)
{
	assert(cam1 != NULL && cam2 != NULL && cam1 != cam2);
	assert(cam1->GetIndex() != cam2->GetIndex()); // (u,u) is not a valid pair
	assert(cam1->GetSize() == cam2->GetSize()); // cam1 & cam2 must have the same number of views

	m_cam1 = cam1;
	m_cam2 = cam2;

	size_t numImages = cam1->GetSize();

	for (size_t img = 0; img < numImages; img++)
	{
		CalibData data1 = cam1->GetData(img);
		CalibData data2 = cam2->GetData(img);

		if (img == 0)
		{
			m_imageSize = data1.GetImageSize();
		}

		assert(data1.GetIndex() == img && data2.GetIndex() == img); // data1.GetIndex() == data2.GetIndex() == img
		assert(m_imageSize == data1.GetImageSize() && m_imageSize == data2.GetImageSize());

		if (data1.IsOkay() && data2.IsOkay())
		{
			// TODO: match image points for different object point ordering
			m_objectPoints.push_back(data1.GetObjectPoints());
			m_imagePoints1.push_back(data1.GetImagePoints());
			m_imagePoints2.push_back(data2.GetImagePoints());

			m_imageIndcies.push_back(img);
		}
	}
}

double CalibPair::Calibrate()
{
	if (!IsOkay())
	{
		return -1;
	}

	if (!m_cam1->IsCalibrated()) m_cam1->Calibrate();
	if (!m_cam2->IsCalibrated()) m_cam2->Calibrate();

    m_rpe = cv::stereoCalibrate(m_objectPoints, m_imagePoints1, m_imagePoints2,
		m_cam1->m_intrinsics.CameraMatrix, m_cam1->m_intrinsics.DistortionMatrix,
		m_cam2->m_intrinsics.CameraMatrix, m_cam2->m_intrinsics.DistortionMatrix,
		m_imageSize, m_extrinsics.Rotation, m_extrinsics.Translation,
		m_essentialMatrix, m_fundamentalMatrix, OptimTermCriteria, cv::CALIB_FIX_INTRINSIC);

	return m_rpe;
}

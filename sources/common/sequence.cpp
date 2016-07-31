#include <seq2map\sequence.hpp>

using namespace cv;
using namespace seq2map;

void Pose::Transform(Points3F& points) const
{

}

void Camera::World2Image(const Points3F& worldPts, Points2F& imagePts) const
{
    Points3F cameraPts;

    World2Camera(worldPts, cameraPts);
    Camera2Image(cameraPts, imagePts);
}

void PinholeCamera::Camera2Image(const Points3F& cameraPts, Points2F& imagePts) const
{
    cv::projectPoints(cameraPts, Mat(), Mat(), m_cameraMatrix, m_distortionCoeffs, imagePts);
}
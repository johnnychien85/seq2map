#ifndef SEQUENCE_HPP
#define SEQUENCE_HPP

#include <seq2map\common.hpp>

namespace seq2map
{
    class Frame
    {
    public:
    protected:
        const Path& m_imageFile;
        const Path& m_featureFile;
    };

    class Pose
    {
    public:
        void Transform(Points3F& points) const;
    };

    class Camera
    {
    public:
        typedef cv::Ptr<Camera> Ptr;

        void World2Image(const Points3F& worldPts, Points2F& imagePts) const;
        inline void World2Camera(const Points3F& worldPts, Points3F& cameraPts) const { m_extrinsics.Transform(cameraPts = worldPts); };
        virtual void Camera2Image(const Points3F& cameraPts, Points2F& imagePts) const = 0;

    protected:
        Pose m_extrinsics;
    };

    class PinholeCamera : public Camera
    {
    public:
        virtual void Camera2Image(const Points3F& cameraPts, Points2F& imagePts) const;
    protected:
        cv::Mat m_cameraMatrix;
        cv::Mat m_distortionCoeffs;
    };

    class Sequence
    {
    protected:
        std::vector<Camera::Ptr> m_cameras;
    };
}
#endif // SEQUENCE_HPP

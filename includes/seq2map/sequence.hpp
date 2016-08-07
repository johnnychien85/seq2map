#ifndef SEQUENCE_HPP
#define SEQUENCE_HPP

#include <seq2map\common.hpp>
#include <seq2map\features.hpp>

namespace seq2map
{
    class Pose : public Persistent<cv::FileStorage, cv::FileNode>
    {
    public:
        Pose() :
          m_matrix(cv::Mat::eye(3, 4, CV_32F)),
          m_rvec(cv::Mat::zeros(3, 1, CV_32F)),
          m_rmat(m_matrix.rowRange(0, 3).colRange(0, 3)),
          m_tvec(m_matrix.rowRange(0, 3).colRange(3, 4)) {}

        void Transform(Points3F& points) const;
        bool SetRotationMatrix(const cv::Mat& rmat);
        bool SetRotationVector(const cv::Mat& rvec);
        bool SetTranslation(const cv::Mat& tvec);
        bool SetMatrix(const cv::Mat& matrix);
        cv::Mat GetMatrix(bool squareMatrix = false, bool preMult = true) const;
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);
    protected:
        cv::Mat m_matrix;
        cv::Mat m_rmat;
        cv::Mat m_rvec;
        cv::Mat m_tvec;
    };

    class Frame
    {
    public:
        size_t idx;
        cv::Mat im;
        ImageFeatureSet features;
    protected:
        Frame(size_t idx = INVALID_INDEX) : idx(idx) {};
        friend class Camera;
    };

    class Camera
    {
    public:
        class Intrinsics : public Persistent<cv::FileStorage, cv::FileNode>
        {
        public:
            typedef boost::shared_ptr<Intrinsics> Ptr;
            virtual void Project(const Points3F& pts3d, Points2F& pts2d) const = 0;
            virtual bool Store(cv::FileStorage& fs) const = 0;
            virtual bool Restore(const cv::FileNode& fn) = 0;
            virtual String GetModelName() const = 0;
        };

        class IntrinsicsFactory : public Factory<String, Intrinsics>
        {
        public:
            IntrinsicsFactory();
            virtual ~IntrinsicsFactory() {}
        };

        typedef SequentialPersistentLoader<PersistentImage> ImageStorage;
        typedef SequentialPersistentLoader<ImageFeatureSet> FeatureStorage;

        inline void SetName(const String& name) { m_cameraName = name; }
        inline void SetIntrinsics(const Intrinsics::Ptr& intrinsics) { m_intrinsics = intrinsics; }
        inline void SetImageSize(const cv::Size& imageSize) { m_imageSize = imageSize; }
        inline String GetName() const { return m_cameraName; }
        inline Pose& GetExtrinsics() { return m_extrinsics; }
        inline size_t GetFrames() const { return m_imageStore.GetSize(); }
        inline ImageStorage& GetImageStorage() { return m_imageStore; }
        inline FeatureStorage& GetFeatureStorage() { return m_featureStore; }
        Frame GetFrame(size_t idx) const;
        inline Frame operator[](size_t idx) const { return GetFrame(idx); }
        inline void World2Camera(const Points3F& worldPts, Points3F& cameraPts) const { m_extrinsics.Transform(cameraPts = worldPts); };
        virtual void Camera2Image(const Points3F& cameraPts, Points2F& imagePts) const;
        void World2Image(const Points3F& worldPts, Points2F& imagePts) const;
    protected:
        String          m_cameraName;
        Intrinsics::Ptr m_intrinsics;
        Pose            m_extrinsics;
        cv::Size        m_imageSize;
        ImageStorage    m_imageStore;
        FeatureStorage  m_featureStore;

        friend class Sequence;
    };

    typedef std::vector<Camera> Cameras;

    class Sequence : public Persistent<Path>
    {
    public:
        Sequence() {}

        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path);
        inline void SetPath(const Path& path) { m_seqPath = path; }
        inline void SetName(const String& name) { m_seqName = name; }
        inline void SetGrabber(const String& grabber) { m_grabberName = grabber; }
        inline void SetFeatureDetextractor(const FeatureDetextractorPtr& dxtor) { m_featureDxtor = dxtor;  }
        void Clear();
        inline const Camera& GetCamera(size_t idx)  const { return m_cameras[idx]; }
        inline const Camera& operator[](size_t idx) const { return GetCamera(idx); }
        inline Cameras& GetCameras() { return m_cameras; }
    protected:
        //static void WriteFileList(cv::FileStorage& fs, const String& node, const Path& root, const Strings& filenames);
        //static void ReadFileList

        Path m_seqPath;
        String m_seqName;
        String m_grabberName;
        Cameras m_cameras;
        FeatureDetextractorPtr m_featureDxtor;
    };

    class BouguetIntrinsics : public Camera::Intrinsics
    {
    public:
        typedef boost::shared_ptr<BouguetIntrinsics> Ptr;

        BouguetIntrinsics(const cv::Mat& cameraMatrix = s_canonical.m_cameraMatrix, const cv::Mat& distCoeffs = s_canonical.m_distCoeffs);
        virtual void Project(const Points3F& pts3d, Points2F& pts2d) const;
        bool SetCameraMatrix(const cv::Mat& cameraMatrix);
        bool SetDistCoeffs(const cv::Mat& distCoeffs);
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);
        virtual String GetModelName() const { return "BOUGUET";  }
    protected:
        static const BouguetIntrinsics s_canonical;
        cv::Mat m_projMatrix;
        cv::Mat m_cameraMatrix;
        cv::Mat m_distCoeffs;
    };

    //class BouguetCamera : public PinholeCamera
    //{
    //};
}
#endif // SEQUENCE_HPP

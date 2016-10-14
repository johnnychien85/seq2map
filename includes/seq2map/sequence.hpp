#ifndef SEQUENCE_HPP
#define SEQUENCE_HPP

#include <seq2map/common.hpp>
#include <seq2map/features.hpp>
#include <seq2map/geometry.hpp>

namespace seq2map
{
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
        typedef SequentialPersistentLoader<PersistentImage> ImageStorage;
        typedef SequentialPersistentLoader<ImageFeatureSet> FeatureStorage;

        inline void SetName(const String& name) { m_cameraName = name; }
        inline void SetIntrinsics(const ProjectionModel::Ptr& intrinsics) { m_intrinsics = intrinsics; }
        inline void SetImageSize(const cv::Size& imageSize) { m_imageSize = imageSize; }
        inline String GetName() const { return m_cameraName; }
        inline const ProjectionModel::Ptr GetIntrinsics() const { return m_intrinsics; }
        inline EuclideanTransform& GetExtrinsics() { return m_extrinsics; }
        inline size_t GetFrames() const { return m_imageStore.GetSize(); }
        inline ImageStorage& GetImageStorage() { return m_imageStore; }
        inline FeatureStorage& GetFeatureStorage() { return m_featureStore; }
        Frame GetFrame(size_t idx) const;
        inline Frame operator[](size_t idx) const { return GetFrame(idx); }
        inline void World2Camera(const Points3D& worldPts, Points3D& cameraPts) const { m_extrinsics.Apply(cameraPts = worldPts); };
        virtual void Camera2Image(const Points3D& cameraPts, Points2D& imagePts) const;
        void World2Image(const Points3D& worldPts, Points2D& imagePts) const;

        class IntrinsicsFactory : public Factory<String, ProjectionModel>
        {
        public:
            IntrinsicsFactory();
            virtual ~IntrinsicsFactory() {}
        };
    protected:
        String               m_cameraName;
        ProjectionModel::Ptr m_intrinsics;
        EuclideanTransform   m_extrinsics;
        cv::Size             m_imageSize;
        ImageStorage         m_imageStore;
        FeatureStorage       m_featureStore;

        friend class Sequence;
    };

    typedef std::vector<Camera> Cameras;

    class Sequence : public Persistent<Path>
    {
    public:
        Sequence() {}
        virtual ~Sequence() {}
        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path);
        inline void SetPath(const Path& path) { m_seqPath = path; }
        inline void SetName(const String& name) { m_seqName = name; }
        inline void SetGrabber(const String& grabber) { m_grabberName = grabber; }
        inline void SetFeatureDetextractor(const FeatureDetextractorPtr& dxtor) { m_featureDxtor = dxtor;  }
        inline FeatureDetextractorPtr GetFeatureDetextractor() const { return m_featureDxtor; }
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
}
#endif // SEQUENCE_HPP

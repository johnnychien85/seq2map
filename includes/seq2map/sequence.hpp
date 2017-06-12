#ifndef SEQUENCE_HPP
#define SEQUENCE_HPP

#include <boost/uuid/uuid.hpp>
#include <seq2map/common.hpp>
#include <seq2map/features.hpp>
#include <seq2map/disparity.hpp>
#include <seq2map/geometry.hpp>
#include <seq2map/seq_file_store.hpp>

namespace seq2map
{
	/**
	 * ...
	 */
	class UUID
	{
	public:
        UUID();
        UUID(const boost::uuids::uuid& uuid) : m_uuid(uuid) {}

		String ToString() const;
		bool FromString(const String& uuid);

        static UUID Generate(const String& seed);

	private:
		boost::uuids::uuid m_uuid;
	};

	/**
	 * An entity has a name and a global unique identifier.
	 */
	class Entity : public Persistent<cv::FileStorage, cv::FileNode>
	{
	public:
		inline void SetName(const String& name) { m_name = name; }
		inline String GetName() const { return m_name; }
		inline UUID& GetUUID() { return m_uuid; }
		virtual bool Store(cv::FileStorage& fn) const;
		virtual bool Restore(const cv::FileNode& fs);
	private:
		String m_name;
		UUID   m_uuid;
	};

    /**
     * A sensor object represents a physical sensor installed.
     * The extrinsic parameters specify the geometric relationship
     * between the sensor and a reference frame.
     */
	class Sensor : public Entity
	{
	public:
        inline String GetModel() const                   { return m_model;      }
        inline void SetModel(const String& model)        { m_model = model;     }

        inline EuclideanTransform  GetExtrinsics() const { return m_extrinsics; }
		inline EuclideanTransform& GetExtrinsics()       { return m_extrinsics; }
		inline void SetExtrinsics(const EuclideanTransform& tform) { m_extrinsics = tform; }

        virtual bool Store(cv::FileStorage& fs) const;
		virtual bool Restore(const cv::FileNode& fn);

    private:
        String m_model;
        EuclideanTransform m_extrinsics;
	};

    /**
     * A wrapper of cv::Mat with disk storage backend.
     */
    class PersistentImage : public Persistent<Path>
    {
    public:
        virtual bool Store(Path& path) const { return cv::imwrite(path.string(), im); };
        virtual bool Restore(const Path& path) { return !(im = cv::imread(path.string())).empty(); };

        cv::Mat im;
    };

    /**
     * Sequence of images stored in the same folder.
     */
    typedef SequentialFileStore<PersistentImage> ImageStore;

    /**
     * A camera instance provides an image sequence in the same image size.
     */
    class Camera
    : public Sensor,
      public IndexReferenced<Camera>
    {
    public:
        /**
         * The intrinsic factory used to restore a camera from file node.
         */
        class IntrinsicsFactory
        : public Factory<String, ProjectionModel>,
          public Singleton<IntrinsicsFactory>
        {
        public:
            friend class Singleton<IntrinsicsFactory>;
        protected:
            virtual void Init();
        };

        //
        // Constructor and desctructor
        //
        Camera(size_t index) : IndexReferenced(index) {}
        virtual ~Camera() {}

        //
        // Accessors
        //
        inline void     SetImageSize(const cv::Size& imageSize)     { m_imageSize = imageSize; }
        inline cv::Size GetImageSize() const                        { return m_imageSize; }
        inline void SetIntrinsics(ProjectionModel::Own& intrinsics) { m_intrinsics = intrinsics; }
        inline ProjectionModel::ConstOwn GetIntrinsics() const      { return m_intrinsics; }
        inline ProjectionModel::Own      GetIntrinsics()            { return m_intrinsics; }
        ImageStore&       GetImageStore()                           { return m_imageStore; }
        const ImageStore& GetImageStore() const                     { return m_imageStore; }
        inline size_t GetFrames() const                             { return m_imageStore.GetItems(); }

        //inline void World2Camera(const Points3D& worldPts, Points3D& cameraPts) const { GetExtrinsics().Apply(cameraPts = worldPts); };
        //void Camera2Image(const Points3D& cameraPts, Points2D& imagePts) const;
        //void World2Image(const Points3D& worldPts, Points2D& imagePts) const;

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fn) const;
        virtual bool Restore(const cv::FileNode& fs);

    private:
        ProjectionModel::Own m_intrinsics;
        cv::Size             m_imageSize;
        ImageStore           m_imageStore;
    };

    /**
     * Rectified stereo camera pair.
     */
    class RectifiedStereo
    : public Referenced<RectifiedStereo>
    {
    public:
        struct Less
        {
            bool operator() (const RectifiedStereo::Own& lhs, const RectifiedStereo::Own& rhs)
            {
                Camera::ConstOwn lcam0 = lhs->GetPrimaryCamera();
                Camera::ConstOwn lcam1 = lhs->GetSecondaryCamera();
                Camera::ConstOwn rcam0 = rhs->GetPrimaryCamera();
                Camera::ConstOwn rcam1 = rhs->GetSecondaryCamera();

                return lcam0 && lcam1 && rcam0 && rcam1 && (*lcam0 < *rcam0 || *lcam1 < *rcam1);
            }
        };

        typedef std::set<RectifiedStereo::Own, Less> Set;

        /**
         * Geometric configuration of the recrified cameras.
         */
        enum Geometry
        {
            LEFT_RIGHT,   // Classical lateral configuration
            TOP_DOWN,     // Vertically arranged configuration
            BACK_FORWARD, // Polar rectified configuration
            UNKNOWN
        };

        //
        // Constructor and destructor
        //
        RectifiedStereo() : m_config(UNKNOWN), m_baseline(0) {}
        RectifiedStereo(Camera::ConstOwn& cam0, Camera::ConstOwn& cam1) { Create(cam0, cam1); }
        virtual ~RectifiedStereo() {}

        //
        // Creation
        //
        bool Create(Camera::ConstOwn& cam0, Camera::ConstOwn& cam1);

        static Own Create(Camera::Own& cam0, Camera::Own& cam1) { return Own(new RectifiedStereo(Camera::ConstOwn(cam0), Camera::ConstOwn(cam1))); }

        //
        // Accessor
        //
        inline Camera::ConstOwn GetPrimaryCamera()   const { return m_priCam; }
        inline Camera::ConstOwn GetSecondaryCamera() const { return m_secCam; }
        inline bool IsOkay() const { return m_priCam && m_secCam; }

        //
        // Misc.
        //
        Geometry Backproject(const cv::Mat& dp) const;
        String ToString() const;

    private:
        friend class Sequence; // for restoring camera references

        Camera::ConstOwn m_priCam; // reference to the primary camera
        Camera::ConstOwn m_secCam; // reference to the secondary camera

        Geometry m_config; // geometric configuration of the stereo pair
        double m_baseline; // length of baseline
    };

    /**
     * Sequence of image feature sets stored in the same folder.
     */
    class FeatureStore
    : public SequentialFileStore<ImageFeatureSet>,
      public IndexReferenced<FeatureStore>
    {
    public:
        //
        // Constructor and destructor
        //
        FeatureStore(size_t index) : IndexReferenced(index) {}
        virtual ~FeatureStore() {}

        //
        // Creation
        //
        bool Create(const Path& root, Camera::ConstOwn& camera, FeatureDetextractor::Own& dxtor);

        //
        // Accessor
        //
        inline Camera::ConstOwn GetCamera() const { return m_cam; }
        inline FeatureDetextractor::Own      GetFeatureDetextractor()       { return m_dxtor; }
        inline FeatureDetextractor::ConstOwn GetFeatureDetextractor() const { return m_dxtor; }

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

    private:
        friend class Sequence; // for restoring camera reference

        Camera::ConstOwn m_cam;
        FeatureDetextractor::Own m_dxtor;
    };

    typedef std::vector<FeatureStore> FeatureStores;

    /**
     * Disparity map store.
     */
    class DisparityStore
    : public SequentialFileStore<PersistentImage>,
      public IndexReferenced<DisparityStore>
    {
    public:
        //
        // Constructor and destructor
        //
        DisparityStore(size_t index) : m_dspace(0, 64, 64*16), IndexReferenced(index) { UpdateMappings(); }
        virtual ~DisparityStore() {}

        //
        // Creation
        //
        bool Create(const Path& root, RectifiedStereo::ConstOwn& stereo, StereoMatcher::ConstOwn& matcher);

        //
        // Accessor
        //
        inline RectifiedStereo::ConstOwn GetStereoPair() const { return m_stereo; }
        inline StereoMatcher::ConstOwn GetMatcher() const { return m_matcher; }

        //
        // Persistence
        //
        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        using SequentialFileStore<PersistentImage>::Append;

    protected:
        virtual bool Append(Path& to, const PersistentImage& dpm) const;
        virtual bool Retrieve(const Path& from, PersistentImage& dpm) const;

    private:
        friend class Sequence; // for restoring camera reference

        struct LinearMapping
        {
            double alpha;
            double beta;
        };

        void UpdateMappings();

        static LinearSpacedVec<double> s_dspace16U;

        StereoMatcher::DisparitySpace m_dspace;
        RectifiedStereo::ConstOwn m_stereo;
        StereoMatcher::ConstOwn   m_matcher;

        LinearMapping m_dspaceTo16U;
        LinearMapping m_dspaceTo32F;
    };

    /**
     * Sequence class contains information of a recorded image sequence,
     * including sensor profiles and the storage of sensory data.
     */
    class Sequence : public Persistent<Path>
    {
    public:
        /**
         * A builder constructs sequence from raw data.
         */
        class Builder : public Parameterised
        {
        public:
            typedef boost::shared_ptr<Builder> Ptr;

            virtual void WriteParams(cv::FileStorage& fs) const = 0;
            virtual bool ReadParams(const cv::FileNode& fn) = 0;
            virtual void ApplyParams() = 0;
            virtual Options GetOptions(int flag = 0) = 0;
            
            virtual bool Build(const Path& from, const String& name, const String& grabber, Sequence& seq) const;

        protected:
            virtual String GetVehicleName(const Path& from) const = 0;
            virtual bool BuildCamera(const Path& from, Camera::Map& cams, RectifiedStereo::Set& stereo) const = 0;
        };

        //
        // Constructor and destructor
        //
        Sequence() { Clear(); }
        virtual ~Sequence() {}

        //
        // Accessor
        //
        inline const Path GetRawPath()   const { return m_rawPath; }
        inline const Path GetRootPath()  const { return m_seqPath; }
        inline const String GetName()    const { return m_seqName; }
        inline const String GetVehicle() const { return m_vehicleName; }
        inline const String GetGrabber() const { return m_grabberName; }

        inline Path GetFeatureStoreRoot() const { return m_seqPath / m_kptsDirName; }
        inline Path GetDisparityStoreRoot() const { return m_seqPath / m_dispDirName; }
        inline size_t GetFrames() const { return m_cameras.size() > 0 && m_cameras.begin()->second ? m_cameras.begin()->second->GetFrames() : 0; }

        inline const Camera::Map& GetCameras()                 const { return m_cameras;    }
        inline const RectifiedStereo::Set& GetStereoPairs()    const { return m_stereo;     }
        inline const FeatureStore::Map& GetFeatureStores()     const { return m_kptsStores; }
        inline const DisparityStore::Map& GetDisparityStores() const { return m_dispStores; }

        RectifiedStereo::ConstOwn FindStereoPair(size_t priCamIdx, size_t secCamIdx) const;

        //
        // Persistence
        //
        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path);

    private:
        void Clear();
        size_t ScanStores();

        Path   m_rawPath;
        Path   m_seqPath;

        String m_seqName;
        String m_vehicleName;
        String m_grabberName;
        String m_kptsDirName;
        String m_dispDirName;
        String m_mapsDirName;

        Camera::Map          m_cameras;
        RectifiedStereo::Set m_stereo;
        FeatureStore::Map    m_kptsStores;
        DisparityStore::Map  m_dispStores;

        static String s_storeIndexFileName;
        static String s_featureStoreDirName;
        static String s_disparityStoreDirName;
        static String s_mapStoreDirName;
    };
}
#endif // SEQUENCE_HPP

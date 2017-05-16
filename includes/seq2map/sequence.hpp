#ifndef SEQUENCE_HPP
#define SEQUENCE_HPP

#include <boost/uuid/uuid.hpp>
#include <seq2map/common.hpp>
#include <seq2map/features.hpp>
#include <seq2map/disparity.hpp>
#include <seq2map/geometry.hpp>

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
     * A data source returns data given a specific frame.
     */
    /*
    template <typename T>
    class DataSource
    {
    public:
        virtual bool GetData(size_t frame, T& data) = 0;
    };
    */

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
     *
     */
    /* class SequentialFileNameGenerator
    {
    public:
        SequentialFilenameGenerator(size_t start = 0, size_t digit = 8);
        String Next();
    private:
        String m_extension;
        size_t m_counter;
    }; */

    /**
     * 
     */
    template <typename T>
    class SequentialFileStore
    : public Persistent<cv::FileStorage, cv::FileNode>,
      public Persistent<Path>
    {
    public:
        bool Create(const Path& root, size_t allocated = 128)
        {
            if (!makeOutDir(root))
            {
                E_ERROR << "error creating directory " << root;
                return false;
            }

            m_root = root;
            m_filenames.clear();
            m_filenames.reserve(allocated);

            return true;
        }

        bool Create(const Path& root, const Strings& filenames)
        {
            if (!Create(root, filenames.size()))
            {
                return false;
            }

            m_filenames = filenames;

            return true;
        }

        void FromExistingFiles(const Path& root, const String& ext = "")
        {
            Paths files = enumerateFiles(root, ext);
            Strings filenames;

            BOOST_FOREACH (const Path& file, files)
            {
                filenames.push_back(file.filename().string());
            }

            Create(root, filenames);
        }

        bool Append(const String& filename, const T& data)
        {
            // TODO: duplication check
            // ...
            // ..
            // .

            Path to = m_root / filename;

            if (!Append(to, data))
            {
                E_ERROR << "error storing to " << filename;
                return false;
            }

            m_filenames.push_back(filename);

            return true;
        }

        bool Retrieve(size_t idx, T& data) const
        {
            if (idx >= m_filenames.size())
            {
                E_ERROR << "index out of bound (index=" << idx << ", size=" << m_filenames.size() << ")";
                return false;
            }

            return Retrieve(m_root / m_filenames[idx], data);
        }

        T operator[] (size_t idx) const
        {
            T data;

            if (!Retrieve(idx, data))
            {
                E_ERROR << "error retrieving data, index=" << idx;
                return T();
            }

            return data;
        }

        inline size_t GetItems() const
        {
            return m_filenames.size();
        }

        inline Path GetRoot() const
        {
            return m_root;
        }

        inline Path GetItemPath(size_t idx) const
        {
            return m_root / m_filenames[idx];
        }

        inline const Strings& GetFileNames() const
        {
            return m_filenames;
        }

        //...
        virtual bool Store(cv::FileStorage& fs) const
        {
            try
            {
                fs << "root"  << m_root.string();
                fs << "items" << m_filenames.size();
                fs << "files" << "[";
                BOOST_FOREACH(const String& filename, m_filenames)
                {
                    fs << filename;
                }
                fs << "]";
            }
            catch (std::exception& ex)
            {
                E_ERROR << "error storing sequential file store";
                E_ERROR << ex.what();

                return false;
            }

            return true;
        }

        virtual bool Restore(const cv::FileNode& fn)
        {
            m_filenames.clear();

            try
            {
                cv::FileNode files = fn["files"];
                String root;
                size_t items = 0;

                fn["root"]  >> root;
                fn["items"] >> items;

                if (!Create(root, items))
                {
                    E_ERROR << "error creating store " << root;
                    return false;
                }

                for (cv::FileNodeIterator itr = files.begin(); itr != files.end(); itr++)
                {
                    m_filenames.push_back((String)*itr);
                }

                if (items != m_filenames.size())
                {
                    E_WARNING << "the number of items " << items << " does not agree with file list size " << m_filenames.size();
                    E_WARNING << "possible file corruption";
                }
            }
            catch (std::exception& ex)
            {
                E_ERROR << "error restoring sequential file store";
                E_ERROR << ex.what();

                return false;
            }

            return true;
        }

        virtual bool Store(Path& to) const
        {
            cv::FileStorage fs(to.string(), cv::FileStorage::WRITE);

            if (!fs.isOpened())
            {
                E_ERROR << "error opening file " << to << " for writing";
                return false;
            }

            return Store(fs);
        }

        virtual bool Restore(const Path& from)
        {
            cv::FileStorage fs(from.string(), cv::FileStorage::READ);

            if (!fs.isOpened())
            {
                E_ERROR << "error opening file " << from << " for reading";
                return false;
            }

            return Restore(fs.root());
        }

    protected:
        virtual bool Append(Path& to, const T& data) const     { return data.Store(to);     }
        virtual bool Retrieve(const Path& from, T& data) const { return data.Restore(from); }

    private:
        Path    m_root;
        Strings m_filenames;
    };

    typedef SequentialFileStore<PersistentImage> ImageStore;

    /**
     *
     */
    class FeatureStore
    : public SequentialFileStore<ImageFeatureSet>,
      public Indexed
    {
    public:
        FeatureStore() : m_camIdx(INVALID_INDEX) {}

        bool Create(const Path& root, size_t cam, FeatureDetextractor::Ptr dxtor);
        inline size_t GetCameraIndex() const { return m_camIdx; }
        inline FeatureDetextractor::Ptr GetFeatureDetextractor()            { return m_dxtor; }
        inline FeatureDetextractor::ConstPtr GetFeatureDetextractor() const { return m_dxtor; }

        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

    private:
        size_t m_camIdx;
        FeatureDetextractor::Ptr m_dxtor;
    };

    typedef std::vector<FeatureStore> FeatureStores;

    /**
     * Disparity map store.
     */
    class DisparityStore
    : public SequentialFileStore<PersistentImage>,
      public Indexed
    {
    public:
        DisparityStore() : m_dspace(0, 64, 64*16) { UpdateMappings(); }

        bool Create(const Path& root, size_t priCamIdx, size_t secCamIdx, StereoMatcher::Ptr matcher);

        virtual bool Store(cv::FileStorage& fs) const;
        virtual bool Restore(const cv::FileNode& fn);

        inline size_t GetPrimaryCameraIndex()   const { return m_priCamIdx; }
        inline size_t GetSecondaryCameraIndex() const { return m_secCamIdx; }

        using SequentialFileStore<PersistentImage>::Append;

    protected:
        virtual bool Append(Path& to, const PersistentImage& dpm) const;
        virtual bool Retrieve(const Path& from, PersistentImage& dpm) const;

    private:
        struct LinearMapping
        {
            double alpha;
            double beta;
        };

        void UpdateMappings();

        static LinearSpacedVec<double> s_dspace16U;

        size_t m_priCamIdx;
        size_t m_secCamIdx;
        StereoMatcher::DisparitySpace m_dspace;
        StereoMatcher::Ptr m_matcher;

        LinearMapping m_dspaceTo16U;
        LinearMapping m_dspaceTo32F;
    };

    typedef std::vector<DisparityStore> DisparityStores;

    /**
     * .....
     */
    class Camera
    : public Sensor, public Indexed
    {
    public:
        typedef boost::shared_ptr<Camera> Ptr;
        typedef boost::shared_ptr<const Camera> ConstPtr;

        class IntrinsicsFactory
        : public Factory<String, ProjectionModel>,
          public Singleton<IntrinsicsFactory>
        {
            friend class Singleton<IntrinsicsFactory>;
        protected:
            virtual void Init();
        };

        virtual bool Store(cv::FileStorage& fn) const;
        virtual bool Restore(const cv::FileNode& fs);

        inline void SetIntrinsics(const ProjectionModel::Ptr& intrinsics) { m_intrinsics = intrinsics; }
        inline ProjectionModel::ConstPtr GetIntrinsics() const { return m_intrinsics; }
        inline ProjectionModel::Ptr      GetIntrinsics()       { return m_intrinsics; }
        
        inline void SetImageSize(const cv::Size& imageSize) { m_imageSize = imageSize; }
        inline cv::Size GetImageSize() const { return m_imageSize; }

        inline size_t GetFrames() const { return m_imageStore.GetItems(); }

        ImageStore&          GetImageStore()                   { return m_imageStore; }
        const ImageStore&    GetImageStore() const             { return m_imageStore; }
        const FeatureStore&  GetFeatureStore(size_t idx) const { return m_featureStores[idx]; }
        const FeatureStores& GetFeatureStores() const          { return m_featureStores; }

        inline void World2Camera(const Points3D& worldPts, Points3D& cameraPts) const { GetExtrinsics().Apply(cameraPts = worldPts); };
        void Camera2Image(const Points3D& cameraPts, Points2D& imagePts) const;
        void World2Image(const Points3D& worldPts, Points2D& imagePts) const;

    private:
        ProjectionModel::Ptr m_intrinsics;
        cv::Size             m_imageSize;
        ImageStore           m_imageStore;
        FeatureStores        m_featureStores;

        friend class Sequence; // Sequence::Restore needs this to bind a scanned feature store
    };

    typedef std::vector<Camera> Cameras;

    /**
     * .....
     */
    class RectifiedStereo /* : public Sensor */
    {
    public:
        RectifiedStereo() : m_lateral(true), m_baseline(0), m_activeStore(0) {}
        RectifiedStereo(const Camera::ConstPtr& primary, const Camera::ConstPtr& secondary) { Create(primary, secondary); }
        RectifiedStereo(const Camera& primary, const Camera& secondary);

        bool Create(const Camera::ConstPtr& primary, const Camera::ConstPtr& secondary);

        String ToString() const;

        //virtual bool Store(cv::FileStorage& fn) const;
        //virtual bool Restore(const cv::FileNode& fs);

        bool SetActiveStore(size_t store);
        const DisparityStore& GetDisparityStore(size_t idx) const { return m_stores[idx]; }
        const DisparityStores& GetDisparityStores() const { return m_stores; }

        cv::Mat GetDepthMap(size_t frame, size_t store) const;
        inline cv::Mat GetDepthMap(size_t frame) const { return GetDepthMap(frame, m_activeStore); }

        inline Camera::ConstPtr GetPrimaryCamera()   const { return m_primary;   }
        inline Camera::ConstPtr GetSecondaryCamera() const { return m_secondary; }

    private:
        Camera::ConstPtr m_primary;
        Camera::ConstPtr m_secondary;

        bool m_lateral; // true: left-right configuration, otherwise top-bottom
        double m_baseline;

        DisparityStores m_stores;
        size_t m_activeStore;

        friend class Sequence; // Sequence::Restore needs this to bind a scanned disparity store
    };

    typedef std::vector<RectifiedStereo> RectifiedStereoPairs;

    /**
     * Sequence class contains information of a recorded image sequence,
     * including sensor profiles and the storage of sensory data.
     */
    class Sequence : public Persistent<Path>
    {
    public:
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
            virtual bool BuildCamera(const Path& from, Cameras& cams, RectifiedStereoPairs& stereo) const = 0;
        };

        Sequence() { Clear(); }
        virtual ~Sequence() {}

        virtual bool Store(Path& path) const;
        virtual bool Restore(const Path& path);

        inline const Path GetRawPath()   const { return m_rawPath; }
        inline const Path GetRootPath()  const { return m_seqPath; }
        inline const String GetName()    const { return m_seqName; }
        inline const String GetVehicle() const { return m_vehicleName; }
        inline const String GetGrabber() const { return m_grabberName; }

        inline const Camera& GetCamera(size_t idx) const { return m_cameras[idx]; }
        inline const Cameras& GetCameras() const { return m_cameras; }
        inline const RectifiedStereoPairs GetRectifiedStereo() const { return m_stereo; }
        inline Path GetFeatureStoreRoot() const { return m_seqPath / m_kptsDirName; }
        inline Path GetDisparityStoreRoot() const { return m_seqPath / m_dispDirName; }
        inline size_t GetFrames() const { return m_cameras.size() > 0 ? m_cameras[0].GetFrames() : 0; }

        bool FindFeatureStore(size_t index, FeatureStore const* &store) const;

    private:
        void Clear();
        size_t ScanStores();

        Path                 m_rawPath;
        Path                 m_seqPath;
        String               m_seqName;
        String               m_vehicleName;
        String               m_grabberName;
        String               m_kptsDirName;
        String               m_dispDirName;
        String               m_mapsDirName;
        Cameras              m_cameras;
        RectifiedStereoPairs m_stereo;

        static String        s_storeIndexFileName;
        static String        s_featureStoreDirName;
        static String        s_disparityStoreDirName;
        static String        s_mapStoreDirName;
    };
}
#endif // SEQUENCE_HPP

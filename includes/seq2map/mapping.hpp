#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <seq2map/sequence.hpp>
#include <seq2map/sparse_node.hpp>
#include <seq2map/geometry_problems.hpp>

namespace seq2map
{
    class Landmark;
    class Frame;
    class Source;

    /**
     * Observation of a landmark
     */
    struct Hit
    {
    public:
        const size_t index; // originating feature index in the store
        Point2D proj;       // observed 2D image coordinates

        Hit& operator= (const Hit& hit)
        {
            proj = hit.proj;
            return *this;
        }

    protected:
        friend class Landmark; // only landmarks can make hits
        Hit(size_t index) : index(index) {}
    };

    typedef NodeND<Hit, 3> HitNode;

    /**
     * A landmark is an observed and tracked distinctive scene point.
     */
    class Landmark : public HitNode::DimensionZero<>
    {
    public:
        typedef std::vector<Landmark*> Ptrs;
        struct Covar3D
        {
            Covar3D() : xx(0), xy(0), xz(0), yy(0), yz(0), zz(0) {}

            double xx;
            double xy;
            double xz;
            double yy;
            double yz;
            double zz;
        };

        Hit& Hit(Frame& frame, Source& src, size_t index);

        Point3D position;
        Covar3D cov;

    protected:
        friend class Map3<seq2map::Hit, Landmark, Frame, Source>;
        Landmark(size_t index = INVALID_INDEX) : DimensionZero(index) {}
    };

    /**
     * A frame represent the state at a point of time.
     */
    class Frame : public HitNode::Dimension<1>
    {
    public:
        std::map<size_t, Landmark::Ptrs> featureLandmarkLookup;
        PoseEstimator::Estimate poseEstimate;

    protected:
        friend class Map3<Hit, Landmark, Frame, Source>;
        Frame(size_t index = INVALID_INDEX) : Dimension(index) { poseEstimate.valid = index == 0; }
    };

    /**
     * A source provides observations of landmarks in each frame.
     */
    class Source : public HitNode::Dimension<2>
    {
    public:
        FeatureStore::ConstOwn store;

    protected:
        friend class Map3<Hit, Landmark, Frame, Source>;
        Source(size_t index = INVALID_INDEX) : Dimension(index) {}
    };

    class Map : protected Map3<Hit, Landmark, Frame, Source>
    {
    public:
        class Operator
        {
        public:
            virtual bool operator() (Map& map, size_t frame) = 0;
        };

        Map() : m_newLandmarkId(0), joinChkMetre("CHK"), joinMetre("JOIN") {}
        virtual ~Map() {}

        Landmark& AddLandmark();
        void RemoveLandmark(Landmark& l);
        bool IsJoinable(const Landmark& li, const Landmark& lj);
        Landmark& JoinLandmark(Landmark& li, Landmark& lj);

        /**
         * Update structure of a set of landmarks
         *
         * \param u m landmark pointers
         * \param g structure estimate of m 3D points
         * \param pose transformation to the frame of g
         * \return updated landmark structure in the frame of g
         */
        StructureEstimation::Estimate UpdateStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& g, const EuclideanTransform& pose);

        inline Landmark& GetLandmark(size_t index) { return Dim0(index); }
        inline Frame&    GetFrame   (size_t index) { return Dim1(index); }
        inline Source&   GetSource  (size_t index) { return Dim2(index); }

        Speedometre joinChkMetre;
        Speedometre joinMetre;

    private:
        size_t m_newLandmarkId;
    };

    class MultiObjectiveOutlierFilter : public FeatureMatcher::Filter
    {
    public:
        class ObjectiveBuilder : public Referenced<ObjectiveBuilder>
        {
        public:
            virtual void AddData(size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx) = 0;
            virtual PoseEstimator::Own GetSolver() = 0;
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector) = 0;
            virtual String ToString() const = 0;
        };

        MultiObjectiveOutlierFilter() {}
        virtual bool operator() (ImageFeatureMap& map, Indices& inliers);

        std::vector<ObjectiveBuilder::Own> builders;
    };

    class FeatureTracker : public Map::Operator
    {
    public:
        /**
         *
         */
        struct FramedStore
        {
            FramedStore(FeatureStore::ConstOwn& store = FeatureStore::ConstOwn(), int offset = 0, DisparityStore::ConstOwn& disp = DisparityStore::ConstOwn())
            : store(store), offset(offset),
              disp(disp && disp->GetStereoPair() && store->GetCamera() && disp->GetStereoPair()->GetPrimaryCamera() == store->GetCamera() ? disp : DisparityStore::ConstOwn())
            {}

            bool operator== (const FramedStore fs) const { return store && fs.store && store->GetIndex() == fs.store->GetIndex() && offset == fs.offset; }

            static const FramedStore Null;

            FeatureStore::ConstOwn store;
            DisparityStore::ConstOwn disp;
            int offset;
        };

        /**
         *
         */
        enum OutlierRejectionScheme
        {
            EPIPOLAR            = 1 << 0, ///< outlier detection using Epipolar constraints
            FORWARD_PROJECTION  = 1 << 1, ///< outlier detection using projective constraints when 3D-to-2D correspondences are available
            BACKWARD_PROJECTION = 1 << 2, ///< outlier detection using projective constraints when 2D-to-3D correspondences are available
            RIGID               = 1 << 3, ///< outlier detection using rigid alignment when 3D-to-3D correspondences are available
            PHOTOMETRIC         = 1 << 4  ///< outlier detection using photometric alignment when intensity data are available
        };

        /**
         * Objective builder for OutlierRejectionScheme::EPIPOLAR
         */
        class EpipolarObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            EpipolarObjectiveBuilder(ProjectionModel::ConstOwn& pi, ProjectionModel::ConstOwn& pj) : pi(pi), pj(pj) {}

            virtual void AddData(size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector);
            virtual PoseEstimator::Own GetSolver() { return PoseEstimator::Own(new EssentialMatrixDecomposer(pi, pj)); }
            virtual String ToString() const { return "EPIPOLAR"; }

            ProjectionModel::ConstOwn pi;
            ProjectionModel::ConstOwn pj;

        private:
            GeometricMapping::ImageToImageBuilder m_builder;
        };

        /**
         * Objective builder for OutlierRejectionScheme::FORWARD_PROJECTION and OutlierRejectionScheme::BACKWARD_PROJECTION
         */
        class PerspectiveObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            PerspectiveObjectiveBuilder(ProjectionModel::ConstOwn& p, const StructureEstimation::Estimate& g, bool forward)
            : p(p), g(g), forward(forward) {}

            virtual void AddData(size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector);
            virtual PoseEstimator::Own GetSolver();

            virtual String ToString() const { return forward ? "FORWARD PROJECTION" : "BACKWARD PROJECTION"; }

            ProjectionModel::ConstOwn p;
            const StructureEstimation::Estimate& g;
            const bool forward;

        private:
            GeometricMapping::WorldToImageBuilder m_builder;
        };

        /**
         * Objective builder for OutlierRejectionScheme::RIGID
         */
        class RigidObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            RigidObjectiveBuilder(const StructureEstimation::Estimate& gi, const StructureEstimation::Estimate& gj)
            : gi(gi), gj(gj) {}

            virtual void AddData(size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector);
            virtual PoseEstimator::Own GetSolver();

            virtual String ToString() const { return "RIGID"; }

            const StructureEstimation::Estimate& gi;
            const StructureEstimation::Estimate& gj;
        };

        /**
         * Objective builder for OutlierRejectionScheme::PHOTOMETRIC
         */
        class PhotometricObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:

            virtual String ToString() const { return "PHOTOMETRIC"; }

        };

        /**
         * The policy to deal with two joined marching paths originating from different landmarks
         * with inconsistent observations 
         */
        enum ConflictResolution
        {
            NO_MERGE,      ///< do not do merging
            KEEP_BOTH,     ///< merge landmarks but keep both histories
            KEEP_LONGEST,  ///< merge landmarks and keep the longer history
            KEEP_SHORTEST, ///< merge landmarks and keep the shorter history
            KEEP_BEST,     ///< merge landmarks and keep the history with lower error
            REMOVE_BOTH    ///< erase both landmarks' histories
        };

        /**
         *
         */
        FeatureTracker(const FramedStore& src = FramedStore::Null, const FramedStore& dst = FramedStore::Null)
        : src(src), dst(dst), policy(KEEP_BOTH), outlierRejection(EPIPOLAR) {}

        /**
         *
         */
        virtual bool operator() (Map& map, size_t frame);

        /**
         * Get feature's 3D coordinates and error covariances from a dense structure.
         *
         * \param f set of image features
         * \param dense structure from, e.g. a depth map
         *
         * \return structure estimates of the given features
         */
        StructureEstimation::Estimate GetFeatureStructure(const ImageFeatureSet& f, const StructureEstimation::Estimate& structure);

        /**
         *
         */
        bool IsOkay() const;

        /**
         *
         */
        bool InRange(size_t t, size_t tn) const;

        /**
         *
         */
        bool IsCrossed() const;

        /**
         *
         */
        bool IsSynchronised() const { return src.offset == dst.offset; }

        /**
         *
         */
        String ToString() const;

        const FramedStore src;
        const FramedStore dst;

        FeatureMatcher matcher;
        ConflictResolution policy;
        int outlierRejection;
    };

    class Mapper
    {
    public:
        struct Capability
        {
            bool motion; // ego-motion estimation
            bool metric; // metric reconstruction
            bool dense;  // dense or semi-dense reconstruction
        };

        virtual Capability GetCapability() const = 0;
        virtual bool SLAM(Map& map, size_t t0, size_t tn) = 0;
    };

    /**
     * Implementation of a generalised multi-frame feature integration algorithm
     * based on the paper "Visual Odometry by Multi-frame Feature Integration"
     */
    class MultiFrameFeatureIntegration : public Mapper
    {
    public:
        virtual Capability GetCapability() const;
        virtual bool SLAM(Map& map, size_t t0, size_t tn);

        bool AddTracking(const FeatureTracker& tracking);

    private:
        std::vector<FeatureTracker> m_tracking;
        std::vector<DisparityStore::ConstOwn> m_dispStores;
    };

    // class OrbSLAM : public Mapper {
    // ...
    // };
    //
    // class LargeScaleDenseSLAM : public Mapper {
    // ...
    // };
}
#endif // MAPPING_HPP

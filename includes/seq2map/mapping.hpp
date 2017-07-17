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

            Covar3D(const cv::Vec6d& cov)
            : xx(cov[0]), xy(cov[1]), xz(cov[2]), yy(cov[3]), yz(cov[4]), zz(cov[5]) {}

            Covar3D& operator= (const cv::Vec6d& cov)
            {
                xx = cov[0];
                xy = cov[1];
                xz = cov[2];
                yy = cov[3];
                yz = cov[4];
                zz = cov[5];

                return *this;
            }

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
        double  icv;

    protected:
        friend class Map3<seq2map::Hit, Landmark, Frame, Source>;
        Landmark(size_t index = INVALID_INDEX) : DimensionZero(index), icv(0) {}
    };

    /**
     * A frame represent the state at a point of time.
     */
    class Frame : public HitNode::Dimension<1>
    {
    public:
        std::map<size_t, Landmark::Ptrs> featureLandmarkLookup;
        std::map<size_t, ImageFeatureSet> augmentedFeaturs;

        PoseEstimator::Estimate poseEstimate;

    protected:
        friend class Map3<Hit, Landmark, Frame, Source>;
        Frame(size_t index = INVALID_INDEX) : Dimension(index) { poseEstimate.valid = (index == 0); }
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

    /**
     *
     */
    class Map : protected Map3<Hit, Landmark, Frame, Source>
    {
    public:
        /**
         *
         */
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

        StructureEstimation::Estimate GetStructure(const Landmark::Ptrs& u) const;
        void SetStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& structure);

        /**
         * Update structure of a set of landmarks
         *
         * \param u m landmark pointers
         * \param g structure estimate of m 3D points
         * \param pose transformation to the frame of g
         * \return updated landmark structure in the frame of g
         */
        StructureEstimation::Estimate UpdateStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& g, const EuclideanTransform& ref);

        /**
         *
         */
        Landmark::Ptrs GetLandmarks(std::vector<size_t> indices);

        inline Landmark& GetLandmark(size_t index) { return Dim0(index); }
        inline Frame&    GetFrame   (size_t index) { return Dim1(index); }
        inline Source&   GetSource  (size_t index) { return Dim2(index); }

        Speedometre joinChkMetre;
        Speedometre joinMetre;

    private:
        size_t m_newLandmarkId;
    };

    /**
     *
     */
    class MultiObjectiveOutlierFilter : public FeatureMatcher::Filter
    {
    public:
        /**
         *
         */
        class ObjectiveBuilder : public Referenced<ObjectiveBuilder>
        {
        public:
            ObjectiveBuilder(AlignmentObjective::InlierSelector::Stats& stats) : stats(stats) {}

            virtual void AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx) = 0;
            virtual PoseEstimator::Own GetSolver() const = 0;
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector) = 0;
            virtual String ToString() const = 0;

            AlignmentObjective::InlierSelector::Stats& stats;
        };

        MultiObjectiveOutlierFilter(size_t maxIterations, double minInlierRatio, double confidence)
        : maxIterations(maxIterations), minInlierRatio(minInlierRatio), confidence(confidence), optimisation(true) {}

        virtual bool operator() (ImageFeatureMap& map, Indices& inliers);

        std::vector<ObjectiveBuilder::Own> builders;
        PoseEstimator::Estimate motion;
        size_t maxIterations;
        double minInlierRatio;
        double confidence;
        bool optimisation;
    };

    /**
     *
     */
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
            EPIPOLAR_ALIGN      = 1 << 0, ///< outlier detection using Epipolar constraints
            FORWARD_PROJ_ALIGN  = 1 << 1, ///< outlier detection using projective constraints when 3D-to-2D correspondences are available
            BACKWARD_PROJ_ALIGN = 1 << 2, ///< outlier detection using projective constraints when 2D-to-3D correspondences are available
            RIGID_ALIGN         = 1 << 3, ///< outlier detection using rigid alignment when 3D-to-3D correspondences are available
            PHOTOMETRIC_ALIGN   = 1 << 4  ///< outlier detection using photometric alignment when intensity data are available
        };

        /**
         * Post-motion correspondence recovery for lost features.
         */
        enum InlierInjectionScheme
        {
            FORWARD_FLOW    = 1 << 0, ///< forward optical flow from the projections in the first frame
            BACKWARD_FLOW   = 1 << 1, ///< backward optical flow from the projections in the second frame
            EPIPOLAR_SEARCH = 1 << 2  ///< search along epipolar lines by block matching
        };

        /**
         * Post-motion structure recovery
         */
        enum StructureRecoveryScheme
        {
            NO_RECOVERY,
            MIDPOINT_TRIANGULATION,
            OPTIMAL_TRIANGULATION
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
        struct OutlierRejectionOptions
        {
            OutlierRejectionOptions(int scheme)
            : scheme(scheme), maxIterations(30), minInlierRatio(0.5), confidence(0.5), epipolarEps(1e2) {}

            int scheme;            ///< strategies to identify the outliers from noisy feature matches
            size_t maxIterations;  ///< the upper bound of trials
            double minInlierRatio; ///< the percentage of inliers minimally required to accept a motion hypothesis
            double confidence;     ///< probability that a random sample contains only inliers
            double epipolarEps;    ///< threshold of the epipolar objective, in the normalised image pixel
        };

        /**
         *
         */
        struct InlierInjectionOptions
        {
            InlierInjectionOptions(int scheme)
            : scheme(scheme), blockSize(5), levels(3), bidirectionalEps(0), extractDescriptor(false) {}

            int scheme;              ///< Strategies to recover missing features.
            size_t blockSize;        ///< Size of search window.
            size_t levels;           ///< Level of pyramid for optical flow computation.
            double bidirectionalEps; ///< Threshold of the forward-backward flow error, in image pixels. Set to a non-positive value to disable the test.
            double epipolarEps;      ///< Threshold of the epipolar objective to decide if a flow is valid, in normalised image pixel.
            double searchRange;      ///< Maximum distance between a prediction and a match hypothesis, applicable to BACKWARD_FLOW and EPIPOLAR_SEARCH.
            bool extractDescriptor;  ///< Recompute descriptor for each recovered landmark, set to false to re-use a previously extracted descriptor.
        };

        /**
         *
         */
        struct Stats
        {
            typedef std::map<int, AlignmentObjective::InlierSelector::Stats> ObjectiveStats;

            Stats() : spawned(0), tracked(0), joined(0), removed(0), injected(0) { motion.valid = false; }

            size_t spawned;  ///< number of newly discovered landmarks
            size_t tracked;  ///< number of tracked landmarks
            size_t joined;   ///< number of joined landmarks
            size_t removed;  ///< number of removed landmarks
            size_t injected; ///< number of recovered landmarks
            ObjectiveStats objectives; ///< per outlier model stats
            PoseEstimator::Estimate motion; ///< ego-motion
            GeometricMapping flow; ///< feature flow
        };


        /**
         * Objective builder for OutlierRejectionScheme::EPIPOLAR_ALIGN
         */
        class EpipolarObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            EpipolarObjectiveBuilder(const ProjectionModel::ConstOwn& pi, const ProjectionModel::ConstOwn& pj, double epsilon, AlignmentObjective::InlierSelector::Stats& stats)
            : pi(pi), pj(pj), epsilon(epsilon), ObjectiveBuilder(stats) {}

            virtual void AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector);
            virtual PoseEstimator::Own GetSolver() const { return PoseEstimator::Own(new EssentialMatrixDecomposer(pi, pj)); }
            virtual String ToString() const { return "EPIPOLAR"; }

            const ProjectionModel::ConstOwn pi;
            const ProjectionModel::ConstOwn pj;

            double epsilon;

        private:
            GeometricMapping::ImageToImageBuilder m_builder;
        };

        /**
         * Objective builder for OutlierRejectionScheme::FORWARD_PROJ_ALIGN and OutlierRejectionScheme::BACKWARD_PROJ_ALIGN
         */
        class PerspectiveObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            PerspectiveObjectiveBuilder(const ProjectionModel::ConstOwn& p, const StructureEstimation::Estimate& g, bool forward, AlignmentObjective::InlierSelector::Stats& stats)
            : p(p), g(g), forward(forward), ObjectiveBuilder(stats) {}

            virtual void AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector);
            virtual PoseEstimator::Own GetSolver() const;

            virtual String ToString() const { return forward ? "FORWARD PROJECTION" : "BACKWARD PROJECTION"; }

            const ProjectionModel::ConstOwn p;
            const StructureEstimation::Estimate& g;
            const bool forward;

        private:
            GeometricMapping::WorldToImageBuilder m_builder;
            Indices m_idx;
        };

        /**
         * Objective builder for OutlierRejectionScheme::PHOTOMETRIC_ALIGN
         */
        class PhotometricObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            PhotometricObjectiveBuilder(const ProjectionModel::ConstOwn& pj, const StructureEstimation::Estimate& gi, const cv::Mat& Ii, const cv::Mat& Ij, AlignmentObjective::InlierSelector::Stats& stats)
            : pj(pj), gi(gi), Ii(Ii), Ij(Ij), ObjectiveBuilder(stats) {}

            virtual void AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector);
            virtual PoseEstimator::Own GetSolver() const { return PoseEstimator::Own(); } // photometric objective has no closed-form solver
            
            virtual String ToString() const { return "PHOTOMETRIC"; }

            const ProjectionModel::ConstOwn pj;
            const StructureEstimation::Estimate& gi;
            const cv::Mat Ii;
            const cv::Mat Ij;

        private:
            Indices m_idx;
            std::vector<size_t> m_localIdx;
        };

        /**
         * Objective builder for OutlierRejectionScheme::RIGID_ALIGN
         */
        class RigidObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            RigidObjectiveBuilder(const StructureEstimation::Estimate& gi, const StructureEstimation::Estimate& gj, AlignmentObjective::InlierSelector::Stats& stats)
            : gi(gi), gj(gj), ObjectiveBuilder(stats) {}

            virtual void AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector);
            virtual PoseEstimator::Own GetSolver() const { return PoseEstimator::Own(new QuatAbsOrientationSolver()); }

            virtual String ToString() const { return "RIGID"; }

            const StructureEstimation::Estimate& gi;
            const StructureEstimation::Estimate& gj;

        private:
            GeometricMapping::WorldToWorldBuilder m_builder;
            Indices m_idx0;
            Indices m_idx1;
        };

        /**
         *
         */
        FeatureTracker(const FramedStore& src = FramedStore::Null, const FramedStore& dst = FramedStore::Null)
        : src(src), dst(dst), policy(KEEP_BOTH), outlierRejection(FORWARD_PROJ_ALIGN), inlierInjection(0), structureScheme(NO_RECOVERY) {}

        /**
         *
         */
        virtual bool operator() (Map& map, size_t frame);

        /**
         * Get feature's 3D coordinates and error covariances from a dense structure.
         *
         * \param f Set of image features
         * \param structure Dense structure from, for example, a depth map.
         *
         * \return Structure estimates of the given features
         */
        StructureEstimation::Estimate GetFeatureStructure(const ImageFeatureSet& f, const StructureEstimation::Estimate& structure);

        /**
         * Find features in the next frame using optical flow.
         *
         * \param Ii Image of source frame.
         * \param Ij Image of target frame.
         * \param fi Feature set of source frame.
         * \param eval Constructed epipolar objective.
         * \param pose Pose of the target frame with respect to the source.
         * \param tracked Input/output array of booleans to indicate the status of each feature's tracking.
         * \return Mapping of tracked feature from source frame to the target.
         */
        GeometricMapping FindLostFeaturesFlow(const cv::Mat& Ii, const cv::Mat& Ij, const ImageFeatureSet& fi, AlignmentObjective::Own eval, const EuclideanTransform& pose, std::vector<bool>& tracked);

        /**
         * Augment a target feature set by means of a feature flow from the source set.
         *
         * \param flow
         * \param fi
         * \param fj
         * \param ui
         * \param uj
         * \param aj
         * \return True when success, otherwise false.
         */
        bool AugmentFeatures(const GeometricMapping flow, const ImageFeatureSet& fi, const ImageFeatureSet& fj, Map& map, Frame& ti, Frame& tj, Source& si, Source& sj, Landmark::Ptrs& ui, Landmark::Ptrs& uj, ImageFeatureSet& aj);

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
        Stats stats;
        ConflictResolution policy;

        OutlierRejectionOptions outlierRejection;
        InlierInjectionOptions  inlierInjection;

        StructureRecoveryScheme structureScheme;
    };
}
#endif // MAPPING_HPP

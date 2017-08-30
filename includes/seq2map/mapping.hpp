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
        const size_t index; ///< originating feature index in the store
        Point2D proj;       ///< observed 2D image coordinates

        /**
         * Assignment.
         */
        Hit& operator= (const Hit& hit)
        {
            proj = hit.proj;
            return *this;
        }

    protected:
        friend class Landmark; // only landmarks can make hits

        /**
         * Constructor; accessible only via the Landmark class.
         */
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

        /**
         * Error covariance of a 3D coordinates.
         */
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

        Point3D position; ///< coordinates with respect to the reference frame
        Covar3D cov;      ///< covariance of the coordinates

    protected:
        friend class Map3<seq2map::Hit, Landmark, Frame, Source>;

        /**
         * A Landmark can only be created by a Map.
         */
        Landmark(size_t index = INVALID_INDEX) : DimensionZero(index) {}
    };

    /**
     * A frame represent the state at a point of time.
     */
    class Frame : public HitNode::Dimension<1>
    {
    public:
        /**
         * A feature index-to-landmark lookup table.
         */
        std::map<size_t, Landmark::Ptrs> featureLandmarkLookup;

        /**
         * Augmented feature sets indexed by the source store IDs.
         */
        std::map<size_t, ImageFeatureSet> augmentedFeaturs;

        /**
         * Compute the ratio of landmarks shared with another frame.
         */
        double GetCovisibility(const Frame& tj) const;

        PoseEstimator::Estimate pose;

    protected:
        friend class Map3<Hit, Landmark, Frame, Source>;

        /**
         * A Frame can only be created by a Map.
         */
        Frame(size_t index = INVALID_INDEX) : Dimension(index) { pose.valid = (index == 0); }
    };

    /**
     * A source provides observations of landmarks in each frame.
     */
    class Source : public HitNode::Dimension<2>
    {
    public:
        FeatureStore::ConstOwn store; ///< the source feature store
        DisparityStore::ConstOwn dpm; ///< the source disparity store

    protected:
        friend class Map3<Hit, Landmark, Frame, Source>;
        Source(size_t index = INVALID_INDEX) : Dimension(index) {}
    };

    /**
     * A Map contains sets of Landmark, Frame and Source.
     * The these classes are connected by Hits, which are internally stored as a sparse data structure.
     */
    class Map
    : protected Map3<Hit, Landmark, Frame, Source>,
      public Persistent<Path>
    {
    public:
        /**
         * Operator applied to a map taking one frame as input.
         *
         * \param map A map to be processed.
         * \param s0 The referenced source.
         * \param t0 The referenced frame.
         *
         * \return True if the process is successful; otherwise false.
         */
        class UnaryOperator
        {
        public:
            virtual bool operator() (Map& map, Source& s0, Frame& t0) = 0;
        };

        /**
         * Operator applied to a map taking two frames as inputs.
         *
         * \param map A map to be processed.
         * \param s0 The first referenced source.
         * \param t0 The first referenced frame.
         * \param s1 The second referenced source.
         * \param t1 The second referenced frame.
         *
         * \return True if the process is successful; otherwise false.
         */
        class BinaryOperator
        {
        public:
            virtual bool operator() (Map& map, Source& s0, Frame& t0, Source& s1, Frame& t1) = 0;
        };

        /**
         *
         */
        Map() : m_newLandmarkId(0), m_newSourcId(0) {}
        virtual ~Map() {}

        /**
         *
         */
        Source& AddSource(FeatureStore::ConstOwn& store, DisparityStore::ConstOwn& dpm);

        /**
         *
         */
        Landmark& AddLandmark();

        /**
         *
         */
        void AddKeyframe(size_t index) { m_keyframes.insert(index); }

        /**
         *
         */
        inline size_t GetLastKeyframe() const { return m_keyframes.empty() ? INVALID_INDEX : *m_keyframes.rbegin(); }

        /**
         * Remove a landmark from the map.
         */
        void RemoveLandmark(Landmark& l);
        
        /**
         * ...
         */
        bool IsJoinable(const Landmark& li, const Landmark& lj);
        
        /**
         * ...
         */
        Landmark& JoinLandmark(Landmark& li, Landmark& lj);

        /**
         * Retrieve structure of landmarks.
         */
        StructureEstimation::Estimate GetStructure(const Landmark::Ptrs& u) const;

        /**
         * Write back structure estimates of landmark.
         */
        void SetStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& structure);

        /**
         * Update structure of a set of landmarks
         *
         * \param u m landmark pointers
         * \param g structure estimate of m 3D points
         * \param pose transformation to the frame of g
         *
         * \return updated landmark structure in the frame of g
         */
        StructureEstimation::Estimate UpdateStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& g, const EuclideanTransform& ref);

        /**
         * Inqury a set of landmark given an index list.
         */
        Landmark::Ptrs GetLandmarks(std::vector<size_t> indices);

        /**
         * Get number of landmarks on the map.
         */
        inline size_t GetLandmarks() const { return GetSize0(); }

        /**
         * Retrieve a landmark.
         */
        inline Landmark& GetLandmark(size_t index) { return Dim0(index); }

        /**
         * Retrieve a frame.
         */
        inline Frame& GetFrame(size_t index) { return Dim1(index); }

        /**
         * Retrieve a source.
         */
        inline Source& GetSource(size_t index) { return Dim2(index); }

        inline void SetSequencePath(const Path& path) { m_seqPath = path; }

        void Clear();

        //
        // Persistent
        //
        bool Store(Path& path) const;
        bool Restore(const Path& path);

    private:
        Path m_seqPath;
        size_t m_newLandmarkId;
        size_t m_newSourcId;
        std::set<size_t> m_keyframes;
    };

    /**
     * A multi-objective RANSAC-based outlier filter.
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
            /**
             * ...
             */
            ObjectiveBuilder(AlignmentObjective::InlierSelector::Stats& stats, bool reduceMetric = false)
            : stats(stats), reduceMetric(reduceMetric) {}

            /**
             * ...
             */
            virtual bool AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx) = 0;

            /**
             * ...
             */
            virtual PoseEstimator::Own GetSolver() const = 0;

            /**
             * ...
             */
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma) = 0;

            /**
             * A method internally used to determine if the builder uses prebuilt data rather than the
             * correspondences established by a FeatureMatcher. Currently only PhotometricObjectiveBuilder
             * uses pre-built data term.
             */
            virtual bool Prebuilt() const { return false; }

            /**
             * ...
             */
            virtual String ToString() const = 0;

            AlignmentObjective::InlierSelector::Stats& stats;
            bool reduceMetric;
        };

        MultiObjectiveOutlierFilter(size_t maxIterations, double minInlierRatio, double confidence, double sigma)
        : maxIterations(maxIterations), minInlierRatio(minInlierRatio), confidence(confidence), optimisation(true), sigma(sigma) {}

        virtual bool operator() (ImageFeatureMap& map, IndexList& inliers);

        std::vector<ObjectiveBuilder::Own> builders;
        PoseEstimator::Estimate motion;
        size_t maxIterations;
        double minInlierRatio;
        double confidence;
        double sigma;
        bool optimisation;
    };

    /**
     * A FeatureTracker is a binary operator applicable to two frames.
     * The tracking performs follow steps:
     * <ol>
     *   <li>
     *     <b>Pre-motion structure retrieval.</b>
     *     A disparity map is loaded and converted to depth map if available.
     *     If the pose of frame is known, the retrieved structure is integrated with the existing
     *     landmark structure using a Bauesian filter.
     *   </li>
     *   <li>
     *     <b>Correspondence establishment and egomotion estimation.</b>
     *     Image features are loaded and matched in the feature space.
     *     An adjoint outlier rejection and pose recovery algorithm is then carried out.
     *   </li>
     *   <li>
     *     <b>Post-motion correspondence discovery.</b>
     *     The features that failed to find corresponding entries are processed by a
     *     motion-based correspondence analysis algorithm.
     *   </li>
     *   <li>
     *     <b>Post-motion structure recovery.</b>
     *     The feature flow obtained in previous two stages is used by a two-view triangulator
     *     to update the landmarks' structure. A newly calculated coodinates is integrated with
     *     an existing one by a Bayesian filter.
     *   </li>
     * </ol>
     */
    class FeatureTracker
    : public Map::BinaryOperator,
      public Parameterised,
      public Named
    {
    public:
        /**
         * Alignment models for outlier rejection and egomotion estimation.
         */
        enum OutlierRejectionScheme
        {
            EPIPOLAR_ALIGN      = 1 << 0, ///< Epipolar constraints
            FORWARD_PROJ_ALIGN  = 1 << 1, ///< projective constraints when 3D-to-2D correspondences are available
            BACKWARD_PROJ_ALIGN = 1 << 2, ///< projective constraints when 2D-to-3D correspondences are available
            RIGID_ALIGN         = 1 << 3, ///< rigid alignment when 3D-to-3D correspondences are available
            PHOTOMETRIC_ALIGN   = 1 << 4  ///< photometric alignment when intensity data are available
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
        enum TriangulationMethod
        {
            DISABLED,
            MIDPOINT,
            OPTIMAL
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
         * Options for outlier rejection and egomotion estimation.
         */
        struct OutlierRejectionOptions
        {
            OutlierRejectionOptions(int model)
            : model(model), maxIterations(30), minInlierRatio(0.5), confidence(0.5), epipolarEps(1e3), sigma(1.0f), fastMetric(false) {}

            int model;             ///< strategies to identify outliers from noisy feature matches
            size_t maxIterations;  ///< the upper bound of trials
            double minInlierRatio; ///< the percentage of inliers minimally required to accept a motion hypothesis
            double confidence;     ///< desired confidence to obtain a valid result, from zero to one
            double epipolarEps;    ///< threshold of the epipolar objective, in the normalised image pixel
            double sigma;          ///< threshold to determine if a model is fit or not
            bool fastMetric;       ///< reduce Mahalanobis metric to a weighted Euclidean one for acceleration
        };

        /**
         * Options for inlier recovery.
         */
        struct InlierInjectionOptions
        {
            InlierInjectionOptions(int scheme)
            : scheme(scheme), blockSize(5), levels(3), bidirectionalTol(1), epipolarEps(1e3), extractDescriptor(false) {}

            int scheme;              ///< Strategies to recover missing features.
            size_t blockSize;        ///< Size of search window.
            size_t levels;           ///< Level of pyramid for optical flow computation.
            double bidirectionalTol; ///< Threshold of the forward-backward flow error, in image pixels. Set to a non-positive value to disable the test.
            double epipolarEps;      ///< Threshold of the epipolar objective to decide if a flow is valid, in normalised image pixel.
            double searchRange;      ///< Maximum distance between a prediction and a match hypothesis, applicable to BACKWARD_FLOW and EPIPOLAR_SEARCH.
            bool extractDescriptor;  ///< Recompute descriptor for each recovered landmark, set to false to re-use a previously extracted descriptor.
        };

        /**
         * Feature tracking statistics updated each time the operator is in action.
         */
        class Stats
        {
        public:
            typedef std::map<int, AlignmentObjective::InlierSelector::Stats> ObjectiveStats;

            Stats() : spawned(0), tracked(0), joined(0), removed(0), injected(0), accumulated(0) { motion.valid = false; }
            String ToString() const;
            void Render(const cv::Mat& canvas, String& tracker, Map& map, const EuclideanTransform& tform);

            size_t spawned;  ///< number of newly discovered landmarks
            size_t tracked;  ///< number of tracked landmarks
            size_t joined;   ///< number of joined landmarks
            size_t removed;  ///< number of removed landmarks
            size_t injected; ///< number of recovered landmarks
            size_t accumulated; ///< number of accumulated landmarks
            ObjectiveStats objectives;      ///< per outlier model stats
            PoseEstimator::Estimate motion; ///< egomotion
            GeometricMapping flow;          ///< feature flow
            cv::Mat im;

        private:
            void PutTextLine(cv::Mat im, const String& text, int face, double scale, const cv::Scalar& colour, int thickness, cv::Point& pt);
        };

        /**
         * Objective builder for OutlierRejectionScheme::EPIPOLAR_ALIGN
         */
        class EpipolarObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            EpipolarObjectiveBuilder(
                const ProjectionModel::ConstOwn& pi, const ProjectionModel::ConstOwn& pj,
                double epsilon, AlignmentObjective::InlierSelector::Stats& stats)
            : pi(pi), pj(pj), epsilon(epsilon), ObjectiveBuilder(stats) {}

            virtual bool AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma);
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
            PerspectiveObjectiveBuilder(
                const ProjectionModel::ConstOwn& p, const StructureEstimation::Estimate& g,
                bool forward, bool reduceMetric, AlignmentObjective::InlierSelector::Stats& stats)
            : p(p), g(g), forward(forward), ObjectiveBuilder(stats, reduceMetric) {}

            virtual bool AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma);
            virtual PoseEstimator::Own GetSolver() const;

            virtual String ToString() const { return forward ? "FORWARD PROJECTION" : "BACKWARD PROJECTION"; }

            const ProjectionModel::ConstOwn p;
            const StructureEstimation::Estimate& g;
            const bool forward;

        private:
            GeometricMapping::WorldToImageBuilder m_builder;
            IndexList m_idx;
        };

        /**
         * Objective builder for OutlierRejectionScheme::PHOTOMETRIC_ALIGN
         */
        class PhotometricObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            PhotometricObjectiveBuilder(
                const ProjectionModel::ConstOwn& pj, const StructureEstimation::Estimate& gi,
                const cv::Mat& Ii, const cv::Mat& Ij, bool reduceMetric, AlignmentObjective::InlierSelector::Stats& stats)
            : pj(pj), gi(gi), Ii(Ii), Ij(Ij), ObjectiveBuilder(stats, reduceMetric) {}

            virtual bool AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma);
            virtual PoseEstimator::Own GetSolver() const { return PoseEstimator::Own(); } // photometric objective has no closed-form solver

            virtual bool Prebuilt() const { return true; }
            virtual String ToString() const { return "PHOTOMETRIC"; }

            const ProjectionModel::ConstOwn pj;
            const StructureEstimation::Estimate& gi;
            const cv::Mat Ii;
            const cv::Mat Ij;

        private:
            IndexList m_idx;
            std::vector<size_t> m_localIdx;
            Points2F m_imagePoints;
        };

        /**
         * Objective builder for OutlierRejectionScheme::RIGID_ALIGN
         */
        class RigidObjectiveBuilder : public MultiObjectiveOutlierFilter::ObjectiveBuilder
        {
        public:
            RigidObjectiveBuilder(
                const StructureEstimation::Estimate& gi, const StructureEstimation::Estimate& gj,
                bool reduceMetric, AlignmentObjective::InlierSelector::Stats& stats)
            : gi(gi), gj(gj), ObjectiveBuilder(stats, reduceMetric) {}

            virtual bool AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx);
            virtual bool Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma);
            virtual PoseEstimator::Own GetSolver() const { return PoseEstimator::Own(new QuatAbsOrientationSolver()); }

            virtual String ToString() const { return "RIGID"; }

            const StructureEstimation::Estimate& gi;
            const StructureEstimation::Estimate& gj;

        private:
            GeometricMapping::WorldToWorldBuilder m_builder;
            IndexList m_idx0;
            IndexList m_idx1;
        };

        /**
         *
         */
        FeatureTracker()
        : policy(REMOVE_BOTH),
          outlierRejection(FORWARD_PROJ_ALIGN),
          inlierInjection(0),
          triangulation(OPTIMAL),
          rendering(true),
          matcher(true, true, false, 0.8f, true)
        {}

        /**
         * Get features' 3D coordinates and error covariances from a dense structure.
         *
         * \param f set of image features
         * \param structure Dense structure from, for example, a depth map.
         *
         * \return Structure estimate of the given features
         */
        StructureEstimation::Estimate GetFeatureStructure(const ImageFeatureSet& f, const StructureEstimation::Estimate& structure);

        /**
         * Get features' 1D indices from their 2D image subscripts.
         *
         * \param f feature set.
         * \param imageSize size of image plane.
         * \resutn Indices of feature in image plane.
         */
        IndexList GetFeatureImageIndices(const ImageFeatureSet& f, const cv::Size& imageSize) const;

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
        GeometricMapping FindFeaturesFlow(const cv::Mat& Ii, const cv::Mat& Ij, const ImageFeatureSet& fi, AlignmentObjective::Own& eval, const EuclideanTransform& pose, std::vector<bool>& tracked);

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
        bool AugmentFeatures(const GeometricMapping flow, const ImageFeatureSet& fi, const ImageFeatureSet& fj, Map& map, Frame& ti, Frame& tj, Source& si, Source& sj, Landmark::Ptrs& ui, Landmark::Ptrs& uj, ImageFeatureSet& aj, size_t& spawned, size_t& tracked);

        /**
         *
         */
        //String ToString() const;

        //
        // Map::BinaryOperator
        //
        virtual bool operator() (Map& map, Source& s0, Frame& t0, Source& s1, Frame& t1);

        //
        // Parametrised
        //
        virtual void WriteParams(cv::FileStorage& fs) const;
        virtual bool ReadParams(const cv::FileNode& fn);
        virtual void ApplyParams();
        virtual Options GetOptions(int flag = 0);

        Stats stats; ///< Feature tracking statistics
        FeatureMatcher matcher;    ///< Descriptor matcher
        ConflictResolution policy; ///< Landmark merge policy
        OutlierRejectionOptions outlierRejection; ///< outlier rejection options
        InlierInjectionOptions  inlierInjection;  ///< inlier recovery options
        TriangulationMethod     triangulation;    ///< triangulation method
        bool rendering;

    private:
        static bool StringToAlignment(const String& flag, int& model);
        static bool StringToFlow(const String& flow, int& scheme);
        static bool StringToTriangulation(const String& triangulation, TriangulationMethod& method);
        static String AlignmentToString(int model);
        static String FlowToString(int scheme);
        static String TriangulationToString(TriangulationMethod method);

        String m_alignString;
        String m_flowString;
        String m_triangulation;
        double m_epipolarEps;
    };
}
#endif // MAPPING_HPP

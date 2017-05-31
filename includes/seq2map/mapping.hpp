#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <seq2map/sequence.hpp>
#include <seq2map/spamap.hpp>

namespace seq2map
{
    class Frame;
    class Landmark;

    struct Hit
    {
    public:
        Landmark& landmark; // owning landmark
        Frame& frame;       // owning frame
        const size_t store; // originating store index
        const size_t index; // originating feature index in the store
        Point2D proj;       // observed 2D image coordinates

    protected:
        friend class Landmark; // only landmarks can make hits

        Hit(Landmark& landmark, Frame& frame, size_t store, size_t index)
        : landmark(landmark), frame(frame), store(store), index(index) {}
    };

    class Landmark : public spamap::Node<Hit>::Row
    {
    public:
        virtual inline Hit& Hit(Frame& frame, size_t store, size_t index);
        Point3D position;

    protected:
        friend class spamap::Map<seq2map::Hit, Landmark, Frame>;
        Landmark(size_t index = INVALID_INDEX) : Row(index) {}
    };

    class Frame : public spamap::Node<Hit>::Column
    {
    public:
        typedef std::vector<size_t> IdList;
        typedef std::map<size_t, IdList> IdLists;

        IdLists featureIdLookup;

    protected:
        friend class spamap::Map<Hit, Landmark, Frame>;
        Frame(size_t index = INVALID_INDEX) : Column(index) {}
    };

    class Map : public spamap::Map<Hit, Landmark, Frame>
    {
    public:
        Map() : m_newLandmarkId(0) {}
        virtual ~Map() {}

        Landmark& AddLandmark();
        inline Landmark& GetLandmark(size_t index) { return Row(index); }
        inline Frame&    GetFrame   (size_t index) { return Col(index); }
        
    private:
        size_t    m_newLandmarkId;
    };

    class Operator
    {
    public:
        virtual bool InScope(size_t frame) = 0;
        virtual bool operator() (Map& map, size_t frame) = 0;
    };

    class MatchFeatures : public Operator
    {
    public:
        MatchFeatures(FeatureStore& src, FeatureStore& dst, size_t ti, size_t tj);

        //inline void UseMotionPrior(bool enable) { m_motionPrior = enable; }
        //inline void MakeMotionPosterior(bool enable) { m_motionPosterior = enable; }

        inline void SetRigidAlignCutoff(double cutoff) {}
        inline void SetProjectiveAlignCutff(double cutff) {}
        inline void SetEpipolarCutoff(double cutoff) {}

        void EnableGeometricFilter(cv::Mat Ki = cv::Mat(), cv::Mat Kj = cv::Mat());

    private:
        //bool m_geometricFiltering;
        //bool m_motionPrior;
        //bool m_motionPosterior;
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

        virtual Capability GetCapability(const Sequence& seq) const = 0;
        virtual bool SLAM(const Sequence& seq, Map& map, size_t t0, size_t tn) = 0;
        inline bool SLAM(const Sequence& seq, Map& map, size_t t0 = 0) { return SLAM(seq, map, t0, seq.GetFrames() - 1); }
    };

    /**
     * Implementation of a generalised multi-frame feature integration algorithm
     * based on the paper "Visual Odometry by Multi-frame Feature Integration"
     */
    class MultiFrameFeatureIntegration : public Mapper
    {
    public:
        // the policy to join two marching paths originating from different landmarks
        enum MultiPathMergePolicy
        {
            NO_MERGE,       // do not do merging
            KEEP_BOTH,      // merge landmarks but keep both histories
            KEEP_LONGEST,   // merge landmarks and keep the longer history
            KEEP_SHORTEST,  // merge landmarks and keep the shorter history
            KEEP_BEST,      // merge landmarks and keep the history with lower error
            REJECT          // erase both landmarks' histories
        };

        using Mapper::SLAM;

        virtual Capability GetCapability(const Sequence& seq) const;
        virtual bool SLAM(const Sequence& seq, Map& map, size_t t0, size_t tn);

        bool AddPathway(size_t f0, size_t f1, int dt0 = 0, int dt1 = 1);
        inline void SetMergePolicy(MultiPathMergePolicy policy) { m_mergePolicy = policy; }

    private:
        struct Pathway
        {
            Pathway(size_t f0, size_t f1, int dt0, int dt1)
            : srcStoreIdx(f0), dstStoreIdx(f1), srcFrameOffset(dt0), dstFrameOffset(dt1) {}

            String ToString() const;
            bool CheckRange(size_t t0, size_t tn, size_t t) const;

            size_t srcStoreIdx; // source feature store
            size_t dstStoreIdx; // destination feature store
            int srcFrameOffset; // source frame t + srcFrameOffset
            int dstFrameOffset; // destination frame t + dstFrameOffset

            FeatureStore const* srcStore; // pointer to the source store, initialised in Map()
            FeatureStore const* dstStore; // pointer to the destination store, initialised in Map()
        };

        bool BindPathways(const Sequence& seq, size_t& maxStoreIndex);

        std::vector<Pathway> m_pathways;
        MultiPathMergePolicy m_mergePolicy;
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

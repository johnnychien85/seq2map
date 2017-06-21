#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <seq2map/sequence.hpp>
#include <seq2map/sparse_node.hpp>

namespace seq2map
{
    class Landmark;
    class Frame;
    class Source;

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
        Hit& Hit(Frame& frame, Source& src, size_t index);
        Point3D position;

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
        typedef std::vector<size_t> IdList;
        typedef std::map<size_t, IdList> IdLists;

        IdLists featureIdLookup;

    protected:
        friend class Map3<Hit, Landmark, Frame, Source>;
        Frame(size_t index = INVALID_INDEX) : Dimension(index) {}
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

        inline Landmark& GetLandmark(size_t index) { return Dim0(index); }
        inline Frame&    GetFrame   (size_t index) { return Dim1(index); }
        inline Source&   GetSource  (size_t index) { return Dim2(index); }

        Speedometre joinChkMetre;
        Speedometre joinMetre;

    private:
        size_t m_newLandmarkId;
    };

    class FeatureMatching : public Map::Operator
    {
    public:
        struct FramedStore
        {
            FramedStore(FeatureStore::ConstOwn& store = FeatureStore::ConstOwn(), int offset = 0) : store(store), offset(offset) {}

            bool operator==(const FramedStore fs) const { return store && fs.store && store->GetIndex() == fs.store->GetIndex() && offset == fs.offset; }

            FeatureStore::ConstOwn store;
            int offset;
            static const FramedStore Null;
        };

        // the policy to deal with two joined marching paths originating from different landmarks
        // with inconsistent observations 
        enum ConflictResolution
        {
            NO_MERGE,       ///< do not do merging
            KEEP_BOTH,      ///< merge landmarks but keep both histories
            KEEP_LONGEST,   ///< merge landmarks and keep the longer history
            KEEP_SHORTEST,  ///< merge landmarks and keep the shorter history
            KEEP_BEST,      ///< merge landmarks and keep the history with lower error
            REMOVE_BOTH     ///< erase both landmarks' histories
        };

        /**
         *
         */
        FeatureMatching(const FramedStore& src = FramedStore::Null, const FramedStore& dst = FramedStore::Null)
        : src(src), dst(dst), policy(KEEP_BOTH) {}

        /**
         *
         */
        virtual bool operator() (Map& map, size_t frame);

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

        bool AddMatching(const FeatureMatching& matching);

    private:
        std::vector<FeatureMatching> m_matchings;
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

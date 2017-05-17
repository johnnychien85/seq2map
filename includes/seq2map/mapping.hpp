#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <seq2map/sequence.hpp>
#include <seq2map/spamap.hpp>

namespace seq2map
{
    class Map
    {
    private:
        template<typename T> class LinkedNode2D : public boost::enable_shared_from_this<LinkedNode2D<T> >
        {
        public:
            LinkedNode2D(const T& value) : value(value) {}
            virtual ~LinkedNode2D() {}

            virtual void Detach();

            typedef LinkedNode2D<T> NodeType;
            typedef boost::shared_ptr<NodeType> Ptr;
            typedef boost::weak_ptr<NodeType>   Ref;

            Ref up;
            Ref down;
            Ref left;
            Ptr right;
            T   value;
        };

    public:
        class Landmark;
        class Frame;
        class Hit
        {
        public:
            Landmark& landmark; // owning landmark
            Frame& frame;       // owning frame
            const size_t store; // originating store index
            const size_t index; // originating feature index in the store
            Point2D proj;       // observed 2D image coordinates

        protected:
            friend class Landmark; // only Landmark can make and own a Hit

            Hit(Landmark& landmark, Frame& frame, size_t store, size_t index)
            : landmark(landmark), frame(frame), store(store), index(index) {}
        };

        class HitNode : public LinkedNode2D<Hit>
        {
        public:
            virtual ~HitNode() { s_kill++; Detach(); }

        protected:
            friend class Landmark;

            HitNode(const Hit& hit);
            virtual void Detach();

        private:
            static size_t s_born;
            static size_t s_kill;
        };

        class HitList
        {
        public:
            size_t Count() const;
            void Clear() { m_tail.reset(); m_head.reset(); }

        protected:
            friend class HitNode; // let HitNode to maintain my head and tail

            HitNode::Ptr m_head;
            HitNode::Ptr m_tail;
        };

        class Landmark : public Indexed
        {
        public:
            virtual inline Hit& Hit(Frame& frame, size_t store, size_t index);

            Point3D position;
            HitList hits;

        protected:
            friend class Map; // only Map can create a new landmark
            Landmark(size_t index = INVALID_INDEX) : Indexed(index) {}
        };

        class Frame : public Indexed
        {
        public:
            typedef std::vector<size_t> IdList;
            typedef std::map<size_t, IdList> IdLists;

            HitList hits;
            IdLists featureIdLists;

        protected:
            friend class Map; // only Map can create a new frame
            Frame(size_t index = INVALID_INDEX) : Indexed(index) {}
        };

        Map() : m_newLandmarkId(0) {}
        virtual ~Map();

        Landmark& AddLandmark();
        Landmark& GetLandmark(size_t index);
        Frame&    GetFrame(size_t index);
        
    private:
        typedef std::map<size_t, Landmark> Landmarks;
        typedef std::map<size_t, Frame>    Frames;

        Landmarks m_landmarks; // observed image landmarks
        Frames    m_frames;    // explored frames

        size_t    m_newLandmarkId;
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

#include <seq2map/mapping.hpp>

using namespace seq2map;

class NullLandmark
: public Map::Landmark,
  public Singleton<NullLandmark>
{
public:
    NullLandmark() : Landmark(INVALID_INDEX) {}
    virtual void SetIndex(size_t) { return; }
};

template<typename T>
void Map::LinkedNode2D<T>::Detach()
{
    if (this->up.lock())   this->up.lock()->down    = this->down;
    if (this->down.lock()) this->down.lock()->up    = this->up;
    if (this->left.lock()) this->left.lock()->right = this->right;
    if (this->right)       this->right->left        = this->left;

    // unlink all the references
    this->up.reset();
    this->down.reset();
    this->left.reset();
    this->right.reset();
}

size_t Map::HitNode::s_born = 0;
size_t Map::HitNode::s_kill = 0;

Map::HitNode::HitNode(const Hit& hit)
: LinkedNode2D(hit)
{
    Landmark& landmark = hit.landmark;
    Frame& frame = hit.frame;

    Ptr hook = Ptr(this);

    std::cout << "landmark " << landmark.GetIndex() << " hits in frame " << frame.GetIndex() << std::endl;

    // append the hit to the end of landmark's hit list
    if (!landmark.hits.m_tail) // first hit of the landmark
    {
        assert(!landmark.hits.m_head); // make sure the list is empty
        landmark.hits.m_head = landmark.hits.m_tail = this->shared_from_this();

        //E_INFO << "first landmark hit!";
    }
    else
    {
        // make sure nothing after the tail and the tail belongs to the landmark of the new hit
        assert(!landmark.hits.m_tail->right);
        assert(landmark.hits.m_tail->value.landmark.GetIndex() == landmark.GetIndex());

        // append the new hit to the right of the tail
        this->left = landmark.hits.m_tail;

        landmark.hits.m_tail = (this->left = landmark.hits.m_tail).lock()->right = this->shared_from_this();
        //E_INFO << "another landmark hit..";
    }

    // update the frame's hit list
    HitNode::Ptr left = this->left.lock();
    if (left && left->value.frame.GetIndex() == frame.GetIndex())
    { // check if the landmark's previous hit and the new one belong to the same frame
        HitNode::Ptr leftDown = left->down.lock();
        if (leftDown) // the previous hit is not the last hit in the frame's hit list
        {
            // make sure next hit in the frame's hit list and the new hit belong to the same frame
            assert(leftDown->value.frame.GetIndex() == frame.GetIndex());
            assert(leftDown->up.lock() == left); // check the validity of double links

            (this->down = leftDown).lock()->up = this->shared_from_this();
            //E_INFO << "hit inserted before landmark " << left->down->value.landmark.GetIndex() << " at stratum " << frame.GetIndex();
        }
        else
        {
            assert(frame.hits.m_tail == left);
            frame.hits.m_tail = this->shared_from_this();
        }

        // insert the new hit right after it's left node
        (this->up = left).lock()->down = this->shared_from_this();
        //E_INFO << "hit inserted after landmark " << left->value.landmark.GetIndex() << " at stratum " << frame.GetIndex();

        return; // insertion finished
    }

    // append the hit to the end of the frame's hit list
    if (!frame.hits.m_tail) // first hit of the frame
    {
        assert(!frame.hits.m_head); // make sure the list is empty
        frame.hits.m_head = frame.hits.m_tail = this->shared_from_this();

        //E_INFO << "first hit of stratum " << frame.GetIndex() << "!";
    }
    else
    {
        // make sure nothing after the tail and the tail belongs to the frame of the new hit
        assert(!frame.hits.m_tail->down.lock());
        assert(frame.hits.m_tail->value.frame.GetIndex() == frame.GetIndex());

        frame.hits.m_tail = ((this->up = frame.hits.m_tail).lock()->down = this->shared_from_this()).lock();
        //E_INFO << "another hit of stratum " << frame.GetIndex() << "..";
    }

    s_born++;
}

void Map::HitNode::Detach()
{
    Landmark& landmark = value.landmark;
    Frame& frame = value.frame;

    std::cout << "a hit is detaching from landmark " << landmark.GetIndex() << " frame " << frame.GetIndex() << std::endl;

    if (landmark.hits.m_head && landmark.hits.m_head.get() == this) landmark.hits.m_head = this->right;
    if (landmark.hits.m_tail && landmark.hits.m_tail.get() == this) landmark.hits.m_tail = this->left.lock();
    if (frame.hits.m_head && frame.hits.m_head.get() == this) frame.hits.m_head = this->down.lock();
    if (frame.hits.m_tail && frame.hits.m_tail.get() == this) frame.hits.m_tail = this->up.lock();
    
    LinkedNode2D::Detach();
}

Map::Hit& Map::Landmark::Hit(Frame& frame, size_t store, size_t index)
{
    return (new HitNode(Map::Hit(*this, frame, store, index)))->value;
}

Map::~Map()
{
    //for (Frames::reverse_iterator itr = m_frames.rbegin(); itr != m_frames.rend(); itr++)
    //{
    //    //itr->second.hits
    //}

    BOOST_FOREACH (Frames::value_type& pair, m_frames)
    {
        pair.second.hits.Clear();
    }

    BOOST_FOREACH (Landmarks::value_type& pair, m_landmarks)
    {
        pair.second.hits.Clear();
    }
}

Map::Landmark& Map::AddLandmark()
{
    std::pair<Landmarks::iterator, bool> result = 
        m_landmarks.insert(Landmarks::value_type(m_newLandmarkId, Landmark(m_newLandmarkId)));

    if (result.second)
    {
        m_newLandmarkId++;
    }
    else
    {
        E_ERROR << "error inserting new landmark " << m_newLandmarkId;
    }

    return result.first->second;
}

Map::Landmark& Map::GetLandmark(size_t index)
{
    Landmarks::iterator itr = m_landmarks.find(index);
    bool found = itr != m_landmarks.end();
    return found ? itr->second : NullLandmark::GetInstance();
}

Map::Frame& Map::GetFrame(size_t index)
{
    Frames::iterator itr = m_frames.find(index);
    bool found = itr != m_frames.end();

    if (found)
    {
        return itr->second;
    }

    return m_frames.insert(Frames::value_type(index, Frame(index))).first->second;
}

/*
Landmark::IdTable::IdList Landmark::IdTable::s_nullList(0);

Landmark::IdTable::IdList& Landmark::IdTable::operator() (size_t f, size_t t)
{
    size_t i = t - m_begin;

    if (i >= m_table.size())
    {
        E_WARNING << "frame index " << t << " out of bound [" << m_begin << "," << (m_begin + m_frames - 1) << "]";
        return s_nullList;
    }

    if (f >= m_stores)
    {
        E_WARNING << "store index " << f << " out of bound (" << m_stores << ")";
        return s_nullList;
    }

    if (m_table[i].empty())
    {
        m_table[i].resize(m_stores);
    }

    return m_table[i][f];
}
*/

Mapper::Capability MultiFrameFeatureIntegration::GetCapability(const Sequence& seq) const
{
    Capability capability;

    size_t featureStores = 0;
    size_t dispStores = 0;

    BOOST_FOREACH (const Camera& cam, seq.GetCameras())
    {
        featureStores += cam.GetFeatureStores().size();
    }

    BOOST_FOREACH (const RectifiedStereo& pair, seq.GetRectifiedStereo())
    {
        dispStores += pair.GetDisparityStores().size();
    }

    capability.motion = featureStores > 0;
    capability.metric = dispStores > 0;
    capability.dense = false;

    return capability;
}

String MultiFrameFeatureIntegration::Pathway::ToString() const
{
    std::stringstream ss, s0, s1;

    if (srcFrameOffset != 0) s0 << std::showpos << srcFrameOffset;
    if (dstFrameOffset != 0) s1 << std::showpos << dstFrameOffset;

    ss << "(" << srcStoreIdx << ",t" << s0.str() << ") -> ";
    ss << "(" << dstStoreIdx << ",t" << s1.str() << ")";
        
    return ss.str();
}

bool seq2map::MultiFrameFeatureIntegration::Pathway::CheckRange(size_t t0, size_t tn, size_t t) const
{
    int ti = static_cast<int>(t) + srcFrameOffset;
    int tj = static_cast<int>(t) + dstFrameOffset;

    return (ti >= 0 && ti <= static_cast<int>(tn) &&
            tj >= 0 && tj <= static_cast<int>(tn));
}

bool MultiFrameFeatureIntegration::AddPathway(size_t f0, size_t f1, int dt0, int dt1)
{
    Pathway newPathway(f0, f1, dt0, dt1);

    if (f0 == f1 && dt0 == dt1)
    {
        E_WARNING << "invalid pathway " << newPathway.ToString();
        return false;
    }

    BOOST_FOREACH (const Pathway& pathway, m_pathways)
    {
        bool duplicated =
            pathway.srcStoreIdx == f0     &&
            pathway.dstStoreIdx == f1     &&
            pathway.srcFrameOffset == dt0 &&
            pathway.dstFrameOffset == dt1;

        if (duplicated)
        {
            E_WARNING << "duplicated pathway " << newPathway.ToString();
            return false;
        }
    }

    m_pathways.push_back(newPathway);
    return true;
}
    
bool MultiFrameFeatureIntegration::BindPathways(const Sequence& seq, size_t& maxStoreIndex)
{
    maxStoreIndex = 0;

    if (m_pathways.empty())
    {
        E_ERROR << "no pathway specified";
        return false;
    }

    BOOST_FOREACH (Pathway& pathway, m_pathways)
    {
        if (!seq.FindFeatureStore(pathway.srcStoreIdx, pathway.srcStore) ||
            !seq.FindFeatureStore(pathway.dstStoreIdx, pathway.dstStore))
        {
            E_ERROR << "cannot locate feature store(s) for pathway " << pathway.ToString();
            return false;
        }

        maxStoreIndex = maxStoreIndex > pathway.srcStoreIdx ? maxStoreIndex : pathway.srcStoreIdx;
        maxStoreIndex = maxStoreIndex > pathway.dstStoreIdx ? maxStoreIndex : pathway.dstStoreIdx;
    }

    return true;
}


bool MultiFrameFeatureIntegration::SLAM(const Sequence& seq, Map& map, size_t t0, size_t tn)
{
    bool valid = t0 >= 0 && tn > t0 && tn < seq.GetFrames();

    if (!valid)
    {
        E_ERROR << "invalid span of sequence (" << t0 << "," << tn << ")";
        E_ERROR << "the sequence has " << seq.GetFrames() << " frame(s)";

        return false;
    }

    size_t fmax;

    if (!BindPathways(seq, fmax))
    {
        E_ERROR << "error binding pathway(s) to sequence";
        return false;
    }

    FeatureMatcher matcher;

    for (size_t t = t0; t < tn; t++)
    {
        BOOST_FOREACH (const Pathway& pathway, m_pathways)
        {
            if (!pathway.CheckRange(t0, tn, t)) continue;

            const FeatureStore& Fi = *pathway.srcStore;
            const FeatureStore& Fj = *pathway.dstStore;

            Map::Frame& ti = map.GetFrame(t + pathway.srcFrameOffset);
            Map::Frame& tj = map.GetFrame(t + pathway.dstFrameOffset);

            ImageFeatureSet fi = Fi[ti.GetIndex()];
            ImageFeatureSet fj = Fj[tj.GetIndex()];
            ImageFeatureMap fmap = matcher.MatchFeatures(fi, fj);

            Map::Frame::IdList& ui = ti.featureIdLists[Fi.GetIndex()];
            Map::Frame::IdList& uj = tj.featureIdLists[Fj.GetIndex()];

            // initialise frame's feature ID list(s) for first time access
            if (ui.empty()) ui.resize(fi.GetSize(), INVALID_INDEX);
            if (uj.empty()) uj.resize(fj.GetSize(), INVALID_INDEX);

            BOOST_FOREACH (const FeatureMatch& m, fmap.GetMatches())
            {
                size_t ui_k = ui[m.srcIdx];
                size_t uj_k = uj[m.dstIdx];

                bool bi = ui_k == INVALID_INDEX;
                bool bj = uj_k == INVALID_INDEX;

                bool firstHit = bi == true && bj == true;
                bool converge = bi != true && bj != true && ui_k != uj_k;

                if (converge)
                {
                    /*
                    Landmark::Hits hi = map.GetLandmark(ui_k).hits;
                    Landmark::Hits hj = map.GetLandmark(uj_k).hits;

                    E_INFO << pathway.ToString() << " t=" << t << " multi-path : feature #" << ui_k << " meets #" << uj_k << "!!";
                    E_INFO << "#" << ui_k << ":";
                    
                    BOOST_FOREACH (Landmark::Hit h, hi)
                    {
                        E_INFO << "(" << h.store << "," << h.frame << ") " << h.proj;
                    }

                    E_INFO << "#" << uj_k << ":";

                    BOOST_FOREACH (Landmark::Hit h, hj)
                    {
                        E_INFO << "(" << h.store << "," << h.frame << ") " << h.proj;
                    }
                    */

                    switch (m_mergePolicy)
                    {
                    case MultiPathMergePolicy::REJECT:
                        //map.GetLandmark(ui_k).hits.clear();
                        //map.GetLandmark(uj_k).hits.clear();
                        break;

                    case MultiPathMergePolicy::KEEP_BEST:
                    case MultiPathMergePolicy::KEEP_BOTH:
                    case MultiPathMergePolicy::KEEP_LONGEST:
                    case MultiPathMergePolicy::KEEP_SHORTEST:
                        ui[m.srcIdx] = uj[m.dstIdx] = ui_k < uj_k ? ui_k : uj_k;
                        break;

                    case MultiPathMergePolicy::NO_MERGE:
                    default: // do nothing..
                        break;
                    }
                    continue;
                }

                Map::Landmark& lk = firstHit ? map.AddLandmark() : (bj ? map.GetLandmark(ui_k) : map.GetLandmark(uj_k));

                if (bi) lk.Hit(ti, Fi.GetIndex(), m.srcIdx).proj = fi[m.srcIdx].keypoint.pt;
                if (bj) lk.Hit(tj, Fj.GetIndex(), m.dstIdx).proj = fj[m.dstIdx].keypoint.pt;

                ui[m.srcIdx] = uj[m.dstIdx] = lk.GetIndex();
            }

            //E_INFO << ti.GetIndex() << "->" << tj.GetIndex() << ": " << matcher.Report();
        }
    }

    return true;
}

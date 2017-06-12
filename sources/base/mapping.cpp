#include <seq2map/mapping.hpp>

using namespace seq2map;

//==[ Map ]===================================================================//

void Map::RegisterSource(FeatureStore::ConstOwn& store)
{
    if (!store) return;
    GetSource(store->GetIndex()).store = store;
}

Landmark& Map::AddLandmark()
{
    return Dim0(m_newLandmarkId++);
}


//==[ Landmark ]==============================================================//

Hit& Landmark::Hit(Frame& frame, Source& src, size_t index)
{
    typedef Container* dnType;
    dnType d12[2];
    d12[0] = static_cast<dnType>(&frame);
    d12[1] = static_cast<dnType>(&src);

    return Insert(::Hit(index), d12);
}

//==[ FeatureMatching ]=======================================================//

bool FeatureMatching::operator() (Map& map, size_t t)
{
    Source& si = map.GetSource(src.store->GetIndex());
    Source& sj = map.GetSource(dst.store->GetIndex());

    if (!si.store) si.store = src.store;
    if (!sj.store) sj.store = dst.store;

    assert(src.store && dst.store);

    const FeatureStore& Fi = *src.store;
    const FeatureStore& Fj = *dst.store;

    Frame& ti = map.GetFrame(static_cast<size_t>(static_cast<int>(t) + src.offset));
    Frame& tj = map.GetFrame(static_cast<size_t>(static_cast<int>(t) + dst.offset));

    ImageFeatureSet fi = Fi[ti.GetIndex()];
    ImageFeatureSet fj = Fj[tj.GetIndex()];
    ImageFeatureMap fmap = matcher.MatchFeatures(fi, fj);

    Frame::IdList& ui = ti.featureIdLookup[Fi.GetIndex()];
    Frame::IdList& uj = tj.featureIdLookup[Fj.GetIndex()];

    // initialise frame's feature ID list(s) for first time access
    if (ui.empty()) ui.resize(fi.GetSize(), INVALID_INDEX);
    if (uj.empty()) uj.resize(fj.GetSize(), INVALID_INDEX);

    BOOST_FOREACH(const FeatureMatch& m, fmap.GetMatches())
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

            switch (policy)
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

        Landmark& lk = firstHit ? map.AddLandmark() : (bj ? map.GetLandmark(ui_k) : map.GetLandmark(uj_k));

        if (bi) lk.Hit(ti, si, m.srcIdx).proj = fi[m.srcIdx].keypoint.pt;
        if (bj) lk.Hit(tj, sj, m.dstIdx).proj = fj[m.dstIdx].keypoint.pt;

        ui[m.srcIdx] = uj[m.dstIdx] = lk.GetIndex();
    }

    E_INFO << ti.GetIndex() << "->" << tj.GetIndex() << ": " << matcher.Report();
    return true;
}

bool FeatureMatching::IsOkay() const
{
    return src.store && dst.store && !(src.store->GetIndex() == dst.store->GetIndex() && src.offset == dst.offset);
}

bool FeatureMatching::InRange(size_t t, size_t tn) const
{
    int ti = static_cast<int>(t) + src.offset;
    int tj = static_cast<int>(t) + dst.offset;

    return (ti >= 0 && ti <= static_cast<int>(tn) &&
            tj >= 0 && tj <= static_cast<int>(tn));
}

String FeatureMatching::ToString() const
{
    std::stringstream ss, f0, f1, s0, s1;

    if (src.offset != 0) s0 << std::showpos << src.offset;
    if (dst.offset != 0) s1 << std::showpos << dst.offset;

    if (src.store) f0 << src.store->GetIndex();
    else           f0 << "?";

    if (dst.store) f1 << dst.store->GetIndex();
    else           f1 << "?";

    ss << "(" << f0.str() << ",t" << s0.str() << ") -> ";
    ss << "(" << f1.str() << ",t" << s1.str() << ")";

    return ss.str();
}

//==[ MultiFrameFeatureIntegration ]==========================================//

bool MultiFrameFeatureIntegration::AddMatching(const FeatureMatching& matching)
{
    if (!matching.IsOkay())
    {
        E_WARNING << "invalid matching " << matching.ToString();
        return false;
    }

    BOOST_FOREACH (const FeatureMatching& m, m_matchings)
    {
        bool duplicated = (m.src == matching.src) && (m.dst == matching.dst);

        if (duplicated)
        {
            E_WARNING << "duplicated matching " << matching.ToString();
            return false;
        }
    }

    m_matchings.push_back(matching);
    return true;
}

Mapper::Capability MultiFrameFeatureIntegration::GetCapability() const
{
    Capability capability;

    capability.motion = m_matchings.size() > 0;
    capability.metric = m_dispStores.size() > 0;
    capability.dense = false;

    return capability;
}

bool MultiFrameFeatureIntegration::SLAM(Map& map, size_t t0, size_t tn)
{
    for (size_t t = t0; t < tn; t++)
    {
        BOOST_FOREACH (FeatureMatching& m, m_matchings)
        {
            if (!m(map, t))
            {
                E_ERROR << "error matching " << m.ToString();
                return false;
            }
        }
    }

    return true;
}

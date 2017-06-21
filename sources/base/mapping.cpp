#include <seq2map/mapping.hpp>

using namespace seq2map;

//==[ Map ]===================================================================//

Landmark& Map::AddLandmark()
{
    return Dim0(m_newLandmarkId++);
}

void Map::RemoveLandmark(Landmark& l)
{
    // update ID lookup table
    for (Landmark::const_iterator h = l.cbegin(); h; h++)
    {
        const Hit& hit = *h;

        Frame& t = h.GetContainer<1, Frame>();
        const Source& c = h.GetContainer<2, Source>();

        t.featureIdLookup[c.GetIndex()][hit.index] = INVALID_INDEX;
    }

    l.clear();
}

bool Map::IsJoinable(const Landmark& li, const Landmark& lj)
{
    AutoSpeedometreMeasure(joinChkMetre, 2);

    // parallel traversal
    Landmark::const_iterator hi = li.cbegin();
    Landmark::const_iterator hj = lj.cbegin();

    while (hi && hj)
    {
        const Frame& ti = hi.GetContainer<1, Frame>();
        const Frame& tj = hj.GetContainer<1, Frame>();

        if (ti < tj)
        {
            hi++;
        }
        else if (tj < ti)
        {
            hj++;
        }
        else
        {
            const Source& ci = hi.GetContainer<2, Source>();
            const Source& cj = hj.GetContainer<2, Source>();

            if (ci < cj)
            {
                hi++;
            }
            else if (cj < ci)
            {
                hj++;
            }
            else
            {
                return false;
            }
        }
    }

    return true;
}

Landmark& Map::JoinLandmark(Landmark& li, Landmark& lj)
{
    AutoSpeedometreMeasure(joinMetre, 2);

    for (Landmark::const_iterator hj = lj.cbegin(); hj; hj++)
    {
        const Hit& hit = *hj;

        Frame&  tj = hj.GetContainer<1, Frame>();
        Source& cj = hj.GetContainer<2, Source>();

        li.Hit(tj, cj, hit.index) = hit;

        // ID table rewriting
        Frame::IdList& uj = tj.featureIdLookup[cj.GetIndex()];
        uj[hit.index] = li.GetIndex();
    }

    lj.clear(); // abondaned

    return li;
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

    //if (tj.GetIndex() == 1 && sj.GetIndex() == 1)
    //    BOOST_FOREACH(const FeatureMatch& m, fmap.GetMatches())
    //{
    //    E_INFO << m.srcIdx << " -> " << m.dstIdx << "[" << m.state << "]";
    //}

    BOOST_FOREACH(const FeatureMatch& m, fmap.GetMatches())
    {
        if (!(m.state & FeatureMatch::INLIER))
        {
            continue;
        }

        size_t& ui_k = ui[m.srcIdx];
        size_t& uj_k = uj[m.dstIdx];

        bool bi = (ui_k == INVALID_INDEX);
        bool bj = (uj_k == INVALID_INDEX);

        bool firstHit = bi == true && bj == true;
        bool converge = bi != true && bj != true && ui_k != uj_k;

        //if (ti.GetIndex() > 0)
        //E_INFO << "(" << ti.GetIndex() << "," << si.GetIndex() << "," << m.srcIdx << ") -> (" << tj.GetIndex() << "," << sj.GetIndex() << "," << m.dstIdx << ")";

        if (!converge)
        {
            Landmark& lk = firstHit ? map.AddLandmark() : (bj ? map.GetLandmark(ui_k) : map.GetLandmark(uj_k));

            if (bi) lk.Hit(ti, si, m.srcIdx).proj = fi[m.srcIdx].keypoint.pt;
            if (bj) lk.Hit(tj, sj, m.dstIdx).proj = fj[m.dstIdx].keypoint.pt;

            ui_k = uj_k = lk.GetIndex();

            //E_INFO << "ui[" << m.srcIdx << "] = " << lk.GetIndex();
            //E_INFO << "uj[" << m.dstIdx << "] = " << lk.GetIndex();

            continue;
        }

        //E_INFO << "(" << ti.GetIndex() << "," << si.GetIndex() << "," << m.srcIdx << ") -> (" << tj.GetIndex() << "," << sj.GetIndex() << "," << m.dstIdx << ")";
        //E_INFO << "multipath detected (" << ui_k << "," << uj_k << ")";

        if (policy == ConflictResolution::NO_MERGE)
        {
            continue;
        }

        //E_INFO << map.joinChkMetre.ToString();
        //E_INFO << map.joinMetre.ToString();

        Landmark& li = map.GetLandmark(ui_k);
        Landmark& lj = map.GetLandmark(uj_k);

        if (policy == ConflictResolution::KEEP_BOTH || map.IsJoinable(li, lj))
        {
            map.JoinLandmark(li, lj);
            continue;
        }

        switch (policy)
        {
        case ConflictResolution::KEEP_BEST:
            throw std::exception("not implemented");
            break;

        case ConflictResolution::KEEP_LONGEST:
            throw std::exception("not implemented");
            break;

        case ConflictResolution::KEEP_SHORTEST:
            throw std::exception("not implemented");
            break;

        case ConflictResolution::REMOVE_BOTH:
            map.RemoveLandmark(li);
            map.RemoveLandmark(lj);
            break;
        }
    }

    E_INFO << "(" << ti.GetIndex() << "," << si.GetIndex() << ") -> (" << tj.GetIndex() << "," << sj.GetIndex() << ") : " << matcher.Report();
    return true;
}

bool FeatureMatching::IsOkay() const
{
    return src.store && dst.store && !(src.store->GetIndex() == dst.store->GetIndex() && src.offset == dst.offset);
}

bool FeatureMatching::IsCrossed() const
{
    Camera::ConstOwn cam0, cam1;
    return IsOkay() && (cam0 = src.store->GetCamera()) && (cam1 = dst.store->GetCamera()) && cam0->GetIndex() != cam1->GetIndex();
}

bool FeatureMatching::InRange(size_t t, size_t tn) const
{
    int ti = static_cast<int>(t) + src.offset;
    int tj = static_cast<int>(t) + dst.offset;

    return (ti >= 0 && ti < static_cast<int>(tn) &&
            tj >= 0 && tj < static_cast<int>(tn));
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

    capability.motion = false;
    capability.metric = m_dispStores.size() > 0;
    capability.dense = false;

    BOOST_FOREACH (const FeatureMatching& m, m_matchings)
    {
        capability.motion |= !m.IsSynchronised();
        capability.metric |= m.IsCrossed();
    }

    return capability;
}

bool MultiFrameFeatureIntegration::SLAM(Map& map, size_t t0, size_t tn)
{
    for (size_t t = t0; t < tn; t++)
    {
        BOOST_FOREACH (FeatureMatching& m, m_matchings)
        {
            if (m.InRange(t, tn) && !m(map, t))
            {
                E_ERROR << "error matching " << m.ToString();
                return false;
            }
        }
    }

    return true;
}

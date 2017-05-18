#include <seq2map/mapping.hpp>

using namespace seq2map;

Hit& Landmark::Hit(Frame& frame, size_t store, size_t index)
{
    return Insert(frame, ::Hit(*this, frame, store, index));
}

Landmark& Map::AddLandmark()
{
    return Row(m_newLandmarkId++);
}

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

            Frame& ti = map.GetFrame(t + pathway.srcFrameOffset);
            Frame& tj = map.GetFrame(t + pathway.dstFrameOffset);

            ImageFeatureSet fi = Fi[ti.GetIndex()];
            ImageFeatureSet fj = Fj[tj.GetIndex()];
            ImageFeatureMap fmap = matcher.MatchFeatures(fi, fj);

            Frame::IdList& ui = ti.featureIdLookup[Fi.GetIndex()];
            Frame::IdList& uj = tj.featureIdLookup[Fj.GetIndex()];

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

                Landmark& lk = firstHit ? map.AddLandmark() : (bj ? map.GetLandmark(ui_k) : map.GetLandmark(uj_k));

                if (bi) lk.Hit(ti, Fi.GetIndex(), m.srcIdx).proj = fi[m.srcIdx].keypoint.pt;
                if (bj) lk.Hit(tj, Fj.GetIndex(), m.dstIdx).proj = fj[m.dstIdx].keypoint.pt;

                ui[m.srcIdx] = uj[m.dstIdx] = lk.GetIndex();
            }

            E_INFO << ti.GetIndex() << "->" << tj.GetIndex() << ": " << matcher.Report();
        }
    }

    return true;
}

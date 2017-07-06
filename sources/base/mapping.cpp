#include <seq2map/mapping.hpp>

using namespace seq2map;

//==[ Map ]===================================================================//

Landmark& Map::AddLandmark()
{
    return Dim0(m_newLandmarkId++);
}

void Map::RemoveLandmark(Landmark& l)
{
    // unlink all references in lookup tables
    for (Landmark::const_iterator h = l.cbegin(); h; h++)
    {
        const Hit& hit = *h;

        Frame& t = h.GetContainer<1, Frame>();
        const Source& c = h.GetContainer<2, Source>();

        t.featureLandmarkLookup[c.GetIndex()][hit.index] = NULL;
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
        Landmark::Ptrs& uj = tj.featureLandmarkLookup[cj.GetIndex()];
        uj[hit.index] = &li;
    }

    lj.clear(); // abondaned

    return li;
}

StructureEstimation::Estimate Map::UpdateStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& g, const EuclideanTransform& ref)
{
    assert(g.structure.shape == Geometry::ROW_MAJOR);
    assert(g.structure.GetElements() == u.size());
    assert(g.structure.GetDimension() == 3);

    typedef cv::Vec6d Covar6D; // 6 coefficients of full 3D covariance matrix

    cv::Mat mat = cv::Mat(u.size(), 3, g.structure.mat.depth()).reshape(3);
    cv::Mat cov = cv::Mat(u.size(), 6, g.structure.mat.depth()).reshape(6);

    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL)
        {
            mat.at<Point3D>(k) = Point3D(0, 0, 0);
            cov.at<Covar6D>(k) = Covar6D(0, 0, 0, 0, 0, 0);
        }
        else
        {
            const Landmark& lk = *u[k];
            const Landmark::Covar3D ck = lk.cov;

            mat.at<Point3D>(k) = lk.position;
            cov.at<Covar6D>(k) = Covar6D(ck.xx, ck.xy, ck.xz, ck.yy, ck.yz, ck.zz);
        }
    }

    StructureEstimation::Estimate g0(
        Geometry(Geometry::ROW_MAJOR, mat),
        Metric::Own(new MahalanobisMetric(MahalanobisMetric::ANISOTROPIC_ROTATED, 3, cov))
    );

    // state update
    g0 += g.Transform(ref.GetInverse());

    // write back to the landmarks
    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL) continue;

        Landmark& lk = *u[k];

        lk.position = mat.at<Point3D>(k);
        lk.cov      = mat.at<Covar6D>(k);
    }

    return g0.Transform(ref);
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

//==[ MultiObjectiveOutlierFilter ]==========================================//

bool MultiObjectiveOutlierFilter::operator() (ImageFeatureMap& fmap, Indices& inliers)
{
    if (builders.empty())
    {
        E_WARNING << "no objective builder available";
        return false;
    }

    //
    // build objectives
    //
    const ImageFeatureSet& Fi = fmap.From();
    const ImageFeatureSet& Fj = fmap.To();

    size_t idx = 0;
    BOOST_FOREACH (size_t k, inliers)
    {
        const FeatureMatch& m = fmap[k];

        BOOST_FOREACH (ObjectiveBuilder::Own builder, builders)
        {
            builder->AddData(k, Fi[m.srcIdx], Fj[m.dstIdx], idx);
        }

        idx++;
    }

    //
    // apply outlier detection on the built models
    //
    ConsensusPoseEstimator estimator;
    PoseEstimator::Estimate estimate;
    PoseEstimator::ConstOwn solver;
    GeometricMapping solverData;

    BOOST_FOREACH (ObjectiveBuilder::Own builder, builders)
    {
        GeometricMapping data;
        AlignmentObjective::InlierSelector selector;
        
        if (builder->Build(data, selector))
        {
            E_WARNING << "objective building failed for " << builder->ToString();
            continue;
        }

        estimator.AddSelector(selector);

        if (!solver)
        {
            solver = builder->GetSolver();
            solverData = data;
        }
    }

    if (!solver)
    {
        E_WARNING << "no objective succesfully built";
        return false;
    }

    estimator.SetStrategy(ConsensusPoseEstimator::RANSAC);
    estimator.SetMaxIterations(30);
    estimator.SetMinInlierRatio(0.75f);
    estimator.SetConfidence(0.8f);
    estimator.SetSolver(solver);

    std::vector<Indices> survived, eliminated;

    if (!estimator(solverData, estimate, survived, eliminated))
    {
        E_ERROR << "consensus outlier detection failed";
        return false;
    }

    // aggregate all inliers from all the selectors
    std::vector<size_t> hits(inliers.size(), 0);

    for (size_t s = 0; s < survived.size(); s++)
    {
        const std::vector<size_t>& idmap = estimator.GetSelectors()[s].objective->GetData().indices;
        BOOST_FOREACH (size_t j, survived[s])
        {
            hits[idmap[j]]++;
        }
    }

    std::vector<size_t>::const_iterator h = hits.begin();
    for (Indices::iterator itr = inliers.begin(); itr != inliers.end(); itr++)
    {
        if (*h < estimator.GetSelectors().size())
        {
            fmap[*itr].Reject(FeatureMatch::GEOMETRIC_TEST_FAILED);
            inliers.erase(itr);
        }

        h++;
    }

    // estimate.pose might be useful..
    // ...
    // ..
    // .
}

//==[ FeatureTracker ]=======================================================//

bool FeatureTracker::operator() (Map& map, size_t t)
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
    Landmark::Ptrs& ui = ti.featureLandmarkLookup[Fi.GetIndex()];
    Landmark::Ptrs& uj = tj.featureLandmarkLookup[Fj.GetIndex()];
    StructureEstimation::Estimate gi(Geometry::ROW_MAJOR);
    StructureEstimation::Estimate gj(Geometry::ROW_MAJOR);

    // initialise frame's feature-landmark lookup table for first time access
    if (ui.empty()) ui.resize(fi.GetSize(), NULL);
    if (uj.empty()) uj.resize(fj.GetSize(), NULL);

    // get feature structure from depthmap
    if (src.disp && src.disp->GetStereoPair()) gi = GetFeatureStructure(fi, src.disp->GetStereoPair()->Backproject((*src.disp)[ti.GetIndex()].im, cv::Mat()));
    if (dst.disp && dst.disp->GetStereoPair()) gj = GetFeatureStructure(fj, dst.disp->GetStereoPair()->Backproject((*dst.disp)[tj.GetIndex()].im, cv::Mat()));

    // perform pre-motion structure update
    if (ti.poseEstimate.valid) gi = map.UpdateStructure(ui, gi, ti.poseEstimate.pose);
    if (tj.poseEstimate.valid) gj = map.UpdateStructure(uj, gj, tj.poseEstimate.pose);

    // build the outlier filter and filtering models
    if (outlierRejection)
    {
        Camera::ConstOwn ci = Fi.GetCamera();
        Camera::ConstOwn cj = Fj.GetCamera();
        ProjectionModel::ConstOwn pi = ci ? ci->GetPosedProjection() : ProjectionModel::ConstOwn();
        ProjectionModel::ConstOwn pj = cj ? cj->GetPosedProjection() : ProjectionModel::ConstOwn();

        cv::Mat Ii = Fi.GetCamera()->GetImageStore()[ti.GetIndex()].im;
        cv::Mat Ij = Fj.GetCamera()->GetImageStore()[tj.GetIndex()].im;

        MultiObjectiveOutlierFilter* filter = new MultiObjectiveOutlierFilter();

        if (outlierRejection & FORWARD_PROJECTION)
        {
            if (pj)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(new PerspectiveObjectiveBuilder(pj, gi, true)));
            }
            else
            {
                E_WARNING << "error adding forward projection objective to the outlier filter due to missing projection model";
                E_WARNING << "forward projection-based outlier rejection deactivated for " << ToString();

                outlierRejection &= ~FORWARD_PROJECTION;
            }
        }

        if (outlierRejection & BACKWARD_PROJECTION)
        {
            if (pi)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(new PerspectiveObjectiveBuilder(pi, gj, false)));
            }
            else
            {
                E_WARNING << "error adding backward projection objective to the outlier filter due to missing projection model";
                E_WARNING << "backward projection-based outlier rejection deactivated for " << ToString();

                outlierRejection &= ~BACKWARD_PROJECTION;
            }
        }

        if (outlierRejection & EPIPOLAR)
        {
            if (pi && pj)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(new EpipolarObjectiveBuilder(pi, pj)));
            }
            else
            {
                E_WARNING << "error adding epipolar objective to the outlier filter due to missing projection model(s)";
                E_WARNING << "epipolar-based outlier rejection deactivated for " << ToString();

                outlierRejection &= ~EPIPOLAR;
            }
        }

        matcher.GetFilters().push_back(FeatureMatcher::Filter::Own(filter));
    }

    ImageFeatureMap fmap = matcher(fi, fj);

    // dispose the outlier filter
    if (outlierRejection)
    {
        matcher.GetFilters().pop_back();
    }

    //cv::Mat im = imfuse(Ii, Ij);
    //fmap.Draw(im);
    //
    //cv::imshow(ToString(), im);
    //cv::waitKey(0);

    // insert the observations into the map
    BOOST_FOREACH(const FeatureMatch& m, fmap.GetMatches())
    {
        if (!(m.state & FeatureMatch::INLIER))
        {
            continue;
        }

        Landmark*& ui_k = ui[m.srcIdx];
        Landmark*& uj_k = uj[m.dstIdx];

        bool bi = (ui_k == NULL);
        bool bj = (uj_k == NULL);

        bool firstHit = bi == true && bj == true;
        bool converge = bi != true && bj != true && ui_k != uj_k;

        //if (ti.GetIndex() > 0)
        //E_INFO << "(" << ti.GetIndex() << "," << si.GetIndex() << "," << m.srcIdx << ") -> (" << tj.GetIndex() << "," << sj.GetIndex() << "," << m.dstIdx << ")";

        if (!converge)
        {
            Landmark& lk = firstHit ? map.AddLandmark() : (bj ? *ui_k : *uj_k);

            if (bi) lk.Hit(ti, si, m.srcIdx).proj = fi[m.srcIdx].keypoint.pt;
            if (bj) lk.Hit(tj, sj, m.dstIdx).proj = fj[m.dstIdx].keypoint.pt;

            ui_k = uj_k = &lk;

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

        Landmark& li = *ui_k;
        Landmark& lj = *uj_k;

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

StructureEstimation::Estimate FeatureTracker::GetFeatureStructure(const ImageFeatureSet& f, const StructureEstimation::Estimate& structure)
{
    Indices indices;
    const cv::Size size = structure.structure.mat.size();

    // convert subscripts to rounded integer indices
    for (size_t k = 0; k < f.GetSize(); k++)
    {
        const Point2F& sub = f[k].keypoint.pt;
        const size_t ind = static_cast<size_t>(std::round(sub.y) * size.width + std::round(sub.x));

        indices.push_back(ind);
    }

    return structure[indices];
}

bool FeatureTracker::IsOkay() const
{
    return src.store && dst.store && !(src.store->GetIndex() == dst.store->GetIndex() && src.offset == dst.offset);
}

bool FeatureTracker::IsCrossed() const
{
    Camera::ConstOwn cam0, cam1;
    return IsOkay() && (cam0 = src.store->GetCamera()) && (cam1 = dst.store->GetCamera()) && cam0->GetIndex() != cam1->GetIndex();
}

bool FeatureTracker::InRange(size_t t, size_t tn) const
{
    int ti = static_cast<int>(t) + src.offset;
    int tj = static_cast<int>(t) + dst.offset;

    return (ti >= 0 && ti < static_cast<int>(tn) &&
            tj >= 0 && tj < static_cast<int>(tn));
}

String FeatureTracker::ToString() const
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

//==[ FeatureTracker::EpipolarOutlierModel ]=================================//

void FeatureTracker::EpipolarObjectiveBuilder::AddData(size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    m_builder.Add(fi.keypoint.pt, fj.keypoint.pt, localIdx);
}

bool FeatureTracker::EpipolarObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new EpipolarObjective(pi, pj));
    data = m_builder.Build();
    
    if (!objective->SetData(data))
    {
        E_WARNING << "error setting epipolar constraints for " << ToString();
        return false;
    }

    selector = objective->GetSelector(0.1f);
    return true;
}

//==[ FeatureTracker::ProjectiveOutlierModel ]===============================//

void FeatureTracker::PerspectiveObjectiveBuilder::AddData(size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    const ImageFeature& fk = forward ? fj : fi;
    const Point3D& gk = g.structure.mat.at<Point3D>(k);

    m_builder.Add(gk, fk.keypoint.pt, localIdx);
}

bool FeatureTracker::PerspectiveObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new ProjectionObjective(p, forward));
    data = m_builder.Build();

    if (!objective->SetData(data))
    {
        E_WARNING << "error setting perspective constraints for " << ToString();
        return false;
    }

    selector = objective->GetSelector(1.0f);
    return true;
}

PoseEstimator::Own FeatureTracker::PerspectiveObjectiveBuilder::GetSolver()
{
    PoseEstimator::Own estimator = PoseEstimator::Own(new PerspevtivePoseEstimator(p));
    return forward ? estimator : PoseEstimator::Own(new InversePoseEstimator(estimator));
}

//==[ MultiFrameFeatureIntegration ]==========================================//

bool MultiFrameFeatureIntegration::AddTracking(const FeatureTracker& matching)
{
    if (!matching.IsOkay())
    {
        E_WARNING << "invalid matching " << matching.ToString();
        return false;
    }

    BOOST_FOREACH (const FeatureTracker& m, m_tracking)
    {
        bool duplicated = (m.src == matching.src) && (m.dst == matching.dst);

        if (duplicated)
        {
            E_WARNING << "duplicated matching " << matching.ToString();
            return false;
        }
    }

    m_tracking.push_back(matching);
    return true;
}

Mapper::Capability MultiFrameFeatureIntegration::GetCapability() const
{
    Capability capability;

    capability.motion = false;
    capability.metric = m_dispStores.size() > 0;
    capability.dense = false;

    BOOST_FOREACH (const FeatureTracker& m, m_tracking)
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
        BOOST_FOREACH (FeatureTracker& f, m_tracking)
        {
            if (f.InRange(t, tn) && !f(map, t))
            {
                E_ERROR << "error matching " << f.ToString();
                return false;
            }
        }


    }

    return true;
}

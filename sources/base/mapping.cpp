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

StructureEstimation::Estimate Map::GetStructure(const Landmark::Ptrs& u) const
{
    typedef cv::Vec6d Covar6D; // 6 coefficients of full 3D covariance matrix

    cv::Mat mat = cv::Mat(static_cast<int>(u.size()), 3, CV_64F).reshape(3);
    cv::Mat cov = cv::Mat(static_cast<int>(u.size()), 6, CV_64F).reshape(6);

    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL)
        {
            mat.at<Point3D>(static_cast<int>(k)) = Point3D(0, 0, 0);
            cov.at<Covar6D>(static_cast<int>(k)) = Covar6D(0, 0, 0, 0, 0, 0);
        }
        else
        {
            const Landmark& lk = *u[k];
            const Landmark::Covar3D& ck = lk.cov;

            mat.at<Point3D>(static_cast<int>(k)) = lk.position;
            cov.at<Covar6D>(static_cast<int>(k)) = Covar6D(ck.xx, ck.xy, ck.xz, ck.yy, ck.yz, ck.zz);
        }
    }

    return StructureEstimation::Estimate(
        Geometry(Geometry::ROW_MAJOR, mat.reshape(1)),
        Metric::Own(new MahalanobisMetric(MahalanobisMetric::ANISOTROPIC_ROTATED, 3, cov.reshape(1)))
    );

    /*
    cv::Mat mat = cv::Mat(static_cast<int>(u.size()), 3, CV_64F).reshape(3);
    cv::Mat icv = cv::Mat(static_cast<int>(u.size()), 1, CV_64F);

    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL)
        {
            mat.at<Point3D>(static_cast<int>(k)) = Point3D(0, 0, 0);
            icv.at <double>(static_cast<int>(k)) = 0;
        }
        else
        {
            const Landmark& lk = *u[k];
            const Landmark::Covar3D ck = lk.cov;

            mat.at<Point3D>(static_cast<int>(k)) = lk.position;
            icv.at<double> (static_cast<int>(k)) = lk.icv;
        }
    }

    return StructureEstimation::Estimate(
        Geometry(Geometry::ROW_MAJOR, mat.reshape(1)),
        Metric::Own(new WeightedEuclideanMetric(icv))
    );
    */
}

void Map::SetStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& structure)
{
    typedef cv::Vec6d Covar6D; // 6 coefficients of full 3D covariance matrix

    boost::shared_ptr<MahalanobisMetric> metric
        = boost::dynamic_pointer_cast<MahalanobisMetric, Metric>(structure.metric);

    if (!metric || metric->type != MahalanobisMetric::ANISOTROPIC_ROTATED)
    {
        E_ERROR << "structure has to be equipped with a full Mahalanobis metric";
        return;
    }

    const cv::Mat& mat = structure.structure.mat.reshape(3);
    const cv::Mat& cov = metric->GetCovariance().mat.reshape(6);

    // write back to the landmarks
    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL) continue;

        Landmark& lk = *u[k];

        lk.position = mat.at<Point3D>(static_cast<int>(k));
        lk.cov      = cov.at<Covar6D>(static_cast<int>(k));
    }

    /*
    const WeightedEuclideanMetric* metric =
        dynamic_cast<const WeightedEuclideanMetric*>(structure.metric.get());

    if (!metric)
    {
        E_ERROR << "structure has to be equipped with a weighted Euclidean metric";
        return;
    }

    const cv::Mat& mat = structure.structure.mat.reshape(3);
    const cv::Mat& icv = metric->GetWeights();

    // write back to the landmarks
    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL) continue;

        Landmark& lk = *u[k];

        lk.position = mat.at<Point3D>(static_cast<int>(k));
        lk.icv      = icv.at<double> (static_cast<int>(k));
    }
    */
}

StructureEstimation::Estimate Map::UpdateStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& g, const EuclideanTransform& ref)
{
    if (u.empty())
    {
        return StructureEstimation::Estimate(Geometry::ROW_MAJOR);
    }

    if (g.structure.IsEmpty())
    {
        return GetStructure(u).Transform(ref);
    }

    assert(g.structure.shape == Geometry::ROW_MAJOR);
    assert(g.structure.GetElements() == u.size());
    assert(g.structure.GetDimension() == 3);

    StructureEstimation::Estimate g0 = GetStructure(u); // extract structure of the involved landmarks
    g0 += g.Transform(ref.GetInverse()); // state update, recursive Bayesian
    SetStructure(u, g0);                 // write back the extracted structure

    return g0.Transform(ref);
}

Landmark::Ptrs Map::GetLandmarks(std::vector<size_t> indices)
{
    Landmark::Ptrs u;
    u.reserve(indices.size());

    BOOST_FOREACH (size_t idx, indices)
    {
        u.push_back(&GetLandmark(idx));
    }

    return u;
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
            builder->AddData(m.srcIdx, m.dstIdx, k, Fi[m.srcIdx], Fj[m.dstIdx], idx);
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

    // to trace the population and inliers in each model
    std::vector<AlignmentObjective::InlierSelector::Stats*> stats;

    if (motion.valid)
    {
        solver = PoseEstimator::ConstOwn(new DummyPoseEstimator(motion.pose));
    }

    BOOST_FOREACH (ObjectiveBuilder::Own builder, builders)
    {
        GeometricMapping data;
        AlignmentObjective::InlierSelector selector;
        
        if (!builder->Build(data, selector, sigma))
        {
            E_WARNING << "objective \"" << builder->ToString() << "\" building failed";
            continue;
        }

        E_TRACE << "objective \"" << builder->ToString() << "\" built from " << data.GetSize() << " correspondence(s)";

        estimator.AddSelector(selector);
        stats.push_back(&builder->stats);

        if (!solver)
        {
            E_TRACE << "main solver set to \"" << builder->ToString() << "\"";

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
    estimator.SetMaxIterations(motion.valid ? 1 : maxIterations);
    estimator.SetMinInlierRatio(minInlierRatio);
    estimator.SetConfidence(confidence);
    estimator.SetSolver(solver);
    estimator.SetVerbose(true);

    if (optimisation && !motion.valid)
    {
        estimator.EnableOptimisation();
    }
    else
    {
        estimator.DisableOptimisation();
    }

    std::vector<Indices> survived, eliminated;

    if (!estimator(solverData, estimate, survived, eliminated))
    {
        E_ERROR << "consensus outlier detection failed";
    }

    // aggregate all inliers from all the selectors
    std::vector<size_t> hits(inliers.size(), 0);

    for (size_t s = 0; s < survived.size(); s++)
    {
        const AlignmentObjective::InlierSelector& sel = estimator.GetSelectors()[s];
        const std::vector<size_t>& idmap = sel.objective->GetData().indices;

        if (stats[s] != NULL)
        {
            stats[s]->population = idmap.size();
            stats[s]->inliers = survived[s].size();
            stats[s]->secs += sel.metre.GetElapsedSeconds();
        }

        BOOST_FOREACH (size_t j, survived[s])
        {
            hits[idmap[j]]++;
        }
    }

    Indices::iterator itr = inliers.begin();
    for (std::vector<size_t>::const_iterator h = hits.begin(); h != hits.end(); h++)
    {
        if (*h < estimator.GetSelectors().size())
        {
            fmap[*itr].Reject(FeatureMatch::GEOMETRIC_TEST_FAILED);
            inliers.erase(itr++);
        }
        else
        {
            itr++;
        }
    }

    // estimate.pose might be useful..
    if (estimate.valid && !motion.valid)
    {
        motion = estimate;
    }

    return estimate.valid;
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

    Camera::ConstOwn ci = Fi.GetCamera();
    Camera::ConstOwn cj = Fj.GetCamera();
    ProjectionModel::ConstOwn pi = ci ? ci->GetPosedProjection() : ProjectionModel::ConstOwn();
    ProjectionModel::ConstOwn pj = cj ? cj->GetPosedProjection() : ProjectionModel::ConstOwn();
    cv::Mat Ii = ci ? ci->GetImageStore()[ti.GetIndex()].im : cv::Mat();
    cv::Mat Ij = cj ? cj->GetImageStore()[tj.GetIndex()].im : cv::Mat();
    cv::Mat Di = src.disp ? (*src.disp)[ti.GetIndex()].im : cv::Mat();
    cv::Mat Dj = dst.disp ? (*dst.disp)[tj.GetIndex()].im : cv::Mat();

    PoseEstimator::Estimate& mi = ti.poseEstimate;
    PoseEstimator::Estimate& mj = tj.poseEstimate;
    StructureEstimation::Estimate gi(Geometry::ROW_MAJOR);
    StructureEstimation::Estimate gj(Geometry::ROW_MAJOR);

    boost::shared_ptr<MultiObjectiveOutlierFilter> filter;
    GeometricMapping::ImageToImageBuilder flow;

    // initialise statistics
    stats = Stats();

    // append augmented feature sets
    fi.Append(ti.augmentedFeaturs[Fi.GetIndex()]);
    fj.Append(tj.augmentedFeaturs[Fj.GetIndex()]);

    // initialise frame's feature-landmark lookup table for first time access
    if (ui.empty()) ui.resize(fi.GetSize(), NULL);
    if (uj.empty()) uj.resize(fj.GetSize(), NULL);

    // get feature structure from depthmap and transform it to the reference camera's coordinates system
    //if (ti.GetIndex() == 0)
    //{
    if (!Di.empty() && src.disp->GetStereoPair()) gi = src.disp->GetStereoPair()->Backproject(Di, cv::Mat(), GetFeatureImageIndices(fi, Di.size())).Transform(ci->GetExtrinsics().GetInverse());
    if (!Dj.empty() && dst.disp->GetStereoPair()) gj = dst.disp->GetStereoPair()->Backproject(Dj, cv::Mat(), GetFeatureImageIndices(fj, Dj.size())).Transform(cj->GetExtrinsics().GetInverse());
    //}

    // perform pre-motion structure update
    if (mi.valid) gi = map.UpdateStructure(ui, gi, mi.pose);
    if (mj.valid) gj = map.UpdateStructure(uj, gj, mj.pose);

    // build the outlier filter and models
    if (outlierRejection.scheme)
    {
        filter = boost::shared_ptr<MultiObjectiveOutlierFilter>(
            new MultiObjectiveOutlierFilter(outlierRejection.maxIterations, outlierRejection.minInlierRatio, outlierRejection.confidence, outlierRejection.sigma)
        );

        if (ti == tj)
        {
            filter->motion.pose = EuclideanTransform::Identity;
            filter->motion.valid = true;
        }
        else if (ti.poseEstimate.valid && tj.poseEstimate.valid)
        {
            filter->motion.pose = mi.pose.GetInverse() >> mj.pose;
            filter->motion.valid = true;
        }

        if (outlierRejection.scheme & FORWARD_PROJ_ALIGN)
        {
            if (pj)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                    new PerspectiveObjectiveBuilder(pj, gi, true, stats.objectives[FORWARD_PROJ_ALIGN])
                ));
            }
            else
            {
                E_WARNING << "error adding forward projection objective to the outlier filter due to missing projection model";
                E_WARNING << "forward projection-based outlier rejection deactivated for " << ToString();

                outlierRejection.scheme &= ~FORWARD_PROJ_ALIGN;
            }
        }

        if (outlierRejection.scheme & BACKWARD_PROJ_ALIGN)
        {
            if (pi)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                    new PerspectiveObjectiveBuilder(pi, gj, false, stats.objectives[BACKWARD_PROJ_ALIGN])
                ));
            }
            else
            {
                E_WARNING << "error adding backward projection objective to the outlier filter due to missing projection model";
                E_WARNING << "backward projection-based outlier rejection deactivated for " << ToString();

                outlierRejection.scheme &= ~BACKWARD_PROJ_ALIGN;
            }
        }

        if (outlierRejection.scheme & PHOTOMETRIC_ALIGN)
        {
            if (pj)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                    new PhotometricObjectiveBuilder(pj, gi, Ii, Ij, stats.objectives[PHOTOMETRIC_ALIGN])
                ));
            }
            else
            {
                E_WARNING << "error adding photometric objective to the outlier filter due to missing projection model";
                E_WARNING << "photometric outlier rejection deactivated for " << ToString();

                outlierRejection.scheme &= ~PHOTOMETRIC_ALIGN;
            }
        }

        if (outlierRejection.scheme & RIGID_ALIGN)
        {
            filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                new RigidObjectiveBuilder(gi, gj, stats.objectives[RIGID_ALIGN])
            ));
        }

        if (outlierRejection.scheme & EPIPOLAR_ALIGN)
        {
            if (pi && pj)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                    new EpipolarObjectiveBuilder(pi, pj, outlierRejection.epipolarEps, stats.objectives[EPIPOLAR_ALIGN])
                ));
            }
            else
            {
                E_WARNING << "error adding epipolar objective to the outlier filter due to missing projection model(s)";
                E_WARNING << "epipolar-based outlier rejection deactivated for " << ToString();

                outlierRejection.scheme &= ~EPIPOLAR_ALIGN;
            }
        }

        matcher.GetFilters().push_back(FeatureMatcher::Filter::Own(filter));
    }

    ImageFeatureMap fmap = matcher(fi, fj);

    // dispose the outlier filter and apply the solved ego-motion whenever useful
    if (outlierRejection.scheme)
    {
        matcher.GetFilters().pop_back();

        if (!filter->motion.valid)
        {
            E_ERROR << "all features lost";
            return false;
        }

        stats.motion = filter->motion;

        if (ti != tj)
        {
            if (mi.valid && !mj.valid)
            {
                mj.pose  = mi.pose >> filter->motion.pose;
                mj.valid = true;
            }

            if (mj.valid && !mi.valid) // i.e. "else if (mj.valid) .."
            {
                mi.pose  = mj.pose >> filter->motion.pose.GetInverse();
                mi.valid = true;
            }
        }
    }

    // insert the observations into the map
    std::vector<bool> qi(fi.GetSize());
    std::vector<bool> qj(fj.GetSize());
    const FeatureMatches& matches = fmap.GetMatches();

    E_INFO << matches.size() << " matches..";

    for (size_t k = 0; k < matches.size(); k++)
    {
        const FeatureMatch& m = matches[k];

        if (!(m.state & FeatureMatch::INLIER))
        {
            continue;
        }

        qi[m.srcIdx] = qj[m.dstIdx] = true;

        Landmark*& ui_k = ui[m.srcIdx];
        Landmark*& uj_k = uj[m.dstIdx];

        bool bi = (ui_k == NULL);
        bool bj = (uj_k == NULL);

        bool firstHit = bi == true && bj == true;
        bool converge = bi != true && bj != true && ui_k != uj_k;

        if (!converge)
        {
            Landmark& lk = firstHit ? map.AddLandmark() : (bj ? *ui_k : *uj_k);

            if (bi) lk.Hit(ti, si, m.srcIdx).proj = fi[m.srcIdx].keypoint.pt;
            if (bj) lk.Hit(tj, sj, m.dstIdx).proj = fj[m.dstIdx].keypoint.pt;

            ui_k = uj_k = &lk;

            if (firstHit)
            {
                stats.spawned++;
            }
            else
            {
                stats.tracked++;
            }

            flow.Add(fi[m.srcIdx].keypoint.pt, fj[m.dstIdx].keypoint.pt, lk.GetIndex());

            continue;
        }

        if (policy == ConflictResolution::NO_MERGE)
        {
            continue;
        }

        Landmark& li = *ui_k;
        Landmark& lj = *uj_k;

        if (policy == ConflictResolution::KEEP_BOTH || map.IsJoinable(li, lj))
        {
            map.JoinLandmark(li, lj);
            stats.joined++;

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

            stats.removed += 2;

            break;
        }
    }

    // initialise structure for the newly spawned landmarks
    // if (mi.valid && !gi.structure.IsEmpty()) map.UpdateStructure(ai.u, gi[ai.f], mi.pose);
    // if (mj.valid && !gj.structure.IsEmpty()) map.UpdateStructure(aj.u, gj[aj.f], mj.pose);

    // post-motion correspondence recovery
    if (inlierInjection.scheme && stats.motion.valid)
    {
        GeometricMapping forward, backward, epipolar;

        if (inlierInjection.scheme & InlierInjectionScheme::FORWARD_FLOW)
        {
            if (pi && pj)
            {
                AlignmentObjective::Own eval(new EpipolarObjective(pi, pj));
                forward = FindLostFeaturesFlow(Ii, Ij, fi, eval, stats.motion.pose, qi);
            }
            else
            {
                E_WARNING << "projection(s) missing for optical flow feature recovery";
                E_WARNING << "forward flow-based inlier injection now deactivated.";

                inlierInjection.scheme &= ~InlierInjectionScheme::FORWARD_FLOW;
            }
        }

        if (inlierInjection.scheme & InlierInjectionScheme::BACKWARD_FLOW)
        {
            if (pi && pj)
            {
                AlignmentObjective::Own eval(new EpipolarObjective(pj, pi));
                backward = FindLostFeaturesFlow(Ij, Ii, fj, eval, stats.motion.pose.GetInverse(), qj);
            }
            else
            {
                E_WARNING << "projection(s) missing for optical flow feature recovery";
                E_WARNING << "backward flow-based inlier injection now deactivated.";

                inlierInjection.scheme &= ~InlierInjectionScheme::BACKWARD_FLOW;
            }
        }

        if (inlierInjection.scheme & InlierInjectionScheme::EPIPOLAR_SEARCH)
        {
            // do something..
        }

        if (!AugmentFeatures(forward, fi, fj, map, ti, tj, si, sj, ui, uj, tj.augmentedFeaturs[Fj.GetIndex()], stats.spawned, stats.tracked))
        {
            E_ERROR << "error augmenting feature set from forward flow";
            return false;
        }

        if (!AugmentFeatures(backward, fj, fi, map, tj, ti, sj, si, uj, ui, ti.augmentedFeaturs[Fi.GetIndex()], stats.spawned, stats.tracked))
        {
            E_ERROR << "error augmenting feature set from backward flow";
            return false;
        }

        for (size_t i = 0; i < forward.GetSize(); i++)
        {
            flow.Add(
                forward.src.mat.reshape(2).at<Point2D>(static_cast<int>(i)),
                forward.dst.mat.reshape(2).at<Point2D>(static_cast<int>(i)),
                ui[forward.indices[i]]->GetIndex()
            );
        }

        for (size_t j = 0; j < backward.GetSize(); j++)
        {
            flow.Add(
                backward.dst.mat.reshape(2).at<Point2D>(static_cast<int>(j)),
                backward.src.mat.reshape(2).at<Point2D>(static_cast<int>(j)),
                uj[backward.indices[j]]->GetIndex()
            );
        }

        stats.injected += forward.GetSize() + backward.GetSize();
    }

    stats.flow = flow.Build();

    if (structureScheme != NO_RECOVERY && mi.valid && stats.motion.valid)
    {
        boost::shared_ptr<const PosedProjection> ppi = boost::dynamic_pointer_cast<const PosedProjection, const ProjectionModel>(pi);
        boost::shared_ptr<const PosedProjection> ppj = boost::dynamic_pointer_cast<const PosedProjection, const ProjectionModel>(pj);

        assert(ppi && ppj);

        MidPointTriangulation tau(*ppi, *ppj, stats.motion.pose);
        map.UpdateStructure(map.GetLandmarks(stats.flow.indices), tau(stats.flow), mi.pose);
    }

    cv::Mat im = imfuse(Ii, Ij);
    ColourMap cmap(255);

    for (size_t i = 0; i < stats.flow.GetSize(); i++)
    {
        static const double zmin = 0;
        static const double zmax = 100;
        const size_t k = stats.flow.indices[i];

        Point3D gk = map.GetLandmark(k).position;

        double zk = mi.pose(gk).z;
        Point2D xk = stats.flow.src.mat.reshape(2).at<Point2D>(static_cast<int>(i));
        Point2D yk = stats.flow.dst.mat.reshape(2).at<Point2D>(static_cast<int>(i));

        cv::line(im, xk, yk, cmap.GetColour(zk, zmin, zmax), 2);
    }

    //std::stringstream ss;
    //ss << "frame-" << ti.GetIndex() << ".png";
    //cv::hconcat(im, fmap.Draw(Ii, Ij), im);
    //cv::imwrite(ss.str(), im);
    cv::imshow(ToString(), im);
    cv::waitKey(1);

    std::stringstream ss; ss << "frame" << ti.GetIndex() << ".jpg";
    cv::imwrite(ss.str(), im);

    return true;
}

StructureEstimation::Estimate FeatureTracker::GetFeatureStructure(const ImageFeatureSet& f, const StructureEstimation::Estimate& structure)
{
    return structure[GetFeatureImageIndices(f, structure.structure.mat.size())];
}

Indices FeatureTracker::GetFeatureImageIndices(const ImageFeatureSet& f, const cv::Size& imageSize) const
{
    Indices indices;

    // convert subscripts to rounded integer indices
    for (size_t k = 0; k < f.GetSize(); k++)
    {
        const Point2F& sub = f[k].keypoint.pt;
        const int i = static_cast<int>(std::round(sub.y));
        const int j = static_cast<int>(std::round(sub.x));

        if (i < 0 || i >= imageSize.height || j < 0 || j >= imageSize.width)
        {
            indices.push_back(INVALID_INDEX);
            E_ERROR << "subscript (" << i << "," << j << ") out of bound";

            continue;
        }

        indices.push_back(static_cast<size_t>(i * imageSize.width + j));
    }

    return indices;
}

GeometricMapping FeatureTracker::FindLostFeaturesFlow(const cv::Mat& Ii, const cv::Mat& Ij, const ImageFeatureSet& fi, AlignmentObjective::Own eval, const EuclideanTransform& pose, std::vector<bool>& tracked)
{
    assert(fi.GetSize() == tracked.size());
    assert(eval);

    // cv::Mat im = imfuse(Ii, Ij);

    struct Flow
    {
        Points2F xi;
        Points2F xj;
        std::vector<size_t> idx;
        std::vector<uchar> found;
        cv::Mat err;
    };

    Flow forward;

    for (size_t k = 0; k < tracked.size(); k++)
    {
        if (tracked[k]) continue;

        // cv::circle(im, fi[k].keypoint.pt, 2, cv::Scalar(0, 0, 255), -1);

        forward.xi.push_back(fi[k].keypoint.pt);
        forward.idx.push_back(k);
    }

    const int bs = static_cast<int>(inlierInjection.blockSize);
    cv::Size win(bs, bs);
    cv::calcOpticalFlowPyrLK(Ii, Ij, forward.xi, forward.xj, forward.found, forward.err, win, static_cast<int>(inlierInjection.levels));

    for (size_t i = 0; i < forward.found.size(); i++)
    {
        const double x = round(forward.xj[i].x);
        const double y = round(forward.xj[i].y);

        if (x < 0 || x >= Ij.cols || y < 0 || y >= Ij.rows)
        {
            forward.found[i] = false;
        }
    }

    if (inlierInjection.bidirectionalEps > 0)
    {
        Flow backward;

        for (size_t i = 0; i < forward.found.size(); i++)
        {
            if (!forward.found[i]) continue;

            backward.xj.push_back(forward.xj[i]);
            backward.idx.push_back(i);

            // cv::line(im, forward.xi[i], forward.xj[i], cv::Scalar(127, 127, 127));
            // cv::circle(im, forward.xj[i], 2, cv::Scalar(255, 0, 0), -1);
        }

        cv::calcOpticalFlowPyrLK(Ij, Ii, backward.xj, backward.xi, backward.found, backward.err, win, static_cast<int>(inlierInjection.levels));
        const double eps2 = inlierInjection.bidirectionalEps * inlierInjection.bidirectionalEps;

        for (size_t j = 0; j < backward.found.size(); j++)
        {
            const size_t i = backward.idx[j];

            if (!backward.found[j])
            {
                forward.found[i] = false;
                continue;
            }

            // cv::circle(im, backward.xi[j], 2, cv::Scalar(0, 127, 255), -1);

            const double dx = forward.xi[i].x - backward.xi[j].x;
            const double dy = forward.xi[i].y - backward.xi[j].y;
            
            forward.found[i] = (dx * dx + dy * dy) < eps2;

            // cv::line(im, forward.xj[i], backward.xi[j], forward.found[i] ? cv::Scalar(192, 64, 64) : cv::Scalar(64, 64, 192));
        }
    }

    GeometricMapping::ImageToImageBuilder builder;
    GeometricMapping mapping;
    Indices inliers;
    // std::vector<size_t> trace;

    for (size_t i = 0; i < forward.found.size(); i++)
    {
        if (!forward.found[i]) continue;

        builder.Add(forward.xi[i], forward.xj[i], forward.idx[i]);

        // trace.push_back(i);
        // cv::line(im, forward.xi[i], forward.xj[i], cv::Scalar(255, 255, 255));
    }

    mapping = builder.Build();
    
    if (!eval->SetData(mapping))
    {
        E_ERROR << "error setting geometric mapping for epipolar alignment verification";
        return GeometricMapping();
    }
   
    if (!eval->GetSelector(1.0f / inlierInjection.epipolarEps)(pose, inliers))
    {
        E_ERROR << "error evaluating epipolar error of the computed flow";
        return GeometricMapping();
    }

    BOOST_FOREACH (size_t idx, inliers)
    {
        tracked[mapping.indices[idx]] = true;
        // cv::line(im, forward.xi[trace[idx]], forward.xj[trace[idx]], cv::Scalar(0, 255, 0));
    }

    // cv::imshow("Lost Feature Flow", im);
    // cv::waitKey(0);

    return mapping[inliers];
}

bool FeatureTracker::AugmentFeatures(
    const GeometricMapping flow, const ImageFeatureSet& fi, const ImageFeatureSet& fj,
    Map& map, Frame& ti, Frame& tj, Source& si, Source& sj,
    Landmark::Ptrs& ui, Landmark::Ptrs& uj, ImageFeatureSet& aj, size_t& spawned, size_t& tracked)
{
    if (flow.GetSize() == 0)
    {
        return true;
    }

    const size_t n = uj.size();
    KeyPoints keypoints;
    cv::Mat descriptors;

    const cv::Mat xj = flow.dst.mat.reshape(2);

    uj.resize(n + flow.GetSize(), NULL);

    for (size_t k = 0; k < flow.GetSize(); k++)
    {
        const size_t i = flow.indices[k];
        const size_t j = n + k;

        Landmark*& ui_k = ui[i];
        Landmark*& uj_k = uj[j];
        cv::KeyPoint kp = fi[i].keypoint;

        if (ui_k == NULL)
        {
            (ui_k = &map.AddLandmark())->Hit(ti, si, i).proj = kp.pt;
            spawned++;
        }
        else
        {
            tracked++;
        }

        kp.pt = xj.at<Point2D>(static_cast<int>(k));
        (uj_k = ui_k)->Hit(tj, sj, j).proj = kp.pt;

        keypoints.push_back(kp);
    }

    FeatureExtractor::ConstOwn xtor = boost::dynamic_pointer_cast<const FeatureExtractor, const FeatureDetextractor>(si.store->GetFeatureDetextractor());
    bool copyDesc = true;

    if (inlierInjection.extractDescriptor && xtor)
    {
        // ImageFeatureSet aj = xtor->ExtractFeatures(Ij, keypoints);
        //
        // if (aj.GetSize() == forward.GetSize())
        // {
        //    descriptors = aj.GetDescriptors().clone();
        //    copyDesc = false;
        // }
        // else
        // {
        //   E_ERROR << "descriptor extraction returns " << aj.GetSize() << " element(s), while given " << keypoints.size();
        // }
    }

    if (copyDesc)
    {
        const cv::Mat src = fi.GetDescriptors();
        descriptors = cv::Mat(static_cast<int>(flow.GetSize()), src.cols, src.type());

        for (size_t k = 0; k < flow.GetSize(); k++)
        {
            src.row(static_cast<int>(flow.indices[k])).copyTo(descriptors.row(static_cast<int>(k)));
        }
    }

    if (!aj.Append(ImageFeatureSet(keypoints, descriptors, fj.GetNormType())))
    {
        E_FATAL << "error augmenting feature set for frame " << tj.GetIndex() << " store " << sj.store->GetIndex();
        return false;
    }

    return true;
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

void FeatureTracker::EpipolarObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    m_builder.Add(fi.keypoint.pt, fj.keypoint.pt, localIdx);
}

bool FeatureTracker::EpipolarObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new EpipolarObjective(pi, pj));
    
    data = m_builder.Build();
    data.metric = Metric::Own(new EuclideanMetric(epsilon * 0.5f));
    
    if (!objective->SetData(data))
    {
        E_WARNING << "error setting epipolar constraints for \"" << ToString() << "\"";
        return false;
    }

    selector = objective->GetSelector(1.0f);
    return true;
}

//==[ FeatureTracker::ProjectiveOutlierModel ]===============================//

void FeatureTracker::PerspectiveObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    if (g.structure.IsEmpty()) return;

    const Point3D& gk = g.structure.mat.reshape(3).at<Point3D>(static_cast<int>(forward ? i : j));
    const Point2F& pk = forward ? fj.keypoint.pt : fi.keypoint.pt;

    if (gk.z > 0)
    {
        m_builder.Add(gk, pk, localIdx);
        m_idx.push_back(forward ? i : j);
    }
}

bool FeatureTracker::PerspectiveObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new ProjectionObjective(p, forward));
    //boost::shared_ptr<const MahalanobisMetric> metric =
    //    boost::dynamic_pointer_cast<const MahalanobisMetric, const Metric>(g.metric ? (*g.metric)[m_idx] : Metric::Own());

    data = m_builder.Build();

    //if (metric)
    //{
    //    data.metric = (*metric).Reduce();
    //}
    //else
    //{
    //    data.metric = Metric::Own(new EuclideanMetric(1));
    //}

    data.metric = g.metric ? (*g.metric)[m_idx] : Metric::Own();

    // WeightedEuclideanMetric* me = dynamic_cast<WeightedEuclideanMetric*>(data.metric.get());
    // if (me) me->scale = 0.05f;

    if (!objective->SetData(data))
    {
        E_WARNING << "error setting perspective constraints for \"" << ToString() << "\"";
        return false;
    }

    selector = objective->GetSelector(sigma);
    return true;
}

PoseEstimator::Own FeatureTracker::PerspectiveObjectiveBuilder::GetSolver() const
{
    PoseEstimator::Own estimator = PoseEstimator::Own(new PerspevtivePoseEstimator(p));
    return forward ? estimator : PoseEstimator::Own(new InversePoseEstimator(estimator));
}

//==[ FeatureTracker::PhotometricObjectiveBuilder ]=========================//

void FeatureTracker::PhotometricObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    const Point3D& gi_k = gi.structure.mat.reshape(3).at<Point3D>(static_cast<int>(i));

    if (gi_k.z > 0)
    {
        m_idx.push_back(i);
        m_localIdx.push_back(localIdx);
    }
}

bool FeatureTracker::PhotometricObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    boost::shared_ptr<PhotometricObjective> objective =
        boost::shared_ptr<PhotometricObjective>(new PhotometricObjective(pj, Ij));

    StructureEstimation::Estimate g = gi[m_idx];

    if (!objective->SetData(g.structure, Ii, g.metric, m_localIdx))
    {
        E_WARNING << "error setting photometric constraints for \"" << ToString() << "\"";
        return false;
    }

    data = objective->GetData();
    selector = objective->GetSelector(sigma);

    return true;
}

//==[ FeatureTracker::RigidObjectiveBuilder ]===============================//

void FeatureTracker::RigidObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    if (gi.structure.IsEmpty() || gj.structure.IsEmpty()) return;

    const Point3D& gi_k = gi.structure.mat.reshape(3).at<Point3D>(static_cast<int>(i));
    const Point3D& gj_k = gj.structure.mat.reshape(3).at<Point3D>(static_cast<int>(j));

    if (gi_k.z > 0 && gj_k.z > 0)
    {
        m_builder.Add(gi_k, gj_k, localIdx);
        m_idx0.push_back(i);
        m_idx1.push_back(j);
    }
}

bool FeatureTracker::RigidObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new RigidObjective());

    data = m_builder.Build();
    data.metric = gi.metric && gj.metric ?  *(*gi.metric)[m_idx0] + *(*gj.metric)[m_idx1] : Metric::Own();
    //data.metric = gi.metric && gj.metric ?  Metric::Own(new DualMetric((*gi.metric)[m_idx0], (*gj.metric)[m_idx1])) : Metric::Own();

    if (!objective->SetData(data))
    {
        E_WARNING << "error setting rigid constraints for \"" << ToString() << "\"";
        return false;
    }

    selector = objective->GetSelector(0.75f);

    return true;
}

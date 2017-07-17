#include <seq2map/geometry_problems.hpp>
#include <random>

using namespace seq2map;

//==[ AlignmentObjective ]====================================================//

//==[ AlignmentObjective::InlierSelector ]====================================//

bool AlignmentObjective::InlierSelector::operator() (const EuclideanTransform& x, Indices& inliers, Indices& outliers) const
{
    inliers.clear();
    outliers.clear();

    try
    {
        metre.Start();
        cv::Mat error = (*objective)(x);
        metre.Stop(error.rows);

        if (error.type() != CV_64F)
        {
            error.convertTo(error, CV_64F);
        }

        const double* e = error.ptr<double>();

        for (size_t i = 0; i < error.total(); i++)
        {
            if (e[i] > threshold)
            {
                outliers.push_back(i);
            }
            else
            {
                inliers.push_back(i);
            }
        }
    }
    catch (std::exception& ex)
    {
        E_ERROR << "error selecting inliers : " << ex.what();
        return false;
    }

    return true;
}

bool AlignmentObjective::InlierSelector::operator() (const EuclideanTransform& x, Indices& inliers) const
{
    Indices outliers;
    return (*this)(x, inliers, outliers);
}

//==[ EpipolarObjective ]=====================================================//

bool EpipolarObjective::SetData(const GeometricMapping& data, ProjectionModel::ConstOwn& src, ProjectionModel::ConstOwn& dst)
{
    if (!data.IsConsistent()) return false;

    const size_t d0 = data.src.GetDimension();
    const size_t d1 = data.dst.GetDimension();

    if (d0 != 2 && d0 != 3)
    {
        E_ERROR << "source geometry has to be either 2D Euclidean or 3D homogeneous (d=" << d0 << ")";
        return false;
    }

    if (d1 != 2 && d1 != 3)
    {
        E_ERROR << "target geometry has to be either 2D Euclidean or 3D homogeneous (d=" << d1 << ")";
        return false;
    }

    if (!src)
    {
        E_ERROR << "missing source projection model";
        return false;
    }

    if (!dst)
    {
        E_ERROR << "missing target projection model";
        return false;
    }

    m_data.indices = data.indices;
    m_data.src = src->Backproject(data.src); //.Reshape(Geometry::ROW_MAJOR);
    m_data.dst = dst->Backproject(data.dst); //.Reshape(Geometry::ROW_MAJOR);
    m_data.metric = data.metric ? data.metric : Metric::Own(new EuclideanMetric());

    return true;
}

AlignmentObjective::Own EpipolarObjective::GetSubObjective(const Indices& indices) const
{
    EpipolarObjective* sub = new EpipolarObjective(m_src, m_dst, m_distType);
    sub->m_data = m_data[indices]; // avoid calling SetData() as the normalised data should not be normalised again

    return AlignmentObjective::Own(sub);
}

cv::Mat EpipolarObjective::operator() (const EuclideanTransform& tform) const
{
    const Metric& d = *m_data.metric;

    cv::Mat x0 = m_data.src.mat;
    cv::Mat x1 = m_data.dst.mat;

    cv::Mat F = tform.ToEssentialMatrix();

    cv::Mat Fx0 = x0 * F.t();
    cv::Mat Fx00 = Fx0.col(0);
    cv::Mat Fx01 = Fx0.col(1);
    cv::Mat Fx02 = Fx0.col(2);

    cv::Mat xFx = Fx00.mul(x1.col(0)) + Fx01.mul(x1.col(1)) + Fx02; 
    //      xFx = Fx00.mul(x1.col(0)) + Fx01.mul(x1.col(1)) + Fx02.mul(x1.col(2));
    //                                                   should be one ^^^^^^^^^

    if (m_distType == ALGEBRAIC)
    {
        // algebraic epipolar distance
        // x1' * F * x0
        Geometry err(m_data.src.shape, xFx);
        return d(err).mat;
    }

    cv::Mat Fx1 = x1 * F;
    cv::Mat Fx10 = Fx1.col(0);
    cv::Mat Fx11 = Fx1.col(1);
    cv::Mat Fx12 = Fx1.col(2);

    cv::Mat nn0 = Fx00.mul(Fx00) + Fx01.mul(Fx01);
    cv::Mat nn1 = Fx10.mul(Fx10) + Fx11.mul(Fx11);

    Geometry err(m_data.src.shape);

    if (m_distType == SAMPSON)
    {
        // Sampson approximation:
        //
        //              x1' * F * x0
        // ---------------------------------------
        // sqrt(Fx00^2 + Fx01^2 + Fx10^2 + Fx11^2)

        cv::Mat nn; cv::sqrt(nn0 + nn1, nn);
        cv::divide(xFx, nn, err.mat);
    }
    else
    {
        // symmetric geometric error
        //
        //        x1' * F * x0              x1' * F * x0
        // ( ---------------------- , ---------------------- )
        //    sqrt(Fx00^2 + Fx01^2)    sqrt(Fx10^2 + Fx11^2)

        err.mat = cv::Mat(xFx.rows, 2, xFx.type());
        cv::sqrt(nn0, nn0);
        cv::sqrt(nn1, nn1);

        cv::divide(xFx, nn0, err.mat.col(0));
        cv::divide(xFx, nn1, err.mat.col(1));
    }

    return d(err).mat;
}

//==[ ProjectionObjective ]===================================================//

AlignmentObjective::Own ProjectionObjective::GetSubObjective(const Indices& indices) const
{
    ProjectionObjective* sub = new ProjectionObjective(m_proj, m_forward);
    sub->m_data = m_data[indices];

    return AlignmentObjective::Own(sub);
}

bool ProjectionObjective::SetData(const GeometricMapping& data)
{
    if (!data.IsConsistent()) return false;

    const size_t d0 = data.src.GetDimension();
    const size_t d1 = data.dst.GetDimension();

    if (d0 != 3 && d0 != 4)
    {
        E_ERROR << "source geometry has to be either 3D Euclidean or 4D homogeneous (d=" << d0 << ")";
        return false;
    }

    if (d1 != 2 && d1 != 3)
    {
        E_ERROR << "target geometry has to be either 2D Euclidean or 3D homogeneous (d=" << d1 << ")";
        return false;
    }

    m_data.indices = data.indices;
    m_data.src = (d0 == 3 ? Geometry::MakeHomogeneous(data.src) : data.src); //.Reshape(Geometry::ROW_MAJOR);
    m_data.dst = (d1 == 3 ? Geometry::FromHomogeneous(data.dst) : data.dst); //.Reshape(Geometry::ROW_MAJOR);
    m_data.metric = data.metric ? data.metric : Metric::Own(new EuclideanMetric());

    return true;
}

cv::Mat ProjectionObjective::operator() (const EuclideanTransform& f) const
{
    if (!m_proj)
    {
        E_ERROR << "missing projection model";
        return cv::Mat();
    }

    const EuclideanTransform tf = m_forward ? f : f.GetInverse();
    const Metric::Own d = m_data.metric->Transform(tf);

    Geometry x = m_data.src;
    Geometry y = m_proj->Project(tf(x, true), ProjectionModel::EUCLIDEAN_2D);

    return (*d)(y, m_data.dst).mat;

    /*
    cv::Mat rpe = y.mat - m_data.dst.mat;

    if (m_separatedRpe)
    {
        return rpe.reshape(1, rpe.total());
    }

    cv::multiply(rpe, rpe, rpe);
    cv::sqrt(cv::Mat(rpe.col(0) + rpe.col(1)), rpe);

    //rpe.convertTo(rpe, CV_64F);

    return rpe;
    */
}

//==[ PhotometricObjective ]==================================================//

AlignmentObjective::Own PhotometricObjective::GetSubObjective(const Indices& indices) const
{
    PhotometricObjective* sub = new PhotometricObjective(m_proj, m_dst, m_type, m_interp);
    sub->m_data = m_data[indices];

    return AlignmentObjective::Own(sub);
}

bool PhotometricObjective::SetData(const GeometricMapping& data)
{
    if (!data.IsConsistent()) return false;

    const size_t d0 = data.src.GetDimension();
    const size_t d1 = data.dst.GetDimension();
    const size_t dI = static_cast<size_t>(m_dst.channels());

    if (d0 != 3 && d0 != 4)
    {
        E_ERROR << "source geometry has to be either 3D Euclidean or 4D homogeneous";
        return false;
    }

    if (d1 != dI)
    {
        E_ERROR << "dimensionality mismatches between the referenced image (d=" << dI << ") and the target (d=" << d1 << ")";
        return false;
    }

    m_data.indices = data.indices;
    m_data.src = (d0 == 3 ? Geometry::MakeHomogeneous(data.src) : data.src);
    m_data.dst = data.dst;
    m_data.metric = data.metric ? data.metric : Metric::Own(new EuclideanMetric());

    if (m_data.dst.mat.depth() != m_type)
    {
        m_data.dst.mat.convertTo(m_data.dst.mat, m_type);
    }

    return true;
}

bool PhotometricObjective::SetData(const Geometry& g, const cv::Mat& src, const Metric::Own metric, const std::vector<size_t>& indices)
{
    GeometricMapping data;

    bool dense = g.shape == Geometry::PACKED && g.mat.rows == src.rows && g.mat.cols == src.cols;

    cv::Mat src32F = src;

    if (src.depth() != m_type)
    {
        src.convertTo(src32F, m_type);
    }

    if (dense)
    {
        Points3D xyz;
        std::vector<cv::Point_<short>> sub;

        for (int i = 0; i < g.mat.rows; i++)
        {
            for (int j = 0; j < g.mat.cols; j++)
            {
                Point3D pt = g.mat.at<Point3D>(i, j);
                if (pt.z > 0)
                {
                    xyz.push_back(pt);
                    sub.push_back(cv::Point_<short>(j, i));
                }
            }
        }

        data.src = Geometry(Geometry::PACKED, cv::Mat(xyz));
        data.dst = Geometry(Geometry::PACKED, interp(src32F, cv::Mat(sub, false), cv::INTER_NEAREST));
    }
    else
    {
        if (!m_proj)
        {
            E_ERROR << "missing projection model";
            return false;
        }

        cv::Mat proj;
        m_proj->Project(g, ProjectionModel::EUCLIDEAN_2D).Reshape(Geometry::PACKED).mat.convertTo(proj, CV_32F);

        data.src = g;
        data.dst = Geometry(Geometry::PACKED, interp(src32F, proj, m_interp));
    }

    data.metric = metric;
    data.indices = indices;

    return SetData(data);
}

cv::Mat PhotometricObjective::operator() (const EuclideanTransform& f) const
{
    if (!m_proj)
    {
        E_ERROR << "missing projection model";
        return cv::Mat();
    }

    Metric::ConstOwn metric = m_data.metric->Transform(f);
    Geometry x = m_data.src;

    cv::Mat proj;
    m_proj->Project(f(x, true), ProjectionModel::EUCLIDEAN_2D).Reshape(Geometry::PACKED).mat.convertTo(proj, CV_32F);

    Geometry y = Geometry(Geometry::PACKED, interp(m_dst, proj, m_interp));

    //if (y.mat.depth() != m_data.dst.mat.depth())
    //{
    //    y.mat.convertTo(y.mat, m_data.dst.mat.depth());
    //}

    return (*metric)(m_data.dst, y).mat;
}

//==[ RigidObjective ]========================================================//

AlignmentObjective::Own RigidObjective::GetSubObjective(const Indices& indices) const
{
    RigidObjective* sub = new RigidObjective();
    sub->m_data = m_data[indices];

    return AlignmentObjective::Own(sub);
}

bool RigidObjective::SetData(const GeometricMapping& data)
{
    if (!data.IsConsistent()) return false;

    const size_t d0 = data.src.GetDimension();
    const size_t d1 = data.dst.GetDimension();

    if (d0 != 3 && d0 != 4)
    {
        E_ERROR << "source geometry has to be either 3D Euclidean or 4D homogeneous (d=" << d0 << ")";
        return false;
    }

    if (d1 != 3 && d1 != 4)
    {
        E_ERROR << "target geometry has to be either 3D Euclidean or 4D homogeneous (d=" << d1 << ")";
        return false;
    }

    m_data.indices = data.indices;
    m_data.src = (d0 == 3 ? Geometry::MakeHomogeneous(data.src) : data.src);
    m_data.dst = (d1 == 4 ? Geometry::FromHomogeneous(data.dst) : data.dst);
    m_data.metric = data.metric;

    return true;
}

cv::Mat RigidObjective::operator() (const EuclideanTransform& f) const
{
    Metric::ConstOwn metric = m_data.metric->Transform(f);

    Geometry x = m_data.src;
    Geometry y = f(x, true);

    return (*metric)(m_data.dst, y).mat;
}

//==[ PoseEstimator ]=========================================================//

//==[ EssentialMatrixDecomposer ]=============================================//

bool EssentialMatrixDecomposer::operator() (const GeometricMapping& mapping, Estimate& estimate) const
{
    if (!mapping.Check(2, 2))
    {
        E_ERROR << "invalid mapping";
        return false;
    }

    if (mapping.GetSize() < GetMinPoints())
    {
        E_ERROR << "insufficient mapping size (" << mapping.GetSize() << " < " << GetMinPoints() << ")";
        return false;
    }

    if (!m_srcProj || !m_dstProj)
    {
        E_ERROR << "missing projeciton model(s)";
        return false;
    }

    GeometricMapping bpm;
    bpm.src.mat = Geometry::FromHomogeneous(m_srcProj->Backproject(mapping.src)).mat;
    bpm.dst.mat = Geometry::FromHomogeneous(m_dstProj->Backproject(mapping.dst)).mat;

    cv::Mat E;

    try
    {
        E = cv::findEssentialMat(bpm.src.mat, bpm.dst.mat, cv::Mat::eye(3, 3, CV_64F));
    }
    catch (std::exception& ex)
    {
        E_ERROR << "cv::findEssentialMat failed : " << ex.what();
        return false;
    }

    return estimate.pose.FromEssentialMatrix(E, bpm);
}

//==[ PerspevtivePoseEstimator ]==============================================//

bool PerspevtivePoseEstimator::operator() (const GeometricMapping& mapping, Estimate& estimate) const
{
    if (!mapping.Check(3, 2))
    {
        E_ERROR << "invalid mapping";
        return false;
    }

    if (mapping.GetSize() < GetMinPoints())
    {
        E_ERROR << "insufficient mapping size (" << mapping.GetSize() << " < " << GetMinPoints() << ")";
        return false;
    }

    if (!m_proj)
    {
        E_ERROR << "missing projection model";
        return false;
    }

    const cv::Mat opts = mapping.src.mat.reshape(3);
    const cv::Mat ipts = Geometry::FromHomogeneous(m_proj->Backproject(mapping.dst)).mat.reshape(2);

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D = cv::Mat();
    cv::Mat rvec, tvec;

    bool success = cv::solvePnP(opts, ipts, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP);

    if (!success)
    {
        E_ERROR << "cv::solvePnP failed";
        return false;
    }

    if (!estimate.pose.GetRotation().FromVector(rvec))
    {
        E_ERROR << "error setting rotation vector";
        return false;
    }

    if (!estimate.pose.SetTranslation(tvec))
    {
        E_ERROR << "error setting translation vector";
        return false;
    }

    return true;
}

//==[ QuatAbsOrientationSolver ]==============================================//

bool QuatAbsOrientationSolver::operator() (const GeometricMapping& mapping, Estimate& estimate) const
{
    throw std::logic_error("not implemented");
    return false;
}

//==[ DummyPoseEstimator ]====================================================//

bool DummyPoseEstimator::operator() (const GeometricMapping& mapping, Estimate& estimate) const
{
    estimate.pose = m_pose;
    return true;
}

//==[ InversePoseEstimator ]=================================================//

bool InversePoseEstimator::operator() (const GeometricMapping& mapping, Estimate& estimate) const
{
    if (!m_estimator || !(*m_estimator)(mapping, estimate))
    {
        return false;
    }

    // invert the solution
    estimate.pose = estimate.pose.GetInverse();

    // TODO: need to invert the metric as well
    // ...
    // ..
    // .

    return true;
}

//==[ ConsensusPoseEstimator ]================================================//

bool ConsensusPoseEstimator::operator() (const GeometricMapping& mapping, Estimate& estimate) const
{
    IndexLists inliers, outliers;
    return (*this)(mapping, estimate, inliers, outliers);
}

bool ConsensusPoseEstimator::operator() (const GeometricMapping& mapping, Estimate& estimate, IndexLists& inliers, IndexLists& outliers) const
{
    if (!m_solver)
    {
        E_ERROR << "missing inner pose estimator";
        return false;
    }

    if (m_selectors.empty())
    {
        E_ERROR << "missing selector";
        return false;
    }

    const PoseEstimator& f = *m_solver;
    const size_t m = mapping.GetSize();
    const size_t n = f.GetMinPoints();

    if (m < n)
    {
        E_ERROR << "insufficient number of correspondences";
        return false;
    }

    const size_t population = GetPopulation();
    const size_t minInliers = static_cast<size_t>(population * m_minInlierRatio);
    const double logOneMinusConfidence = std::log(1 - m_confidence);
    size_t numInliers = 0;

    Speedometre solveMetre;
    Speedometre evalMetre;

    if (m_verbose)
    {
        E_INFO << std::setw(80) << std::setfill('=') << "";
        E_INFO << std::setw(6)  << std::right << "Trial"
               << std::setw(19) << std::right << "Inliers"
               << std::setw(19) << std::right << "Best"
               << std::setw(18) << std::right << "Solve Time"
               << std::setw(18) << std::right << "Eval. Time";
        E_INFO << std::setw(80) << std::setfill('=') << "";
    }

    struct Result
    {
        IndexLists inliers;
        IndexLists outliers;
    };

    for (size_t k = 0, iter = m_maxIter; k < iter; k++)
    {
        Indices idx = DrawSamples(m, n);
        GeometricMapping samples = mapping[idx];
        PoseEstimator::Estimate trial;
        Result result;
        size_t hits = 0;

        solveMetre.Start();
        if (!f(samples, trial))
        {
            E_ERROR << "inner pose estimator failed";
            return false;
        }
        solveMetre.Stop(1);

        evalMetre.Start();
        BOOST_FOREACH (const AlignmentObjective::InlierSelector& g, m_selectors)
        {
            Indices accepted, rejected;
            if (!g(trial.pose, accepted, rejected))
            {
                E_ERROR << "selector failed";
                return false;
            }

            result.inliers.push_back(accepted);
            result.outliers.push_back(rejected);

            hits += accepted.size();
        }
        evalMetre.Stop(1);

        if (m_verbose)
        {
            E_INFO << std::setw(6)  << std::right << (k + 1)
                   << std::setw(12) << std::right << hits       << " (" << std::setw(3) << std::right << (100*hits/population)       << "%)"
                   << std::setw(12) << std::right << numInliers << " (" << std::setw(3) << std::right << (100*numInliers/population) << "%)"
                   << std::setw(15) << std::right << (solveMetre.GetElapsedSeconds() * 1000) << " ms"
                   << std::setw(15) << std::right << (evalMetre .GetElapsedSeconds() * 1000) << " ms";
        }

        if (hits < numInliers) continue;

        // accept the trial
        numInliers = hits;
        estimate = trial;
        inliers  = result.inliers;
        outliers = result.outliers;

        if (hits < minInliers) continue;

        // convergence control
        iter = std::min(iter, static_cast<size_t>(std::ceil(logOneMinusConfidence / std::log(1 - std::pow(hits / population, n)))));
    }

    const bool success = numInliers >= minInliers;

    if (!success)
    {
        const int inlierRate = (int) (std::round(100 * numInliers / population));
        const int targetRate = (int) (std::round(100 * minInliers / population));

        E_WARNING << "unable to find enough inliers in " << m_maxIter << " iteration(s)";
        E_WARNING << "the best trial achieves " << inlierRate << "% inliers while " << targetRate << "% is required";
    }

    // post-estimation non-linear optimisation
    if (success && m_optimisation)
    {
        MultiObjectivePoseEstimation refinement;
        refinement.SetPose(estimate.pose);

        size_t i = 0;
        BOOST_FOREACH (const AlignmentObjective::InlierSelector& g, m_selectors)
        {
            AlignmentObjective::Own sub = g.objective->GetSubObjective(inliers[i++]);

            if (!sub)
            {
                E_WARNING << "error building sub-objective for model " << (i-1);
                continue;
            }

            refinement.AddObjective(AlignmentObjective::ConstOwn(sub));
        }

        LevenbergMarquardtAlgorithm levmar;
        levmar.SetVervbose(m_verbose);

        if (!levmar.Solve(refinement))
        {
            E_ERROR << "non-linear optimisation failed";
            return false;
        }

        //////////////////////////////////////////////////////////////
        // VectorisableD::Vec x; refinement.GetPose().Store(x);
        // PersistentMat(cv::Mat(refinement(x))).Store(Path("y.bin"));
        //////////////////////////////////////////////////////////////

        estimate.pose = refinement.GetPose();
        inliers.clear();
        outliers.clear();
        numInliers = 0;

        BOOST_FOREACH (const AlignmentObjective::InlierSelector& g, m_selectors)
        {
            Indices accepted, rejected;

            g(estimate.pose, accepted, rejected);

            inliers.push_back(accepted);
            outliers.push_back(rejected);

            numInliers += accepted.size();
        }

        if (numInliers < minInliers)
        {
            E_WARNING << "the optimised model results too few inliers (" << numInliers << " < " << minInliers << ")";
        }
    }

    return estimate.valid = numInliers >= minInliers;
}

size_t ConsensusPoseEstimator::GetPopulation() const
{
    size_t n = 0;

    BOOST_FOREACH (const AlignmentObjective::InlierSelector& selector, m_selectors)
    {
        n += selector.objective ? selector.objective->GetData().GetSize() : 0;
    }

    return n;
}

Indices ConsensusPoseEstimator::DrawSamples(size_t population, size_t samples)
{
    if (population < samples)
    {
        E_ERROR << "insufficient population (n=" << population << ") for " << samples << " sample(s)";
        return Indices();
    }

    if (samples == 0)
    {
        return Indices();
    }

    std::vector<size_t> idx(population);
    for (size_t i = 0; i < population; i++)
    {
        idx[i] = i;
    }

    std::random_shuffle(idx.begin(), idx.end());

    return Indices(idx.begin(), std::next(idx.begin(), samples));
}

//==[ MultiObjectivePoseEstimation ]==========================================//

bool MultiObjectivePoseEstimation::Initialise(VectorisableD::Vec& x)
{
    m_conds = GetConds();
    return m_conds >= 6 && m_tform.Store(x);
}

VectorisableD::Vec MultiObjectivePoseEstimation::operator() (const VectorisableD::Vec& x) const
{
    EuclideanTransform tform(m_tform.GetRotation().GetParameterisation());

    if (!tform.Restore(x))
    {
        E_ERROR << "error devectorising transform";
        return VectorisableD::Vec();
    }

    VectorisableD::Vec y;
    size_t m = 0;

    y.reserve(m_conds);

    BOOST_FOREACH (const AlignmentObjective::ConstOwn& obj, m_objectives)
    {
        if (!obj) continue;

        cv::Mat yi = (*obj)(tform);
        size_t  mi = static_cast<size_t>(yi.total());

        assert(m + mi <= m_conds);

        if (yi.type() != CV_64F)
        {
            yi.convertTo(yi, CV_64F);
        }

        y.insert(y.end(), (double*)yi.datastart, (double*)yi.dataend);
        m += mi;
    }

    assert(m == m_conds);

    return y;
}

size_t MultiObjectivePoseEstimation::GetConds() const
{
    size_t m = 0;

    BOOST_FOREACH (const AlignmentObjective::ConstOwn& obj, m_objectives)
    {
        if (!obj) continue;
        m += obj->GetData().dst.GetElements();
    }

    return m;
}

//==[ StructureEstimation ]===================================================//

//==[ StructureEstimation::Estimate ]=========================================//

StructureEstimation::Estimate StructureEstimation::Estimate::operator[] (const Indices& indices) const
{
    Estimate estimate(structure.shape);
    estimate.structure = structure[indices];
    estimate.metric    = metric ? (*metric)[indices] : Metric::Own();

    return estimate;
}

StructureEstimation::Estimate StructureEstimation::Estimate::operator+ (const Estimate& estimate) const
{
    Estimate clone(Geometry(structure), metric ? metric->Clone() : Metric::Own());
    return clone += estimate;
}

StructureEstimation::Estimate& StructureEstimation::Estimate::operator+= (const Estimate& estimate)
{
    if (!metric || !estimate.metric)
    {
        E_ERROR << "fusion failed due to missing metric(es)";
        return *this;
    }

    bool native;

    boost::shared_ptr</***/ MahalanobisMetric> m0 = metric->ToMahalanobis(native);
    boost::shared_ptr<const MahalanobisMetric> m1 = estimate.metric->ToMahalanobis();

    if (!m0 || !m1)
    {
        E_ERROR << "fusion failed due to missing Mahalanobis metric(es)";
        return *this;
    }

    Geometry g0  = structure.Reshape(Geometry::ROW_MAJOR);
    Geometry g10 = estimate.structure - g0;

    if (g10.IsEmpty())
    {
        E_WARNING << "the subtraction of structure geometry is empty";
        return *this;
    }
    
    cv::Mat K = m0->Update(*m1);
    
    if (K.empty())
    {
        E_ERROR << "error updating metric, invalid Kalman gain returned";
        return *this;
    }

    switch (m0->type)
    {
    case MahalanobisMetric::ISOTROPIC:

        for (int j = 0; j < g0.mat.cols; j++)
        {
            g0.mat.col(j) += K.mul(g10.mat.col(j));
        }

        break;

    case MahalanobisMetric::ANISOTROPIC_ORTHOGONAL:

        for (int j = 0; j < g0.mat.cols; j++)
        {
            g0.mat.col(j) += K.col(j).mul(g10.mat.col(j));
        }

        break;

    case MahalanobisMetric::ANISOTROPIC_ROTATED:

        for (int j0 = 0; j0 < g0.mat.cols; j0++)
        {
            for (int j1 = 0; j1 < g10.mat.cols; j1++)
            {
                const int j = sub2symind(j0, j1, static_cast<int>(m0->dims));
                g0.mat.col(j0) += K.col(j).mul(g10.mat.col(j1));
            }
        }

        break;
    }

    // just in case : structure.shape == Geometry::COL_MAJOR
    structure = g0;

    if (!native)
    {
        boost::shared_ptr<WeightedEuclideanMetric> we =
            boost::dynamic_pointer_cast<WeightedEuclideanMetric, Metric>(metric);

        if (!we)
        {
            E_WARNING << "metric fusion impossible";
            return *this;
        }

        if (!we->FromMahalanobis(*m0))
        {
            E_ERROR << "error converting Mahalanobis distance to weighted Euclidean";
        }
    }

    return *this;
}

//==[ OptimalTriangulation ]==================================================//

StructureEstimation::Estimate OptimalTriangulation::operator() (const GeometricMapping& m) const
{
    if (m.src.IsConsistent(m.dst)) return Estimate(m.src.shape);

    Geometry p0 = Geometry::FromHomogeneous(P0.proj.Backproject(m.src.Reshape(Geometry::COL_MAJOR)));
    Geometry p1 = Geometry::FromHomogeneous(P1.proj.Backproject(m.dst.Reshape(Geometry::COL_MAJOR)));
    Geometry g(Geometry::COL_MAJOR);

    cv::Mat pts0 = p0.mat;
    cv::Mat pts1 = p1.mat;

    // type casting due to the limitation of cv::triangulatePoints
    if (pts0.type() != CV_32F) pts0.convertTo(pts0, CV_32F);
    if (pts1.type() != CV_32F) pts1.convertTo(pts1, CV_32F);

    cv::Mat pmat0 = P0.pose.GetTransformMatrix(false, true, CV_32F); // I * (R0 | t0)
    cv::Mat pmat1 = P1.pose.GetTransformMatrix(false, true, CV_32F); // I * (R1 | t1)

    cv::triangulatePoints(pmat0, pmat1, pts0, pts1, g.mat);

    // cast structure to the type of the given geometry data, if neccessary
    if (g.mat.type() != m.src.mat.type())
    {
        g.mat.convertTo(g.mat, m.src.mat.type());
    }

    return Estimate(Geometry::FromHomogeneous(g).Reshape(m.src));
}

//==[ MidPointTriangulation ]=================================================//

void MidPointTriangulation::DecomposeProjMatrix(const cv::Mat& P, cv::Mat& KRinv, cv::Mat& c)
{
    cv::Mat KR = P.rowRange(0, 3).colRange(0, 3);
    KRinv = KR.inv();
    c = -KRinv * P.rowRange(0, 3).col(3);
}

StructureEstimation::Estimate MidPointTriangulation::operator() (const GeometricMapping& m) const
{
    if (!m.src.IsConsistent(m.dst))
    {
        E_ERROR << "given mapping is inconsistent";
        return Estimate(m.src.shape);
    }

    cv::Mat c0 = P0.pose.GetInverse().GetTranslation();
    cv::Mat c1 = P1.pose.GetInverse().GetTranslation();

    cv::Mat s = c0 + c1;
    cv::Mat t = c1 - c0;
    cv::Mat x0 = P0.Backproject(m.src).Reshape(Geometry::ROW_MAJOR).mat;
    cv::Mat x1 = P1.Backproject(m.dst).Reshape(Geometry::ROW_MAJOR).mat;

    if (x1.type() != x0.type())
    {
        x1.convertTo(x1, x0.type());
    }

    cv::Mat g = cv::Mat(x0.rows, 3, x0.type());
    cv::Mat w = cv::Mat(x0.rows, 1, x0.type());

    for (int i = 0; i < g.rows; i++)
    {
        cv::Mat Bt = cv::Mat(2, 3, x0.type());
        x0.row(i).copyTo(Bt.row(0));
        x1.row(i).copyTo(Bt.row(1));

        cv::Mat At = Bt.clone();
        At.row(1) = -Bt.row(1);

        cv::Mat A = At.t();
        cv::Mat B = Bt.t();
        cv::Mat k = (At * A).inv() * At * t;
        cv::Mat x = (B * k + s) * 0.5f;
        cv::Mat d = (A * k - t);

        g.row(i) = x.t();
        w.row(i) = 1.0f / cv::norm(d);
    }
    
    return Estimate(
        Geometry(Geometry::ROW_MAJOR, g).Reshape(m.src),
        Metric::Own(new WeightedEuclideanMetric(w)));
}

//==[ MultipleViewBlockMatcher ]==============================================//

PinholeModel MultipleViewBlockMatcher::RefView::NullProjection;
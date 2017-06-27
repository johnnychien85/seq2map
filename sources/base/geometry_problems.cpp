#include <seq2map/geometry_problems.hpp>

using namespace seq2map;

//==[ AlignmentObjective ]====================================================//

//==[ AlignmentObjective::InlierSelector ]====================================//

bool AlignmentObjective::InlierSelector::operator() (const EuclideanTransform& x, Indices& inliers, Indices& outliers) const
{
    inliers.clear();
    outliers.clear();

    try
    {
        cv::Mat error = (*objective)(x);

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

bool ProjectionObjective::SetData(const GeometricMapping& data)
{
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

    const Metric& d = *m_data.metric;

    Geometry x = m_data.src;
    Geometry y = m_proj->Project(m_forward ? f(x, true) : f.GetInverse()(x, true), ProjectionModel::EUCLIDEAN_2D);

    return d(y, m_data.dst).mat;

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

bool PhotometricObjective::SetData(const Geometry& g, const cv::Mat& src)
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


    return SetData(data);
}

cv::Mat PhotometricObjective::operator() (const EuclideanTransform& f) const
{
    if (!m_proj)
    {
        E_ERROR << "missing projection model";
        return cv::Mat();
    }

    Geometry x = m_data.src;
    cv::Mat y;

    //Geometry x1 = m_data.src;
    //cv::Mat p0 = m_proj->Project(Geometry::FromHomogeneous(x), ProjectionModel::EUCLIDEAN_2D).mat;
    //cv::Mat p1 = m_proj->Project(tform(x1, true), ProjectionModel::EUCLIDEAN_2D).mat;

    //PersistentMat(tform.GetTransformMatrix()).Store(Path("E.bin"));
    //PersistentMat(p0).Store(Path("p0.bin"));
    //PersistentMat(p1).Store(Path("p1.bin"));

    //p0.convertTo(p0, CV_32F);
    //p1.convertTo(p1, CV_32F);

    //int id = std::rand(); std::stringstream ss; ss << id;
    //E_INFO << ss.str();
    //PersistentMat(m_data.dst.mat.clone()).Store(Path("dst" + ss.str() + ".bin"));
    //PersistentMat y0(interp(m_dst, p0.reshape(2), cv::INTER_LINEAR)); y0.Store(Path("y0" + ss.str() + ".bin"));
    //PersistentMat y1(interp(m_dst, p1.reshape(2), cv::INTER_LINEAR)); y1.Store(Path("y1" + ss.str() + ".bin"));

    m_proj->Project(f(x, true), ProjectionModel::EUCLIDEAN_2D).Reshape(Geometry::PACKED).mat.convertTo(y, CV_32F);

    return (*m_data.metric)(m_data.dst, Geometry(Geometry::PACKED, interp(m_dst, y, m_interp))).mat;
}

//==[ RigidObjective ]========================================================//

bool RigidObjective::SetData(const GeometricMapping& data)
{
    if (!data.IsConsistent()) return false;

    const size_t d0 = data.src.GetDimension();
    const size_t d1 = data.dst.GetDimension();

    if (d0 != 3 || d0 != 4)
    {
        E_ERROR << "source geometry has to be either 3D Euclidean or 4D homogeneous";
        return false;
    }

    if (d1 != 3 || d1 != 4)
    {
        E_ERROR << "target geometry has to be either 3D Euclidean or 4D homogeneous";
        return false;
    }

    m_data.indices = data.indices;
    m_data.src = (d0 == 3 ? Geometry::MakeHomogeneous(data.src) : data.src);
    m_data.dst = (d1 == 3 ? Geometry::MakeHomogeneous(data.dst) : data.dst);
    m_data.metric = data.metric;

    return true;
}

cv::Mat RigidObjective::operator() (const EuclideanTransform& f) const
{
    const Metric& d = *m_data.metric;

    Geometry x = m_data.src;
    Geometry y = f(x);

    return d(m_data.dst, y).mat;
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

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D = cv::Mat();
    cv::Mat rvec, tvec;

    bool success = cv::solvePnP(mapping.src.mat, m_proj->Backproject(mapping.dst).mat, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP);

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
                   << std::setw(15) << std::right << (evalMetre.GetElapsedSeconds() * 1000)  << " ms";
        }

        if (hits < minInliers || hits < numInliers) continue;

        // accept the trial
        numInliers = hits;
        estimate = trial;
        inliers  = result.inliers;
        outliers = result.outliers;

        // convergence control
        iter = std::min(iter, static_cast<size_t>(std::ceil(logOneMinusConfidence / std::log(1 - std::pow(hits / population, n)))));
    }

    // post-estimation non-linear optimisation
    if (numInliers > 0 && m_optimisation)
    {
        MultiObjectivePoseEstimation refinement;
        refinement.SetPose(estimate.pose);

        BOOST_FOREACH (const AlignmentObjective::InlierSelector& g, m_selectors)
        {
            refinement.AddObjective(AlignmentObjective::ConstOwn(g.objective));
        }

        LevenbergMarquardtAlgorithm levmar;
        levmar.SetVervbose(m_verbose);

        if (!levmar.Solve(refinement))
        {
            E_ERROR << "nonlinear optimisation failed";
            return false;
        }

        estimate.pose = refinement.GetPose();
        inliers.clear();
        outliers.clear();

        BOOST_FOREACH (const AlignmentObjective::InlierSelector& g, m_selectors)
        {
            Indices accepted, rejected;

            g(estimate.pose, accepted, rejected);

            inliers.push_back(accepted);
            outliers.push_back(rejected);

            numInliers += accepted.size();
        }
    }

    return numInliers > 0;
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
    Estimate est(m.src.shape);

    if (m.src.IsConsistent(m.dst))
    {
        E_ERROR << "given mapping is inconsistent";
        return est;
    }

    MahalanobisMetric e;

    Geometry p0 = P0.Backproject(m.src).Reshape(Geometry::ROW_MAJOR);
    Geometry p1 = P1.Backproject(m.dst).Reshape(Geometry::ROW_MAJOR);

    Geometry g(Geometry::ROW_MAJOR);

    cv::Mat c0 = P0.pose.GetInverse().GetTranslation();
    cv::Mat c1 = P1.pose.GetInverse().GetTranslation();

    //std::ofstream of("tri.m");
    //of << mat2string(cv::Mat(map.From()).reshape(1), "x0") << std::endl;
    //of << mat2string(cv::Mat(map.To())  .reshape(1), "x1") << std::endl;

    cv::Mat s = c0 + c1;
    cv::Mat t = c1 - c0;
    cv::Mat x0 = p0.mat;
    cv::Mat x1 = p1.mat;

    if (x1.type() != x0.type())
    {
        x1.convertTo(x1, x0.type());
    }

    //x0.convertTo(x0, CV_64F);
    //x1.convertTo(x1, CV_64F);
    //t.convertTo(t, CV_64F);
    //m.convertTo(m, CV_64F);

    //of << mat2string(m_projMatrix0, "P0") << std::endl;
    //of << mat2string(m_projMatrix1, "P1") << std::endl;
    //of << mat2string(x0, "x0_h") << std::endl;
    //of << mat2string(x1, "x1_h") << std::endl;

    g.mat = cv::Mat(x0.rows, 3, x0.type());
    e.icv.mat = cv::Mat(x0.rows, 1, x0.type());

    for (int i = 0; i < g.mat.rows; i++)
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

        double n = cv::norm(d);
        double n2 = n * n;

        g.mat.row(i) = x.t();
        e.icv.mat.row(i) = 1.0f / n2;

        //if (i == 0)
        //{
        //    of << mat2string(A, "A0") << std::endl;
        //    of << mat2string((At*A).inv(), "A0Ai") << std::endl;
        //    of << mat2string(k, "k0") << std::endl;
        //}
    }
    //of << mat2string(x3d, "g") << std::endl;

    return est;
}

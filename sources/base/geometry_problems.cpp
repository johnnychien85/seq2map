#include <seq2map/geometry_problems.hpp>

using namespace seq2map;

//==[ AlignmentObjective ]====================================================//

//==[ AlignmentObjective::InlierSelector ]====================================//

Indices AlignmentObjective::InlierSelector::operator() (const EuclideanTransform& x) const
{
    Indices inliers;
    cv::Mat error = (*objective)(x);

    if (error.type() != CV_64F)
    {
        error.convertTo(error, CV_64F);
    }

    const double* e = error.ptr<double>();

    for (size_t i = 0; i < error.total(); i++)
    {
        if (e[i] > threshold) continue;
        inliers.push_back(i);
    }

    return inliers;
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

    if (m_dist == ALGEBRAIC)
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

    if (m_dist == SAMPSON)
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

    m_data.src = (d0 == 3 ? Geometry::MakeHomogeneous(data.src) : data.src); //.Reshape(Geometry::ROW_MAJOR);
    m_data.dst = (d1 == 3 ? Geometry::FromHomogeneous(data.dst) : data.dst); //.Reshape(Geometry::ROW_MAJOR);

    return true;
}

cv::Mat ProjectionObjective::operator() (const EuclideanTransform& tform) const
{
    if (!m_proj)
    {
        E_ERROR << "missing projection model";
        return cv::Mat();
    }

    const Metric& d = *m_data.metric;

    Geometry x = m_data.src;
    Geometry y = m_proj->Project(tform(x), ProjectionModel::EUCLIDEAN_2D);

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

    if (d0 != 3 || d0 != 4)
    {
        E_ERROR << "source geometry has to be either 3D Euclidean or 4D homogeneous";
        return false;
    }

    if (d1 != dI)
    {
        E_ERROR << "dimensionality mismatches between the referenced image (d=" << dI << ") and the target (d=" << d1 << ")";
        return false;
    }

    m_data.src = (d0 == 3 ? Geometry::MakeHomogeneous(data.src) : data.src);
    m_data.dst = data.dst;

    return true;
}

cv::Mat PhotometricObjective::operator() (const EuclideanTransform& tform) const
{
    if (!m_proj)
    {
        E_ERROR << "missing projection model";
        return cv::Mat();
    }

    const Metric& d = *m_data.metric;

    Geometry x = m_data.src;
    cv::Mat y;
    cv::remap(m_dst, y, m_proj->Project(tform(x), ProjectionModel::EUCLIDEAN_2D).mat.reshape(2), cv::Mat(), cv::INTER_NEAREST);
    
    return d(m_data.dst, Geometry(x.shape, y.reshape(1))).mat;
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

    m_data.src = (d0 == 3 ? Geometry::MakeHomogeneous(data.src) : data.src);
    m_data.dst = (d1 == 3 ? Geometry::MakeHomogeneous(data.dst) : data.dst);

    return true;
}

cv::Mat RigidObjective::operator() (const EuclideanTransform& tform) const
{
    const Metric& d = *m_data.metric;

    Geometry x = m_data.src;
    Geometry y = tform(x);

    return d(m_data.dst, y).mat;
}

//==[ PoseEstimation ]========================================================//

//==[ EssentialMatDecomposition ]=============================================//

bool EssentialMatDecomposition::operator() (EuclideanTransform& pose) const
{
    return false;
}

//==[ PerspevtivePoseEstimation ]=============================================//

bool PerspevtivePoseEstimation::operator() (EuclideanTransform& pose) const
{
    return false;
}

//==[ QuatAbsOrientationSolver ]==============================================//

bool QuatAbsOrientationSolver::operator() (EuclideanTransform& pose) const
{
    return false;
}

//==[ MultiObjectivePoseEstimation ]==========================================//

bool MultiObjectivePoseEstimation::operator() (EuclideanTransform& pose) const
{
    return false;
}

VectorisableD::Vec MultiObjectivePoseEstimation::Initialise()
{
    // TODO: improve the initialisation
    //
    //

    m_conds = GetConds();

    return m_transform.ToVector();
}

size_t MultiObjectivePoseEstimation::GetConds() const
{
    size_t m = 0;

    BOOST_FOREACH (const AlignmentObjective::Own& obj, m_objectives)
    {
        if (!obj) continue;
        m += obj->GetData().dst.GetElements();
    }

    return m;
}

VectorisableD::Vec MultiObjectivePoseEstimation::operator()(const VectorisableD::Vec & x) const
{
    EuclideanTransform tform;
    tform.FromVector(x);

    VectorisableD::Vec y;
    size_t m = 0;

    y.reserve(m_conds);

    BOOST_FOREACH (const AlignmentObjective::Own& obj, m_objectives)
    {
        if (!obj) continue;
        
        cv::Mat yi = (*obj)(tform);
        size_t  mi = static_cast<size_t>(yi.total());

        assert(yi.type == CV_64F);
        assert(m + mi <= m_conds);

        y.insert(y.end(), (double*)yi.datastart, (double*)yi.dataend);

        m += mi;
    }

    assert(m == m_conds);

    return y;
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

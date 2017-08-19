#include <seq2map/geometry.hpp>

using namespace seq2map;

namespace seq2map
{
    cv::Mat skewsymat(const cv::Mat& x)
    {
        assert(x.total() == 3);

        cv::Mat x64f;
        x.convertTo(x64f, CV_64F);

        // normalisation to make sure ||x|| = 1
        x64f /= cv::norm(x64f);

        double* v = x64f.ptr<double>();

        cv::Mat y64f = (cv::Mat_<double>(3, 3) <<
             0.0f,  -v[2],  v[1],
             v[2],   0.0f, -v[0],
            -v[1],   v[0], 0.0f);

        cv::Mat y;
        y64f.convertTo(y, x.type());

        return y;
    }
}

//==[ Geometry ]==============================================================//

Geometry& Geometry::operator= (const Geometry& g)
{
    if (!Reshape(g.shape, shape, mat = g.mat))
    {
        // make a deep copy if mat remains a shallow copy of g.mat
        mat = mat.clone();
    }

    return *this;
}

Geometry& Geometry::operator= (Geometry& g)
{
    Reshape(g.shape, shape, mat = g.mat);
    return *this;
}

Geometry Geometry::operator- (const Geometry& g) const
{
    if (!IsConsistent(g)) return Geometry(shape);

    Geometry d(shape);

    if (shape == g.shape)
    {
        cv::subtract(mat, g.mat, d.mat);
    }
    else
    {
        cv::subtract(mat, g.Reshape(shape).mat, d.mat);
    }

    return d;
}

Geometry Geometry::Step(size_t dim, double eps) const
{
    int d = static_cast<int>(dim);
    cv::Mat g = mat.clone();

    switch (shape)
    {
    case ROW_MAJOR: g.col(d) += eps; break;
    case COL_MAJOR: g.row(d) += eps; break;
    case PACKED:    g.reshape(1, mat.rows * mat.cols).col(d) += eps; break;
    }

    return Geometry(shape, g);
}

bool Geometry::IsConsistent(const Geometry& g) const
{
    if (GetDimension() != g.GetDimension() || GetElements() != g.GetElements())
    {
        E_WARNING << "inconsistent dimensionality";

        throw std::exception("shit!");

        return false;
    }

    return true;
}

size_t Geometry::GetElements() const
{
    switch (shape)
    {
    case ROW_MAJOR: return mat.rows; break;
    case COL_MAJOR: return mat.cols; break;
    case PACKED:    return mat.rows * mat.cols; break;
    default:        return 0; // invalid shape
    }
}

size_t Geometry::GetDimension() const
{
    switch (shape)
    {
    case ROW_MAJOR: return mat.cols; break;
    case COL_MAJOR: return mat.rows; break;
    case PACKED:    return mat.channels(); break;
    default:        return 0; // invalid shape
    }
}

Geometry Geometry::Reshape(const Geometry& g) const
{
    if (!IsConsistent(g) || g.shape != Geometry::PACKED)
    {
        return Reshape(g.shape);
    }

    cv::Mat m = mat;
    
    if (!Reshape(shape, g.shape, m, g.mat.rows))
    {
        m = m.clone();
    }

    return Geometry(g.shape, m);
}

Geometry Geometry::Reshape(const Geometry& g)
{
    if (!IsConsistent(g) || g.shape != Geometry::PACKED)
    {
        return Reshape(g.shape);
    }

    cv::Mat m = mat;
    Reshape(shape, g.shape, m, g.mat.rows);

    return Geometry(g.shape, m);
}

bool Geometry::Reshape(Shape src, Shape dst, cv::Mat& mat, size_t rows)
{
    if (src == dst) // same shape, nothing to be done
    {
        return false;
    }

    if (src != PACKED && dst != PACKED) // simple transpose
    {
        mat = mat.t(); // element cloning
        return true;
    }

    switch (src)
    {
    case ROW_MAJOR: // ROW_MAJOR -> PACKED
        mat = mat.reshape(mat.cols, static_cast<int>(rows));
        return false;

    case COL_MAJOR: // COL_MAJOR -> PACKED
        mat = cv::Mat(mat.t()).reshape(mat.rows, static_cast<int>(rows));
        return true;
    }

    switch (dst)
    {
    case ROW_MAJOR: // PACKED -> ROW_MAJOR
        mat = mat.reshape(1, mat.rows * mat.cols);
        return false;

    case COL_MAJOR: // PACKED -> COL_MAJOR
        mat = mat.reshape(1, mat.rows * mat.cols).t();
        return true;
    }

    assert(0); // should never reach this point!
    return false;
}

Geometry Geometry::MakeHomogeneous(const Geometry& g, double w)
{
    const cv::Mat& src = g.mat;
    /***/ cv::Mat  dst;

    const int m = static_cast<int>(g.GetElements());

    switch (g.shape)
    {
    case ROW_MAJOR:
        dst = cv::Mat(src.rows, src.cols + 1, src.type());
        src.copyTo(dst.colRange(0, src.cols));
        dst.col(src.cols).setTo(w);
        break;

    case COL_MAJOR:
        dst = cv::Mat(src.rows + 1, src.cols, src.type());
        src.copyTo(dst.rowRange(0, src.rows));
        dst.row(src.rows).setTo(w);
        break;

    case PACKED:
        //cv::convertPointsToHomogeneous(src, dst);
        dst = cv::Mat(src.rows, src.cols, CV_MAKETYPE(src.depth(), src.channels() + 1));
        src.reshape(1, m).copyTo(dst.reshape(1, m).colRange(0, src.channels()));
        dst.reshape(1, m).col(src.channels()).setTo(w);
        break;
    }

    return Geometry(g.shape, dst);
}

Geometry Geometry::FromHomogeneous(const Geometry& g)
{
    const cv::Mat& src = g.mat;
    /***/ cv::Mat  dst;

    const int m = static_cast<int>(g.GetElements());

    switch (g.shape)
    {
    case ROW_MAJOR:
        dst = cv::Mat(src.rows, src.cols - 1, src.type());
        for (int j = 0; j < dst.cols; j++)
        {
            cv::divide(src.col(j), src.col(dst.cols), dst.col(j));
        }
        break;

    case COL_MAJOR:
        dst = cv::Mat(src.rows - 1, src.cols, src.type());
        for (int i = 0; i < dst.rows; i++)
        {
            cv::divide(src.row(i), src.row(dst.rows), dst.row(i));
        }
        break;

    case PACKED:
        //cv::convertPointsFromHomogeneous(src, dst);
        dst = cv::Mat(src.rows, src.cols, CV_MAKETYPE(src.depth(), src.channels() - 1));
        dst = dst.reshape(1, m); // to row major order
        for (int j = 0; j < dst.cols; j++)
        {
            cv::divide(src.reshape(1, m).col(j), src.reshape(1, m).col(dst.cols), dst.col(j));
        }
        dst = dst.reshape(src.channels() - 1, src.rows);
        break;
    }

    return Geometry(g.shape, dst);
}

Geometry& Geometry::Dehomogenise()
{
    if (shape == PACKED)
    {
        this->Reshape(ROW_MAJOR).Dehomogenise(); // operate on a shallow copy of myself
        return *this;
    }

    const int n = static_cast<int>(GetDimension()) - 1;

    switch (shape)
    {
    case ROW_MAJOR:
        for (int j = 0; j < mat.cols; j++)
        {
            cv::divide(mat.col(j), mat.col(n), mat.col(j));
        }
        break;

    case COL_MAJOR:
        for (int i = 0; i < mat.rows; i++)
        {
            cv::divide(mat.row(i), mat.row(n), mat.row(i));
        }
        break;
    }

    return *this;
}

Geometry Geometry::operator[] (const IndexList& indices) const
{
    Geometry g(shape);

    const int n = static_cast<int>(indices.size());
    int i = 0;

    switch (shape)
    {
    case ROW_MAJOR:
        g.mat = cv::Mat(n, mat.cols, mat.type());
        BOOST_FOREACH (size_t idx, indices)
        {
            mat.row(static_cast<int>(idx)).copyTo(g.mat.row(i++));
        }
        break;

    case COL_MAJOR:
        g.mat = cv::Mat(mat.rows, n, mat.type());
        BOOST_FOREACH (size_t idx, indices)
        {
            mat.col(static_cast<int>(idx)).copyTo(g.mat.col(i++));
        }
        break;

    case PACKED:
        g = Reshape(ROW_MAJOR)[indices]; // the assignment is required to make the shape right
    }

    return g;
}

//==[ Metric ]================================================================//

//==[ EuclideanMetric ]=======================================================//

Geometry EuclideanMetric::operator() (const Geometry& x) const
{
    if (x.GetDimension() == 1)
    {
        return Geometry(x.shape, cv::abs(x.mat) * scale);
    }

    if (x.shape != Geometry::ROW_MAJOR)
    {
        return (*this)(x.Reshape(Geometry::ROW_MAJOR)).Reshape(x.shape);
    }

    cv::Mat d = cv::Mat::zeros(x.mat.rows, 1, x.mat.type());
    const double s2 = scale * scale;

    // work only for row-major shape
    for (int j = 0; j < x.mat.cols; j++)
    {
        cv::Mat dj;
        cv::multiply(x.mat.col(j), x.mat.col(j), dj, s2); // dj = dj .^ 2
        cv::add(d, dj, d);                                // d  = d + dj
    }

    cv::sqrt(d, d); // DIMS = sqrt(DIMS)

    return Geometry(x.shape, d);
}

Metric::Own EuclideanMetric::operator+ (const Metric& metric) const
{
    const EuclideanMetric* m = dynamic_cast<const EuclideanMetric*>(&metric);

    if (!m)
    {
        E_ERROR << "incompatible metric";
        return Metric::Own();
    }

    return Metric::Own(new EuclideanMetric(0.5f * (scale + m->scale)));
}

//==[ WeightedEuclideanMetric ]===============================================//

WeightedEuclideanMetric::WeightedEuclideanMetric(cv::Mat weights, double scale)
: bayesian(true), m_weights(weights.cols == 1 && weights.channels() == 1 ? weights : cv::Mat()), EuclideanMetric(scale)
{
    if (m_weights.empty())
    {
        E_ERROR << "weights has to be a non-empty row vector, while the given matrix is "
            << weights.rows << "x" << weights.cols << "x" << weights.channels();

        throw std::exception("!!");
    }
}

Geometry WeightedEuclideanMetric::operator() (const Geometry& x) const
{
    Geometry d(Geometry::ROW_MAJOR);

    if (m_weights.rows != x.GetElements())
    {
        E_ERROR << "inconsistent geometry weightings, the given geometry has "
                << x.GetElements() << " element(s) while there are "
                << m_weights.rows  << " weighting term(s)";

        return d;
    }

    d = EuclideanMetric::operator()(x);
    cv::multiply(d.mat, m_weights, d.mat, 1.0f, m_weights.type());

    for (int i = 0; i < d.mat.rows; i++)
    {
        if (!std::isfinite(d.mat.at<double>(i)))
        {
            E_ERROR << i;
            E_ERROR << x.mat.row(i);
            E_ERROR << d.mat.at<double>(i);
            E_ERROR << m_weights.at<double>(i);

            throw std::exception("GG");
        }
    }

    return d.Reshape(x.shape);
}

Metric::Own WeightedEuclideanMetric::operator[] (const IndexList& indices) const
{
    cv::Mat w = cv::Mat(static_cast<int>(indices.size()), 1, m_weights.depth());

    int dst = 0;
    BOOST_FOREACH (size_t src, indices)
    {
        m_weights.row(static_cast<int>(src)).copyTo(w.row(dst++));
    }

    return Metric::Own(new WeightedEuclideanMetric(w, scale));
}

Metric::Own WeightedEuclideanMetric::operator+ (const Metric& metric) const
{
    const WeightedEuclideanMetric* m = dynamic_cast<const WeightedEuclideanMetric*>(&metric);

    if (!m)
    {
        E_ERROR << "incompatible metric";
        return Metric::Own();
    }

    if (m_weights.rows != m->m_weights.rows)
    {
        E_ERROR << "inconsistent weightings";
        return Metric::Own();
    }

    const cv::Mat& w0 = m_weights;
    const cv::Mat& w1 = m->m_weights;

    cv::Mat weights = bayesian ? w0.mul(w1) / (w0 + w1) : 0.5f * (w0 + w1);

    return Metric::Own(new WeightedEuclideanMetric(weights, 0.5f * (scale + m->scale)));
}

boost::shared_ptr<MahalanobisMetric> WeightedEuclideanMetric::ToMahalanobis(bool& native)
{
    cv::Mat w = m_weights;
    cv::Mat cov = 1 / w.mul(w);

    for (int i = 0; i < w.rows; i++)
    {
        if (w.at<double>(i) <= 0) cov.at<double>(i) = 0;
    }

    native = false;
    return boost::shared_ptr<MahalanobisMetric>(new MahalanobisMetric(MahalanobisMetric::ISOTROPIC, 3, cov));
}

boost::shared_ptr<const MahalanobisMetric> WeightedEuclideanMetric::ToMahalanobis() const
{
    bool native;
    return Clone()->ToMahalanobis(native);
}

bool WeightedEuclideanMetric::FromMahalanobis(const MahalanobisMetric& metric)
{
    if (metric.type != MahalanobisMetric::ISOTROPIC)
    {
        return false;
    }

    cv::Mat w;
    
    cv::sqrt(metric.GetCovariance().mat, w);
    m_weights = 1 / w;

    return true;
}

//==[ MahalanobisMetric ]=====================================================//

double MahalanobisMetric::s_rcondThreshold = 1e-10;

boost::shared_ptr<MahalanobisMetric> MahalanobisMetric::Identity(size_t numel, size_t dims, int depth)
{
    const int ELEM = static_cast<int>(numel);
    const int DIMS = static_cast<int>(dims);

    MahalanobisMetric* metric = new MahalanobisMetric(ANISOTROPIC_ROTATED, dims);
    metric->m_cov.mat = cv::Mat::zeros(ELEM, static_cast<int>(metric->GetCovMatCols()), depth);

    for (int d = 0; d < DIMS; d++)
    {
        metric->m_cov.mat.col(sub2symind(d, d, DIMS)).setTo(1.0f);
    }

    return boost::shared_ptr<MahalanobisMetric>(metric);
}

MahalanobisMetric::MahalanobisMetric(CovarianceType type, size_t dims, const cv::Mat& cov)
: MahalanobisMetric(type, dims)
{
    if (!SetCovarianceMat(cov))
    {
        E_ERROR << "error setting covariance matrix";
    }
}

Metric::Own MahalanobisMetric::operator[] (const IndexList& indices) const
{
    MahalanobisMetric* metric = new MahalanobisMetric(type, dims);
    metric->m_cov = m_cov[indices];

    return Metric::Own(metric);
}

Metric::Own MahalanobisMetric::operator+ (const Metric& metric) const
{
    const MahalanobisMetric* m = dynamic_cast<const MahalanobisMetric*>(&metric);

    if (!m)
    {
        E_ERROR << "incompatible metric";
        return Metric::Own();
    }

    if (!m->m_cov.IsConsistent(m_cov))
    {
        E_ERROR << "inconsistent covariance";
        return Metric::Own();
    }

    cv::Mat cov = 8 * (m_cov.mat + m->m_cov.mat);

    return Metric::Own(new MahalanobisMetric(type, dims, cov));
}

Metric::Own MahalanobisMetric::Transform(const EuclideanTransform& tform, const Geometry& jac) const
{
    const bool jacobianOnly = tform.IsIdentity();

    if (!jacobianOnly && dims != 3)
    {
        E_ERROR << "An Euclidean transformation can only apply to a 3D Mahalanobis metric (d=" << dims << ")";
        return Metric::Own();
    }

    const size_t d0 = dims;
    const size_t d1 = jac.IsEmpty() ? d0 : jac.GetDimension() / dims;
    const size_t n0 = m_cov.GetElements();
    const size_t n1 = jac.GetElements();
    // const int dtype = jac.IsEmpty() ? m_cov.mat.depth() : jac.mat.depth();

    if (!jac.IsEmpty())
    {
        if (jac.shape != Geometry::ROW_MAJOR)
        {
            return Transform(tform, jac.Reshape(Geometry::ROW_MAJOR));
        }

        if (n0 != 1 && n1 != 1 && n0 != n1)
        {
            E_ERROR << "given Jacobian is for " << n1 << " element(s) while the metric has " << n0;
            return Metric::Own();
        }

        if (jac.GetDimension() != d0 * d1)
        {
            E_ERROR << "given Jacobian has invalid dimension of " << jac.GetDimension();
            return Metric::Own();
        }
    }
    else if (jacobianOnly)
    {
        return Clone();
    }

    const int D0 = static_cast<int>(d0);
    const int D1 = static_cast<int>(d1);
    const int N1 = static_cast<int>(n0 == 1 ? n1 : n0);
    const int DIMS = static_cast<int>(dims);

    MahalanobisMetric* metric = new MahalanobisMetric(ANISOTROPIC_ROTATED, D1);
    const cv::Mat cov0 = GetFullCovMat();
    /***/ cv::Mat cov1 = cv::Mat(N1, static_cast<int>(metric->GetCovMatCols()), cov0.depth());
    cv::Mat R = !jacobianOnly ? tform.GetRotation().ToMatrix() : cv::Mat();

    if (!R.empty() && R.depth() != cov0.depth()) 
    {
        R.convertTo(R, cov0.depth());
    }

    if (jac.IsEmpty())
    {
        cv::Mat Rt = R.t();

        for (int i = 0; i < cov0.rows; i++)
        {
            cv::Mat Ci = cv::Mat(DIMS, DIMS, cov0.type());

            for (int d0 = 0; d0 < DIMS; d0++)
            {
                for (int d1 = 0; d1 < DIMS; d1++)
                {
                    const int j = sub2symind(d0, d1, DIMS);
                    Ci.at<double>(d0, d1) = cov0.at<double>(i, j);
                }
            }

            Ci = R * Ci * Rt;

            for (int d0 = 0; d0 < DIMS; d0++)
            {
                for (int d1 = d0; d1 < DIMS; d1++)
                {
                    const int j = sub2symind(d0, d1, DIMS);
                    cov1.at<double>(i, j) = Ci.at<double>(d0, d1);
                }
            }
        }
    }
    else if (n0 == 1) // one covariance expanded via many Jacobians (one-to-many)
    {
        cv::Mat C = symmat(cov0, D0);

        for (int i = 0; i < N1; i++)
        {
            cv::Mat Ai = cv::Mat(D1, D0, cov0.type());

            for (int d0 = 0; d0 < D0; d0++)
            {
                for (int d1 = 0; d1 < D1; d1++)
                {
                    Ai.at<double>(d1, d0) = jac.mat.at<double>(i, d0 * D1 + d1);
                }
            }

            if (!R.empty()) Ai = Ai * R;

            cv::Mat Ci = Ai * C * Ai.t();
            symmat(Ci).copyTo(cov1.row(i));
        }
    }
    else if (n1 == 1) // many covariances propagated via one Jacobian (many-to-one)
    {
        cv::Mat A = cv::Mat(D1, D0, cov0.type());

        for (int d0 = 0; d0 < D0; d0++)
        {
            for (int d1 = 0; d1 < D1; d1++)
            {
                A.at<double>(d1, d0) = jac.mat.at<double>(0, d0 * D1 + d1);
            }
        }

        if (!R.empty())
        {
            A = A * R;
        }

        cv::Mat At = A.t();

        for (int i = 0; i < N1; i++)
        {
            symmat(A * symmat(cov0.row(i), D0) * At).copyTo(cov1.row(i));
        }
    }
    else // propagation via per-covariance Jacobian (one-to-one)
    {
        for (int i = 0; i < N1; i++)
        {
            cv::Mat Ai = cv::Mat(D1, D0, cov0.type());
            cv::Mat Ci = cv::Mat(D0, D0, cov0.type());

            for (int d0 = 0; d0 < D0; d0++)
            {
                for (int d1 = 0; d1 < D0; d1++)
                {
                    const int j = sub2symind(d0, d1, D0);
                    Ci.at<double>(d0, d1) = cov0.at<double>(i, j);
                }

                for (int d1 = 0; d1 < D1; d1++)
                {
                    Ai.at<double>(d1, d0) = jac.mat.at<double>(i, d0 * D1 + d1);
                }
            }

            if (!R.empty())
            {
                Ai = Ai * R;
            }

            Ci = Ai * Ci * Ai.t(); // Ci becomes D1 x D1, was D0 x D0

            symmat(Ci).copyTo(cov1.row(i));
        }
    }

    metric->SetCovarianceMat(cov1);

    return Metric::Own(metric);
}

Metric::Own MahalanobisMetric::Reduce() const
{
    cv::Mat cov;
    const int DIMS = static_cast<int>(dims);

    switch (type)
    {
    case ISOTROPIC:
        cv::sqrt(m_cov.mat, cov);
        break;

    case ANISOTROPIC_ORTHOGONAL:
        cv::sqrt(m_cov.mat, cov);
        cv::reduce(cov, cov, 1, cv::REDUCE_AVG);
        break;

    case ANISOTROPIC_ROTATED:
        // find the lengths of three ellipsoidal axes (i.e. square roots of the covaraicne matrix's eigenvalues)
        // and use the averaged lengths as the inverse weight
        cov = cv::Mat(m_cov.mat.rows, 3, m_cov.mat.depth());

        for (int i = 0; i < cov.rows; i++)
        {
            cv::Mat eigenVals;

            if (!eigen(symmat(m_cov.mat.row(i), DIMS), eigenVals))
            {
                cov.row(i).setTo(0.0f);
                continue;
            }
            
            eigenVals.reshape(1, 1).copyTo(cov.row(i));
        }

        cv::sqrt(cov, cov);
        cv::reduce(cov, cov, 1, cv::REDUCE_AVG);
    }

    // the weights are inverses of the reduced sigma
    cv::Mat icv = 1 / cov;

    // singularity check
    for (int i = 0; i < icv.rows; i++) 
    {
        if (!std::isfinite(icv.at<double>(i)))
        {
            icv.row(i).setTo(0.0f);
        }
    }

    return Metric::Own(new WeightedEuclideanMetric(icv));
}

cv::Mat MahalanobisMetric::Update(const MahalanobisMetric& metric)
{
    if (this->type != metric.type)
    {
        E_ERROR << "incompatible types (d0=" << this->type << ", d1=" << metric.type << ")";
        return cv::Mat();
    }

    if (m_cov.GetElements() != metric.m_cov.GetElements())
    {
        E_ERROR << "incompatible dimensionality (n0=" << m_cov.GetElements() << ", n1=" << metric.m_cov.GetElements() << ")";
        return cv::Mat();
    }

    /***/ cv::Mat K;
    /***/ cv::Mat& cov0 = m_cov.mat;
    const cv::Mat& cov1 = metric.m_cov.mat;

    if (type != ANISOTROPIC_ROTATED)
    {
        cv::Mat sum;
        cv::add(cov0, cov1, sum);
        cv::divide(cov0, sum, K); // K = C0 / (C0 + C1)

        if (type == ISOTROPIC)
        {
            for (int j = 0; j < cov0.cols; j++)
            {
                cv::Mat sub;
                cv::multiply(cov0.col(j), K, sub);
                cv::subtract(cov0.col(j), sub, cov0.col(j)); // C0 = C0 - K * C0
            }
        }
        else
        {
            for (int j = 0; j < cov0.cols; j++)
            {
                cv::Mat sub;
                cv::multiply(cov0.col(j), K.col(j), sub);
                cv::subtract(cov0.col(j), sub, cov0.col(j)); // C0 = C0 - K * C0
            }
        }

        for (int i = 0; i < K.rows; i++)
        {
            double& Ki = K.at<double>(i);

            if (Ki <= 0 || !isfinite(Ki))
            {
                Ki = 1.0f;
                cov1.row(i).copyTo(cov0.row(i));
            }
        }

        return K;
    }

    const int DIMS = static_cast<int>(dims);

    // initialise Kalman gain with the identity matrix
    K = cv::Mat::zeros(cov0.rows, DIMS * DIMS, cov0.depth());
    for (int d = 0; d < DIMS; d++)
    {
        K.col(d * DIMS + d).setTo(1.0f);
    }

    for (int i = 0; i < cov0.rows; i++)
    {
        // check the validity using the first covariance coefficient C_{00}
        const bool valid0 = cov0.at<double>(i, 0) > 0;
        const bool valid1 = cov1.at<double>(i, 0) > 0;

        // use the covariance from the given metric as the initial state
        if (!valid0 && valid1)
        {
            cov1.row(i).copyTo(cov0.row(i));
            continue;
        }

        // deny to update using second measure if it is not valid
        if (!valid1)
        {
            K.row(i).setTo(0.0f); // nullify the Kalman gain
            continue;
        }

        // make a full covariance matrix
        cv::Mat sum = cv::Mat(DIMS, DIMS, cov0.depth());
        cv::Mat cov = cv::Mat(DIMS, DIMS, cov0.depth());

        for (int d0 = 0; d0 < DIMS; d0++)
        {
            for (int d1 = 0; d1 < DIMS; d1++)
            {
                const int j = sub2symind(d0, d1, DIMS);
                const double C0_ij = cov0.at<double>(i, j);

                cov.at<double>(d0, d1) = C0_ij;
                sum.at<double>(d0, d1) = C0_ij + cov1.at<double>(i, j);
            }
        }

        // find the inverse of sum of the i-th covariance matrices
        cv::Mat inv; // inverse of C0 + C1
        cv::Mat kal; // Kalman gain, in full matrix
        double rcond = cv::invert(sum, inv, cv::DECOMP_SVD);

        if (rcond < s_rcondThreshold)
        {
            E_TRACE << "entry " << i << " will not be updated as it is possibly singular (RCOND = " << rcond << ")";
            K.row(i).setTo(0.0f); // deny the update and nullify the Kalman gain

            continue;
        }

        kal = cov * inv;       // K = C0 * inv(C0 + C1)
        cov = cov - kal * cov; // updated covariance: C0' = C0 - K * C0;

        // check positive definiteness
#ifndef NDEBUG
        if (!checkPositiveDefinite(cov))
        {
            E_ERROR << "entry " << i << " has non-positive-definite covariance";
            E_ERROR << "C0 = "  << mat2string(cov0.row(i));
            E_ERROR << "C1 = "  << mat2string(cov1.row(i));
            E_ERROR << "inv = " << mat2string(inv);
            E_ERROR << "sum = " << mat2string(sum);
            E_ERROR << "kal = " << mat2string(kal);
            E_ERROR << "cov = " << mat2string(cov);
            E_ERROR << "rcond = " << rcond;
        }
#endif

        kal.reshape(1, 1).copyTo(K.row(i));

        // fill the upper triangle parts back to the output matrices
        symmat(cov).copyTo(cov0.row(i));
    }

    return K;
}

size_t MahalanobisMetric::GetCovMatCols(CovarianceType type, size_t dims)
{
    switch (type)
    {
    case ISOTROPIC:              return 1;
    case ANISOTROPIC_ORTHOGONAL: return dims;
    case ANISOTROPIC_ROTATED:    return dims * (dims + 1) / 2;
    }

    return 0;
}

bool MahalanobisMetric::SetCovarianceMat(const cv::Mat& cov)
{
    if (cov.channels() != 1)
    {
        E_WARNING << "covariance matrix has to be single-channel (d=" << cov.channels() << ")";
        return false;
    }

    const size_t cols = GetCovMatCols();

    if (cols != static_cast<size_t>(cov.cols))
    {
        E_WARNING << "covariance matrix has " << cov.cols << " column(s) instead of " << cols;
        return false;
    }

    m_cov.mat = cov.clone();
    m_icv.mat = cv::Mat(); // do lazy evaluation
    // m_icv.mat = GetInverseCovMat(cov);

    const int DIMS = static_cast<int>(dims);

    for (int i = 0; i < cov.rows; i++)
    {
        if (/*cov.at<double>(i,0) != 0 &&*/ !checkPositiveDefinite(symmat(cov.row(i), DIMS)))
        {
            E_TRACE << "entry " << i << " is not positive-definite or nearly singular";
            m_cov.mat.row(i).setTo(0.0f); // nullify the entry
        }
    }

    return true;
}

cv::Mat MahalanobisMetric::GetInverseCovMat(const cv::Mat& cov) const
{
    if (type != ANISOTROPIC_ROTATED)
    {
        return 1 / cov;
    }

    const int DIMS = static_cast<int>(dims);
    cv::Mat icv = cv::Mat(cov.rows, cov.cols, cov.type());

    for (int i = 0; i < cov.rows; i++)
    {
        cv::Mat S2_inv;
        double rcond = cv::invert(symmat(cov.row(i), DIMS), S2_inv, cv::DECOMP_SVD);

        if (rcond < s_rcondThreshold)
        {
            E_TRACE << "the inverse of entry " << i << " is badly scaled hence discarded (RCOND = " << rcond << ")";
            icv.row(i).setTo(0.0f);

            continue;
        }

        symmat(S2_inv).copyTo(icv.row(i));
    }

    return icv;
}

cv::Mat MahalanobisMetric::GetFullCovMat() const
{
    if (m_cov.IsEmpty())
    {
        E_WARNING << "empty covariance matrix";
        return cv::Mat();
    }

    if (type == ANISOTROPIC_ROTATED)
    {
        return m_cov.mat.clone();
    }

    const int DIMS = static_cast<int>(dims);
    const int COLS = static_cast<int>(GetCovMatCols(ANISOTROPIC_ROTATED, dims));

    cv::Mat cov = cv::Mat::zeros(m_cov.mat.rows, COLS, m_cov.mat.depth());

    switch (type)
    {
    case ISOTROPIC:

        for (int d = 0; d < DIMS; d++)
        {
            const int j = sub2symind(d, d, DIMS);
            m_cov.mat.copyTo(cov.col(j));
        }

        break;

    case ANISOTROPIC_ORTHOGONAL:

        for (int d = 0; d < DIMS; d++)
        {
            const int j = sub2symind(d, d, DIMS);
            m_cov.mat.col(d).copyTo(cov.col(j));
        }

    }

    return cov;
}

cv::Mat MahalanobisMetric::GetFullCovMat(size_t index) const
{
    const int ROW = static_cast<int>(index);
    const int DIMS = static_cast<int>(dims);
    cv::Mat C = cv::Mat::zeros(DIMS, DIMS, m_cov.mat.depth());

    switch (type)
    {
    case ISOTROPIC:

        for (int d = 0; d < DIMS; d++)
        {
            const int j = sub2symind(d, d, DIMS);
            C.at<double>(d, d) = m_cov.mat.at<double>(ROW, 0);
        }

        break;

    case ANISOTROPIC_ORTHOGONAL:

        for (int d = 0; d < DIMS; d++)
        {
            const int j = sub2symind(d, d, DIMS);
            C.at<double>(d, d) = m_cov.mat.at<double>(ROW, d);
        }

        break;

    case ANISOTROPIC_ROTATED:

        C = symmat(m_cov.mat.row(ROW), DIMS);

        break;
    }

    return C;
}

boost::shared_ptr<MahalanobisMetric> MahalanobisMetric::ToMahalanobis(bool& native)
{
    native = true;
    return boost::static_pointer_cast<MahalanobisMetric, Metric>(shared_from_this());
}

boost::shared_ptr<const MahalanobisMetric> MahalanobisMetric::ToMahalanobis() const
{
    return boost::static_pointer_cast<MahalanobisMetric, Metric>(Clone());
}

bool MahalanobisMetric::FromMahalanobis(const MahalanobisMetric& metric)
{
    if (metric.type != this->type || metric.dims != this->dims)
    {
        return false;
    }

    SetCovarianceMat(metric.GetCovariance().mat);
    return true;
}

Geometry MahalanobisMetric::operator() (const Geometry& x) const
{
    if (x.shape != Geometry::ROW_MAJOR)
    {
        return (*this)(x.Reshape(Geometry::ROW_MAJOR)).Reshape(x.shape);
    }

    if (dims != x.GetDimension())
    {
        E_ERROR << "mis-matched dimensionalities (" << dims << " != " << x.GetDimension() << ")";
        return Geometry(x.shape);
    }

    if (m_cov.GetElements() != x.GetElements())
    {
        E_ERROR << "mis-matched number of elements (" << m_cov.GetElements() << " != " << x.GetElements() << ")";
        return Geometry(x.shape);
    }

    if (m_icv.mat.empty())
    {
        m_icv.mat = GetInverseCovMat(m_cov.mat);
    }

    cv::Mat d = cv::Mat::zeros(x.mat.rows, 1, x.mat.depth());

    switch (type)
    {
    case ISOTROPIC:

        for (int j = 0; j < x.mat.cols; j++)
        {
            cv::Mat dj;
            cv::multiply(x.mat.col(j), x.mat.col(j), dj, 1.0f, d.depth()); // dj = xj .^ 2
            cv::add(d, dj, d);                                             // d  = d + dj
        }

        cv::multiply(m_icv.mat, d, d, 1.0f, d.depth()); // d = w .* d
        cv::sqrt(d, d);                                 // d = sqrt(d)

        break;

    case ANISOTROPIC_ORTHOGONAL:

        for (int j = 0; j < x.mat.cols; j++)
        {
            cv::Mat dj;
            cv::multiply(x.mat.col(j), x.mat.col(j), dj, 1.0f, d.depth()); // dj = xj .^ 2
            cv::multiply(m_icv.mat.col(j), dj, dj, 1.0f, d.depth());       // dj = wj .* dj
            cv::add(d, dj, d);                                             // d  = d + dj
        }

        cv::sqrt(d, d); // DIMS = sqrt(DIMS)

        break;

    case ANISOTROPIC_ROTATED:

        for (int j0 = 0; j0 < x.mat.cols; j0++)
        {
            for (int j1 = 0; j1 < x.mat.cols; j1++)
            {
                const int j = sub2symind(j0, j1, static_cast<int>(dims));
                cv::Mat dj;

                cv::multiply(x.mat.col(j0), x.mat.col(j1), dj, 1.0f, d.depth()); // dj = x_j0 .* x_j1
                cv::multiply(m_icv.mat.col(j), dj, dj, 1.0f, d.depth());         // dj = wj .* dj
                cv::add(d, dj, d);
            }
        }

        // check..
        /*
        for (int i = 0; i < x.mat.rows; i++)
        {
            const double di = d.at<double>(i);
            if (di < 0 || !std::isfinite(di))
            {
                E_ERROR << "entry " << i << " has invalid squared distance " << di;
                E_ERROR << "dist: " << mat2string(x.mat.row(i));
                E_ERROR << "cov: "  << mat2string(m_cov.mat.row(i));
                E_ERROR << "icv: "  << mat2string(m_icv.mat.row(i));

                return Geometry(x.shape);
            }
        }
        */

        cv::sqrt(d, d);

        break;
    }
    
    return Geometry(x.shape, d);
}

//==[ GeometricMapping ]======================================================//

bool GeometricMapping::IsConsistent() const
{
    const size_t n = src.GetElements();
    return (n > 0 && n == dst.GetElements()) && (indices.empty() || indices.size() == n);;
}

bool GeometricMapping::Check(size_t d0, size_t d1) const
{
    if (!IsConsistent())
    {
        E_ERROR << "consistency check failed";
        return false;
    }

    if (src.GetDimension() != d0)
    {
        E_ERROR << "source points have to be in " << d0 << "D space (d=" << src.GetDimension() << ")";
        return false;
    }

    if (dst.GetDimension() != d1)
    {
        E_ERROR << "target points have to be in " << d1 << "D space (d=" << dst.GetDimension() << ")";
        return false;
    }

    return true;
}

GeometricMapping GeometricMapping::operator[] (const IndexList& idx) const
{
    GeometricMapping mapping;
    mapping.src = src[idx];
    mapping.dst = dst[idx];
    mapping.metric = metric ? (*metric)[idx] : metric;

    if (!indices.empty())
    {
        BOOST_FOREACH(size_t i, idx)
        {
            mapping.indices.push_back(indices[i]);
        }
    }

    return mapping;
}

//==[ GeometricTransform ]====================================================//

Points3F& GeometricTransform::operator() (Points3F& pts) const
{
    (*this)(Geometry(Geometry::PACKED, cv::Mat(pts))).mat.copyTo(cv::Mat(pts));
    return pts;
}

Points3D& GeometricTransform::operator() (Points3D& pts) const
{
    (*this)(Geometry(Geometry::PACKED, cv::Mat(pts))).mat.copyTo(cv::Mat(pts));
    return pts;
}

//==[ Rotation ]==============================================================//

const Rotation Rotation::Identity(Rotation::EULER_ANGLES, cv::Mat::eye(3, 3, CV_64F));

cv::Mat Rotation::RotX(double rad)
{
    const double c = std::cos(rad), s = std::sin(rad);
    return (cv::Mat_<double>(3, 3) <<
        1,  0,  0,
        0,  c, -s,
        0,  s,  c);
}

cv::Mat Rotation::RotY(double rad)
{
    const double c = std::cos(rad), s = std::sin(rad);
    return (cv::Mat_<double>(3, 3) <<
        c,  0,  s,
        0,  1,  0,
       -s,  0,  c);
}

cv::Mat Rotation::RotZ(double rad)
{
    const double c = std::cos(rad), s = std::sin(rad);
    return (cv::Mat_<double>(3, 3) <<
        c, -s,  0,
        s,  c,  0,
        0,  0,  1);
}

Rotation::Rotation(Parameterisation param, cv::Mat rmat)
: m_param(param)
{
    if (rmat.rows != 3 || rmat.cols != 3 || rmat.type() != CV_64FC1)
    {
        E_WARNING << "given matrix ignored due to wrong size and/or type";
        m_rmat = Identity.ToMatrix();
    }
    else
    {
        m_rmat = rmat;
    }

    if (!FromMatrix(m_rmat))
    {
        E_WARNING << "error initialising rotation matrix";
    }
}

bool Rotation::FromMatrix(const cv::Mat &rmat)
{
    if (rmat.rows != 3 || rmat.cols != 3)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(rmat.size()), " rather than 3x3";
        return false;
    }

    cv::Mat rvec;
    cv::Rodrigues(rmat, rvec);

    return FromVector(rvec);
}

bool Rotation::FromVector(const cv::Mat& rvec)
{
    if (rvec.total() != 3)
    {
        E_ERROR << "given vector has " << rvec.total() << " element(s) rather than 3";
        return false;
    }

    cv::Mat rmat;
    rvec.reshape(1, 3).convertTo(m_rvec, m_rmat.type());

    cv::Rodrigues(m_rvec, rmat);
    rmat.copyTo(m_rmat);

    return true;
}

bool Rotation::FromAngles(double x, double y, double z)
{
    cv::Mat Rx = RotX(/*x*/ ToRadian(x));
    cv::Mat Ry = RotY(/*y*/ ToRadian(y));
    cv::Mat Rz = RotZ(/*z*/ ToRadian(z));

    //cv::Mat Rx = RotX(x);
    //cv::Mat Ry = RotY(y);
    //cv::Mat Rz = RotZ(z);

    return FromMatrix(Rz * Ry * Rx);
}

void Rotation::ToVector(Vec& rvec) const
{
    rvec.assign((double*)m_rvec.datastart, (double*)m_rvec.dataend);
}

void Rotation::ToAngles(double& x, double& y, double& z) const
{
    double r00 = m_rmat.at<double>(0, 0);
    double r10 = m_rmat.at<double>(1, 0);
    double r11 = m_rmat.at<double>(1, 1);
    double r20 = m_rmat.at<double>(2, 0);
    double r21 = m_rmat.at<double>(2, 1);
    double r22 = m_rmat.at<double>(2, 2);

    double n12 = std::sqrt(r00 * r00 + r11 * r11);

    x = ToDegree(std::atan2(r21, r22));
    y = ToDegree(std::atan2(r20, n12));
    z = ToDegree(std::atan2(r10, r00));

    //x = std::atan2(r21, r22);
    //y = std::atan2(r20, n12);
    //z = std::atan2(r10, r00);
}

Point3F& Rotation::operator() (Point3F& pt) const
{
    const double* m = m_rmat.ptr<double>();

    pt = Point3F(
        (float)(m[0] * pt.x + m[3] * pt.y + m[6] * pt.z),
        (float)(m[1] * pt.x + m[4] * pt.y + m[7] * pt.z),
        (float)(m[2] * pt.x + m[5] * pt.y + m[8] * pt.z)
    );

    return pt;
}

Point3D& Rotation::operator() (Point3D& pt) const
{
    const double* m = m_rmat.ptr<double>();

    pt = Point3D(
        m[0] * pt.x + m[3] * pt.y + m[6] * pt.z,
        m[1] * pt.x + m[4] * pt.y + m[7] * pt.z,
        m[2] * pt.x + m[5] * pt.y + m[8] * pt.z
    );

    return pt;
}

Geometry& Rotation::operator() (Geometry& g) const
{
    const size_t d = g.GetDimension();
    const int    m = g.mat.rows; // for reshaping a packed geometry

    if (d != 3 && d != 4)
    {
        E_ERROR << "geometry matrix must store either Euclidean 3D or homogeneous 4D coordinates (d=" << d << ")";
        return g;
    }

    switch (g.shape)
    {
    case Geometry::ROW_MAJOR:
        g.mat.colRange(0, 3) = g.mat.colRange(0, 3) * m_rmat.t();
        break;
    case Geometry::COL_MAJOR:
        g.mat.rowRange(0, 3) = m_rmat * g.mat.rowRange(0, 3);
        break;
    case Geometry::PACKED:
        g.mat = g.mat.reshape(1, static_cast<int>(g.GetElements()));
        g.mat.colRange(0, 3) = g.mat.colRange(0, 3) * m_rmat.t();
        g.mat = g.mat.reshape(static_cast<int>(d), m);
        break;
    }

    return g;
}

bool Rotation::Store(cv::FileStorage& fs) const
{
    fs << "rvec" << m_rvec;
    return true;
}

bool Rotation::Restore(const cv::FileNode& fn)
{
    cv::Mat rvec;
    fn["rvec"] >> rvec;

    return FromVector(rvec);
}

bool Rotation::Store(Vec& v) const
{
    switch (m_param)
    {
    case Rotation::EULER_ANGLES:
        v.resize(3);
        ToAngles(v[0], v[1], v[2]);
        break;

    case Rotation::RODRIGUES:
        v = m_rvec;
        break;

    default:
        return false;
    }
    return true;
}

bool Rotation::Restore(const Vec& v)
{
    switch (m_param)
    {
    case Rotation::EULER_ANGLES:
        if (v.size() != 3) return false;
        FromAngles(v[0], v[1], v[2]);
        break;

    case Rotation::RODRIGUES:
        FromVector(v);
        break;

    default:
        return false;
    }

    return true;
}

//==[ EuclideanTransform ]====================================================//

const EuclideanTransform EuclideanTransform::Identity(Rotation::Identity.ToMatrix(), cv::Mat::zeros(3, 1, CV_64F));

EuclideanTransform::EuclideanTransform(const cv::Mat& rotation, const cv::Mat& tvec)
: EuclideanTransform()
{
    if (rotation.total() == 3)
    {
        if (!m_rotation.FromVector(rotation))
        {
            E_WARNING << "given rotation is not valid as a angle-axis form";
        }
    }
    else
    {
        if (!m_rotation.FromMatrix(rotation))
        {
            E_WARNING << "given rotation is not valid as a direct-cosine matrix";
        }
    }

    SetTranslation(tvec);
}

EuclideanTransform EuclideanTransform::operator<<(const EuclideanTransform& tform) const
{
    cv::Mat R0 = m_rotation.ToMatrix();
    cv::Mat R1 = tform.m_rotation.ToMatrix();

    return EuclideanTransform(R1 * R0, R1 * m_tvec + tform.m_tvec);
}

bool EuclideanTransform::SetTranslation(const cv::Mat& tvec)
{
    if (tvec.total() != 3)
    {
        E_ERROR << "given vector has " << tvec.total() << " element(s) rather than 3";
        return false;
    }

    cv::Mat _tvec;
    tvec.convertTo(_tvec, m_tvec.type());
    _tvec.reshape(0, 3).copyTo(m_tvec);

    return true;
}

bool EuclideanTransform::SetTranslation(const Vec& tvec)
{
    return false;
}

bool EuclideanTransform::SetTransformMatrix(const cv::Mat& matrix)
{
    if ((matrix.rows != 3 && matrix.rows != 4) || matrix.cols != 4)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(matrix.size()) << " rather than 3x4 or 4x4";
    }

    // TODO: check the fourth row if a square matrix is passed
    // ...
    // ...

    const cv::Mat rmat = matrix.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tvec = matrix.rowRange(0, 3).colRange(3, 4);

    return m_rotation.FromMatrix(rmat) && SetTranslation(tvec);
}

cv::Mat EuclideanTransform::GetTransformMatrix(bool sqrMat, bool preMult, int type) const
{
    if (!sqrMat && preMult) return m_matrix.clone(); // this is the intrinsic form

    cv::Mat matrix = cv::Mat::eye(sqrMat ? 4 : 3, 4, m_matrix.type());
    m_matrix.copyTo(matrix.rowRange(0, 3).colRange(0, 4));

    if (matrix.type() != type)
    {
        matrix.convertTo(matrix, type);
    }

    return preMult ? matrix : matrix.t();
}

cv::Mat EuclideanTransform::ToEssentialMatrix() const
{
    return skewsymat(m_tvec) * m_rotation.ToMatrix();
}

bool EuclideanTransform::FromEssentialMatrix(const cv::Mat& E, const GeometricMapping& m)
{
    cv::Mat rvec, tvec;
    int inliers;
    
    try
    {
        inliers = cv::recoverPose(E, m.src.mat, m.dst.mat, rvec, tvec);
    }
    catch (std::exception& ex)
    {
        E_ERROR << "cv::recoverPose failed : " << ex.what();
        return false;
    }

    return m_rotation.FromMatrix(rvec) && SetTranslation(tvec);
}

EuclideanTransform EuclideanTransform::GetInverse() const
{
    cv::Mat Rt = m_rotation.ToMatrix().t();
    return EuclideanTransform(Rt, -Rt*m_tvec);
}

Point3F& EuclideanTransform::operator() (Point3F& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3F(
        (float)(m[0] * pt.x + m[1] * pt.y + m[2]  * pt.z + m[3]),
        (float)(m[4] * pt.x + m[5] * pt.y + m[6]  * pt.z + m[7]),
        (float)(m[8] * pt.x + m[9] * pt.y + m[10] * pt.z + m[11])
    );

    return pt;
}

Point3D& EuclideanTransform::operator() (Point3D& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3D(
        m[0] * pt.x + m[1] * pt.y + m[2]  * pt.z + m[3],
        m[4] * pt.x + m[5] * pt.y + m[6]  * pt.z + m[7],
        m[8] * pt.x + m[9] * pt.y + m[10] * pt.z + m[11]
    );

    return pt;
}

Geometry& EuclideanTransform::operator() (Geometry& g, bool euclidean) const
{
    const size_t d = g.GetDimension();
    const int    m = g.mat.rows; // for reshaping a packed geometry

    if (d != 3 && d != 4)
    {
        E_ERROR << "geometry matrix must store either Euclidean 3D or homogeneous 4D coordinates (d=" << d << ")";
        return g;
    }

    if (d == 3)
    {
        g = Geometry::MakeHomogeneous(g);
    }

    switch (g.shape)
    {
    case Geometry::ROW_MAJOR:
        g.mat = g.mat * GetTransformMatrix(!euclidean, false, g.mat.type());
        break;
    case Geometry::COL_MAJOR:
        g.mat = GetTransformMatrix(!euclidean, true, g.mat.type()) * g.mat;
        break;
    case Geometry::PACKED:
        g.mat = g.mat.reshape(1, static_cast<int>(g.GetElements()));
        g.mat = g.mat * GetTransformMatrix(!euclidean, false, g.mat.type());
        g.mat = g.mat.reshape(euclidean ? 3 : 4, m);
        break;
    }

    return g;
}

bool EuclideanTransform::Store(cv::FileStorage& fs) const
{
    fs << "tvec" << m_tvec;
    return m_rotation.Store(fs);
}

bool EuclideanTransform::Restore(const cv::FileNode& fn)
{
    cv::Mat tvec;
    fn["tvec"] >> tvec;

    return m_rotation.Restore(fn) && SetTranslation(tvec);
}

bool EuclideanTransform::Store(VectorisableD::Vec& v) const
{
    Vec rvec, tvec(3);

    if (!m_rotation.Store(rvec))
    {
        return false;
    }

    for (int j = 0; j < 3; j++)
    {
        tvec[j] = m_tvec.at<double>(j);
    }

    v.clear();
    v.insert(v.end(), rvec.begin(), rvec.end());
    v.insert(v.end(), tvec.begin(), tvec.end());

    return true;
}

bool EuclideanTransform::Restore(const Vec& v)
{
    if (v.size() != GetDimension()) return false;

    Vec rvec(v.begin(), v.begin() + m_rotation.GetDimension());

    if (!m_rotation.Restore(rvec))
    {
        return false;
    }

    cv::Mat tvec(3, 1, CV_64F);
    size_t i = m_rotation.GetDimension();

    for (int j = 0; j < 3; j++)
    {
        tvec.at<double>(j) = v[i++];
    }

    return SetTranslation(tvec);
}

//==[ Motion ]================================================================//

size_t Motion::Update(const EuclideanTransform& tform)
{
    if (m_tforms.empty())
    {
        m_tforms.push_back(tform);
    }
    else
    {
        m_tforms.push_back(m_tforms.back() << tform);
    }

    return m_tforms.size();
}

bool Motion::Store(Path& path) const
{
    std::ofstream of(path.string().c_str(), std::ios::out);
    BOOST_FOREACH (const EuclideanTransform& tform, m_tforms)
    {
        cv::Mat M = tform.GetTransformMatrix();
        double* m = M.ptr<double>();

        for (size_t i = 0; i < 12; i++) of << m[i] << (i < 11 ? " " : "");
        of << std::endl;
    }

    return true;
}

bool Motion::Restore(const Path& path)
{
    m_tforms.clear();

    std::ifstream rf(path.string().c_str(), std::ios::in);

    if (!rf.is_open())
    {
        E_ERROR << "error reading " << path;
        return false;
    }

    for (String line; std::getline(rf, line); /**/)
    {
        std::stringstream ss(line);
        std::vector<double> m(
            (std::istream_iterator<double>(ss)), // begin
            (std::istream_iterator<double>()));  // end

        if (m.size() != 12)
        {
            E_ERROR << "error parsing " << path;
            E_ERROR << "line " << (m_tforms.size() + 1) << ": " << line;

            return false;
        }

        EuclideanTransform tform;
        tform.SetTransformMatrix(cv::Mat(m).reshape(1, 3));

        m_tforms.push_back(tform);
    }

    return true;
}

//==[ ProjectionModel ]=======================================================//

cv::Mat ProjectionModel::Project(const Geometry& g, const cv::Mat& im) const
{
    throw std::exception("not implemented");
}

Geometry ProjectionModel::GetJacobian(const Geometry& g) const
{
    return GetJacobian(
        g,
        Project(g, ProjectionModel::EUCLIDEAN_2D).Reshape(Geometry::ROW_MAJOR)
    );
}

Geometry ProjectionModel::GetJacobian(const Geometry& g, const Geometry& proj) const
{
    if (proj.shape != Geometry::ROW_MAJOR)
    {
        return GetJacobian(g, proj.Reshape(Geometry::ROW_MAJOR));
    }

    const double dx = 1e-2;

    Geometry jac(Geometry::ROW_MAJOR);
    jac.mat = cv::Mat(static_cast<int>(g.GetElements()), 6, g.mat.type());

    for (int i = 0; i < 3; i++)
    {
        const cv::Mat yi = Project(g.Step(i, dx).Reshape(Geometry::ROW_MAJOR), ProjectionModel::EUCLIDEAN_2D).mat;
        const int j = i * yi.cols;

        jac.mat.colRange(j, j + yi.cols) = (yi - proj.mat) / dx;
    }

    return g.shape == Geometry::PACKED ?
        Geometry(Geometry::PACKED, jac.mat.reshape(6, g.mat.rows)) :
        jac.Reshape(g.shape);
}

//==[ PinholeModel ]==========================================================//

bool seq2map::PinholeModel::operator== (const PinholeModel& rhs) const
{
    Point2D f0, c0;
    Point2D f1, c1;

    const PinholeModel& lhs = *this;

    lhs.GetValues(f0.x, f0.y, c0.x, c0.y);
    rhs.GetValues(f1.x, f1.y, c1.x, c1.y);

    return f0 == f1 && c0 == c1;
}

Point3F& PinholeModel::operator() (Point3F& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3F(
        (float)(m[0] * pt.x / pt.z + m[2]),
        (float)(m[4] * pt.y / pt.z + m[5]),
        pt.z != 0 ? pt.z : 0
    );

    return pt;
}

Point3D& PinholeModel::operator() (Point3D& pt) const
{
    const double* m = m_matrix.ptr<double>();

    pt = Point3D(
        m[0] * pt.x / pt.z + m[2],
        m[4] * pt.y / pt.z + m[5],
        pt.z != 0 ? pt.z : 0
    );

    return pt;
}

Geometry PinholeModel::Project(const Geometry& g, ProjectiveSpace space) const
{
    Geometry proj(g.shape);

    if (g.GetDimension() != 3)
    {
        E_ERROR << "given geometry has to be 3D Euclidean (d=" << g.GetDimension() << ")";
        return proj;
    }

    cv::Mat K = m_matrix;

    if (K.type() != g.mat.type())
    {
        K.convertTo(K, g.mat.type());
    }

    // compute projection
    switch (g.shape)
    {
    case Geometry::ROW_MAJOR:
        proj.mat = g.mat * K.t();
        break;

    case Geometry::COL_MAJOR:
        proj.mat = K * g.mat;
        break;

    case Geometry::PACKED:
        proj.mat = cv::Mat(g.mat.reshape(1, static_cast<int>(g.GetElements())) * K.t()).reshape(3, g.mat.rows);
        break;
    }

    // de-homogenisation
    switch (space)
    {
    case EUCLIDEAN_3D:
        // no need to do normalisation
        break;

    case EUCLIDEAN_2D:
        // dehomonise and remove the last dimension
        proj = Geometry::FromHomogeneous(proj);
        break;

    case HOMOGENEOUS_3D:
        // economy in-place dehomonisation
        proj.Dehomogenise();
        break;
    }

    return proj;
}

Geometry PinholeModel::Backproject(const Geometry& g) const
{
    const size_t d = g.GetDimension();

    if (d != 2 && d != 3)
    {
        E_ERROR << "geometry matrix must store either Euclidean 2D or homogeneous 3D coordinates (d=" << d << ")";
        return Geometry(g.shape);
    }

    // the forward projection of the inverse camera would do the work
    return PinholeModel(GetInverseCameraMatrix()).Project(
        d == 2 ? Geometry::MakeHomogeneous(g) : g,
        EUCLIDEAN_3D
    );
}

Geometry PinholeModel::GetJacobian(const Geometry& g) const
{
    if (g.shape != Geometry::ROW_MAJOR)
    {
        return GetJacobian(g.Reshape(Geometry::ROW_MAJOR));
    }

    Geometry jac(Geometry::ROW_MAJOR);
    jac.mat = cv::Mat(static_cast<int>(g.GetElements()), 6, g.mat.type());

    const double fx = m_matrix.at<double>(0, 0);
    const double fy = m_matrix.at<double>(1, 1);
#if 1
    const cv::Mat w = 1 / g.mat.col(2);
    const cv::Mat u = g.mat.col(0).mul(w);
    const cv::Mat v = g.mat.col(1).mul(w);

    cv::multiply(fx, w, jac.mat.col(0));     // dx/dX
    jac.mat.col(1).setTo(0.0f);              // dy/dX
    jac.mat.col(2).setTo(0.0f);              // dx/dY
    cv::multiply(fy, w, jac.mat.col(3));     // dy/dY
    cv::multiply( u, w, jac.mat.col(4), -1); // dx/dZ
    cv::multiply( v, w, jac.mat.col(5), -1); // dx/dZ
#else
    cv::cuda::GpuMat gmat, w, u, v, dxx, dyy, dxz, dyz;

    gmat.upload(g.mat);

    cv::cuda::divide(1, gmat.col(2), w);
    cv::cuda::multiply(g.mat.col(0), w, u);
    cv::cuda::multiply(g.mat.col(1), w, v);
    cv::cuda::multiply(fx, w, dxx);
    cv::cuda::multiply(fy, w, dyy);
    cv::cuda::multiply(u, w, dxz, -1);
    cv::cuda::multiply(v, w, dyz, -1);

    dxx.download(jac.mat.col(0));
    jac.mat.col(1).setTo(0.0f);
    jac.mat.col(2).setTo(0.0f);
    dyy.download(jac.mat.col(3));
    dxz.download(jac.mat.col(4));
    dyz.download(jac.mat.col(5));
#endif
    return jac;
}

cv::Mat PinholeModel::ToProjectionMatrix(const EuclideanTransform& tform) const
{
    cv::Mat P = m_matrix * tform.GetTransformMatrix(true, true, m_matrix.type());
    return P;
}

bool PinholeModel::FromProjectionMatrix(const cv::Mat& P, const EuclideanTransform& tform)
{
    cv::Mat K = P * tform.GetInverse().GetTransformMatrix(true, true, P.type());
    return SetCameraMatrix(P);
}

bool PinholeModel::SetCameraMatrix(const cv::Mat& K)
{
    if (K.rows != 3 || K.cols != 3 || K.channels() != 1)
    {
        E_ERROR << "given matrix has wrong size of " << size2string(K.size()) << " rather than 3x3x1";
        return false;
    }

    cv::Mat K64f;
    K.convertTo(K64f, CV_64F);

    if (K64f.at<double>(0, 1) != 0 ||
        K64f.at<double>(1, 0) != 0 ||
        K64f.at<double>(2, 0) != 0 ||
        K64f.at<double>(2, 1) != 0 ||
        K64f.at<double>(2, 2) != 1)
    {
        E_ERROR << "the matrix has to follow the form:";
        E_ERROR << "| fx,  0, cx |";
        E_ERROR << "|  0, fy, cy |";
        E_ERROR << "|  0,  0,  1 |";

        return false;
    }

    m_matrix = K64f;
    return true;
}

cv::Mat PinholeModel::GetInverseCameraMatrix() const
{
    double fx, fy, cx, cy;
    GetValues(fx, fy, cx, cy);

    double k00 = 1.0f / fx, k02 = -cx / fx;
    double k11 = 1.0f / fy, k12 = -cy / fy;

    return (cv::Mat_<double>(3, 3) <<
        k00, 0,   k02,
        0,   k11, k12,
        0,   0,   1);
}

void PinholeModel::SetValues(double fx, double fy, double cx, double cy)
{
    m_matrix.at<double>(0, 0) = fx;
    m_matrix.at<double>(1, 1) = fy;
    m_matrix.at<double>(0, 2) = cx;
    m_matrix.at<double>(1, 2) = cy;
}

void PinholeModel::GetValues(double& fx, double& fy, double& cx, double& cy) const
{
    fx = m_matrix.at<double>(0, 0);
    fy = m_matrix.at<double>(1, 1);
    cx = m_matrix.at<double>(0, 2);
    cy = m_matrix.at<double>(1, 2);
}

bool PinholeModel::Store(cv::FileStorage& fs) const
{
    Point2D f, c;

    GetValues(f.x, f.y, c.x, c.y);

    fs << "focalLengths"   << f;
    fs << "principalPoint" << c;

    return true;
}

bool PinholeModel::Restore(const cv::FileNode& fn)
{
    Point2D f, c;

    fn["focalLengths"]   >> f;
    fn["principalPoint"] >> c;

    SetValues(f.x, f.y, c.x, c.y);

    return true;
}

bool PinholeModel::Store(Vec& v) const
{
    v.resize(4);
    GetValues(v[0], v[1], v[2], v[3]);

    return true;
}

bool PinholeModel::Restore(const Vec& v)
{
    if (v.size() != GetDimension()) return false;

    SetValues(v[0], v[1], v[2], v[3]);
    return true;
}

//==[ BouguetModel ]==========================================================//

BouguetModel::BouguetModel(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
    if (!SetCameraMatrix(cameraMatrix))   E_ERROR << "error setting camera matrix";
    if (!SetDistortionCoeffs(distCoeffs)) E_ERROR << "error setting distortion coefficients";
}

Point3F& BouguetModel::operator() (Point3F& pt) const
{
    throw std::exception("not implemented");
    return pt;
}

Point3D& BouguetModel::operator() (Point3D& pt) const
{
    throw std::exception("not implemented");
    return pt;
}

Geometry BouguetModel::Project(const Geometry& g, ProjectiveSpace space) const
{
    Geometry proj(g.shape);

    if (g.GetDimension() != 3)
    {
        E_ERROR << "given geometry has to be 3D Euclidean (d=" << g.GetDimension() << ")";
        return proj;
    }

    cv::projectPoints(g.shape == Geometry::PACKED ? g.mat.reshape(1, static_cast<int>(g.GetElements())) : g.mat,
        EuclideanTransform::Identity.GetRotation().ToMatrix(),
        EuclideanTransform::Identity.GetTranslation(),
        m_matrix, m_distCoeffs, proj.mat);

    if (g.shape == Geometry::PACKED)
    {
        proj.mat = proj.mat.reshape(3, g.mat.rows); // restore to a MxNx3 matrix
    }

    switch (space)
    {
    case HOMOGENEOUS_3D:
    case EUCLIDEAN_3D:
        return Geometry::MakeHomogeneous(proj);
    }

    return proj;
}

Geometry BouguetModel::Backproject(const Geometry& g) const
{
    throw std::exception("not implemented");
    return Geometry(g.shape);
}

bool BouguetModel::SetDistortionCoeffs(const cv::Mat& D)
{
    cv::Mat D64f = D;
    Vec d;

    if (D.type() != CV_64F)
    {
        D.convertTo(D64f, CV_64F);
    }

    d.assign((double*) D64f.datastart, (double*) D64f.dataend);

    double fx, fy, cx, cy;
    GetValues(fx, fy, cx, cy);

    switch (d.size())
    {
    case RADIAL_TANGENTIAL_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3]);
        break;
    case HIGH_RADIAL_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4]);
        break;
    case RATIONAL_RADIAL_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        break;
    case THIN_PSISM_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11]);
        break;
    case TILTED_SENSOR_DISTORTION:
        SetValues(fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13]);
        break;
    default:
        E_ERROR << "given matrix has " << d.size() << " element(s) instead of 4, 5, 8, 12 or 14";
        return false;
    }

    return true;
}

cv::Mat BouguetModel::GetDistortionCoeffs() const
{
    int dof = static_cast<int>(m_distModel);
    return cv::Mat(m_distCoeffs).rowRange(0, dof).clone();
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2)
{
    SetValues(fx, fy, cx, cy);
    SetDistortionModel(RADIAL_TANGENTIAL_DISTORTION);

    m_distCoeffs[0] = k1;
    m_distCoeffs[1] = k2;
    m_distCoeffs[2] = p1;
    m_distCoeffs[3] = p2;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2);
    SetDistortionModel(HIGH_RADIAL_DISTORTION);

    m_distCoeffs[4] = k3;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3);
    SetDistortionModel(RATIONAL_RADIAL_DISTORTION);

    m_distCoeffs[5] = k4;
    m_distCoeffs[6] = k5;
    m_distCoeffs[7] = k6;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6, double s1, double s2, double s3, double s4)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6);
    SetDistortionModel(THIN_PSISM_DISTORTION);

    m_distCoeffs[8]  = s1;
    m_distCoeffs[9]  = s2;
    m_distCoeffs[10] = s3;
    m_distCoeffs[11] = s4;
}

void BouguetModel::SetValues(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double k4, double k5, double k6, double s1, double s2, double s3, double s4, double taux, double tauy)
{
    SetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4);
    SetDistortionModel(TILTED_SENSOR_DISTORTION);

    m_distCoeffs[12] = taux;
    m_distCoeffs[13] = tauy;
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2) const
{
    GetValues(fx, fy, cx, cy);

    k1 = m_distCoeffs[0];
    k2 = m_distCoeffs[1];
    p1 = m_distCoeffs[2];
    p2 = m_distCoeffs[3];
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2);

    if (m_distModel < HIGH_RADIAL_DISTORTION)
    {
        k3 = 0.0f;
    }
    else
    {
        k3 = m_distCoeffs[4];
    }
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3);

    if (m_distModel < RATIONAL_RADIAL_DISTORTION)
    {
        k4 = k5 = k6 = 0.0f;
    }
    else
    {
        k4 = m_distCoeffs[5];
        k5 = m_distCoeffs[6];
        k6 = m_distCoeffs[7];
    }
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6, double& s1, double& s2, double& s3, double& s4) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6);

    if (m_distModel < THIN_PSISM_DISTORTION)
    {
        s1 = s2 = s3 = s4 = 0.0f;
    }
    else
    {
        s1 = m_distCoeffs[8];
        s2 = m_distCoeffs[9];
        s3 = m_distCoeffs[10];
        s4 = m_distCoeffs[11];
    }
}

void BouguetModel::GetValues(double& fx, double& fy, double& cx, double& cy, double& k1, double& k2, double& p1, double& p2, double& k3, double& k4, double& k5, double& k6, double& s1, double& s2, double& s3, double& s4, double& taux, double& tauy) const
{
    GetValues(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4);

    if (m_distModel < TILTED_SENSOR_DISTORTION)
    {
        taux = tauy = 0.0f;
    }
    else
    {
        taux = m_distCoeffs[12];
        tauy = m_distCoeffs[13];
    }
}

bool BouguetModel::Store(cv::FileStorage & fs) const
{
    if (!PinholeModel::Store(fs)) return false;
    fs << "distCoeffs" << GetDistortionCoeffs();

    return true;
}

bool BouguetModel::Restore(const cv::FileNode & fn)
{
    if (!PinholeModel::Restore(fn)) return false;

    cv::Mat distCoeffs;
    fn["distCoeffs"]>> distCoeffs;

    return SetDistortionCoeffs(distCoeffs);
}

bool BouguetModel::Store(Vec& v) const
{
    throw std::exception("not implemented");
    return false;
}

bool BouguetModel::Restore(const Vec& v)
{
    throw std::exception("not implemented");
    return false;
}

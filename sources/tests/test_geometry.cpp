#define BOOST_TEST_MODULE "Geometry"
#include <boost/test/unit_test.hpp>
#include <seq2map/geometry.hpp>

using namespace seq2map;

BOOST_AUTO_TEST_CASE(constructors)
{
    cv::Mat mat = (cv::Mat_<int>(3, 4) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    Geometry g0(Geometry::ROW_MAJOR, mat);
    Geometry g1(g0);
    Geometry g2(static_cast<const Geometry&>(g0));

    BOOST_CHECK(g0.mat.rows == g1.mat.rows && g0.mat.cols == g1.mat.cols && g0.mat.type() == g1.mat.type());
    BOOST_CHECK(g0.mat.rows == g2.mat.rows && g0.mat.cols == g2.mat.cols && g0.mat.type() == g2.mat.type());

    for (int i = 0; i < g0.mat.rows; i++)
    {
        g1.mat.row(i) = g1.mat.row((i + 1) % g0.mat.rows) - g1.mat.row(i);
        g2.mat.row(i) = g2.mat.row(i) - g2.mat.row((i + 1) % g0.mat.rows);
    }

    for (int i = 0; i < g0.mat.rows; i++)
    {
        for (int j = 0; j < g0.mat.cols; j++)
        {
            BOOST_CHECK(g0.mat.at<int>(i, j) == g1.mat.at<int>(i, j));
            BOOST_CHECK(g0.mat.at<int>(i, j) != g2.mat.at<int>(i, j));
        }
    }
}

BOOST_AUTO_TEST_CASE(reshape)
{
    cv::Mat mat = (cv::Mat_<int>(4, 3) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

    const Geometry g0(Geometry::ROW_MAJOR, mat);
    Geometry g1 = g0.Reshape(Geometry::ROW_MAJOR);
    Geometry g2 = g0.Reshape(Geometry::COL_MAJOR);
    Geometry g3 = g2.Reshape(Geometry::PACKED);

    BOOST_CHECK(g0.GetDimension() == g1.GetDimension() && g0.GetElements() == g1.GetElements());
    BOOST_CHECK(g0.GetDimension() == g2.GetDimension() && g0.GetElements() == g2.GetElements());

    for (int i = 0; i < g0.mat.rows; i++)
    {
        for (int j = 0; j < g0.mat.cols; j++)
        {
            const int& g_ij = g0.mat.at<int>(i, j);

            BOOST_CHECK(g_ij == g1.mat.at<int>(i, j));
            BOOST_CHECK(g_ij == g2.mat.at<int>(j, i));
            BOOST_CHECK(g_ij == g3.mat.at<cv::Vec3i>(i, 0)[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(rotation)
{
    Rotation R0, R1, R2;

    BOOST_CHECK(R0.FromAngles( 45, 0, 0));
    BOOST_CHECK(R1.FromAngles(-45, 0, 0));

    BOOST_CHECK(R2.FromMatrix(R1.ToMatrix() * R0.ToMatrix()));
    BOOST_CHECK(R2.IsIdentity());

    BOOST_CHECK(R2.FromMatrix(R0.ToMatrix().t() * R0.ToMatrix()));
    BOOST_CHECK(R2.IsIdentity());

    Rotation::Vec v;
    BOOST_CHECK(R0.Store(v));
    BOOST_CHECK(R1.Restore(v));
    BOOST_CHECK(R0 == R1);
}

BOOST_AUTO_TEST_CASE(mahalanobis)
{
    const double EPSILON = 1e-10;

    const cv::Mat gmat0 = (cv::Mat_<double>(4, 3) << 68, 76, 74, 39, 66, 17, 71,  3, 28,  5, 10, 82);
    const cv::Mat gmat1 = (cv::Mat_<double>(4, 3) << 69, 32, 95,  3, 44, 38, 77, 80, 19, 49, 45, 65);
    const cv::Mat cvar0 = (cv::Mat_<double>(4, 1) << 1, 3, 5, 7);
    const cv::Mat cvar1 = (cv::Mat_<double>(4, 3) << 1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7);
    const cv::Mat cvar2 = (cv::Mat_<double>(4, 6) << 1, 0, 0, 1, 0, 1, 3, 0, 0, 3, 0, 3, 5, 0, 0, 5, 0, 5, 7, 0, 0, 7, 0, 7);
    const cv::Mat cvar3 = (cv::Mat_<double>(4, 6) << 47, 42, 39, 98, 66, 70, 48, 57, 51, 74, 66, 61, 56, 48, 55, 48, 48, 56, 33, 25, 46, 20, 35, 66);
    const cv::Mat tfrm0 = (cv::Mat_<double>(4, 6) << 40,  8, 24, 12, 18, 24, 21,  3, 45, 47, 25, 25, 17, 45, 19,  6, 39, 20, 19, 32,  8, 10, 75, 77);
    const cv::Mat cvar4 = (cv::Mat_<double>(4, 3) << 348152, 197056, 118496, 379092, 209904, 117408, 348224, 186928, 100736, 230184, 140136,  85632);
    const cv::Mat cvar5 = (cv::Mat_<double>(4, 3) << 348152, 197056, 118496, 531757, 476305, 433449, 332089, 264150, 235423, 620107, 700211, 793670);
    const cv::Mat cvar6 = (cv::Mat_<double>(4, 3) << 348152, 197056, 118496, 518923, 444013, 380847, 293762, 282625, 273968, 565143, 639472, 723694);
    const cv::Mat tfrm1 = (cv::Mat_<double>(1, 9) << 2, 4, 6, 8, 1, 3, 5, 7, 9);

    Geometry g0(Geometry::ROW_MAJOR, gmat0);
    Geometry g1(Geometry::ROW_MAJOR, gmat1);

    MahalanobisMetric m0(MahalanobisMetric::ISOTROPIC,              3);
    MahalanobisMetric m1(MahalanobisMetric::ANISOTROPIC_ORTHOGONAL, 3);
    MahalanobisMetric m2(MahalanobisMetric::ANISOTROPIC_ROTATED,    3);

    // test covariance matrix setter
    E_INFO << "expect to see 6 warnings below..";
    BOOST_CHECK( m0.SetCovarianceMat(cvar0));
    BOOST_CHECK(!m0.SetCovarianceMat(cvar1));
    BOOST_CHECK(!m0.SetCovarianceMat(cvar2));
    BOOST_CHECK(!m1.SetCovarianceMat(cvar0));
    BOOST_CHECK( m1.SetCovarianceMat(cvar1));
    BOOST_CHECK(!m1.SetCovarianceMat(cvar2));
    BOOST_CHECK(!m2.SetCovarianceMat(cvar0));
    BOOST_CHECK(!m2.SetCovarianceMat(cvar1));
    BOOST_CHECK( m2.SetCovarianceMat(cvar2));
    E_INFO << "expect to see 6 warnings above..";

    // test distance measuring
    Geometry d0 = ((Metric&)m0)(g0, g1);
    Geometry d1 = ((Metric&)m1)(g0, g1);
    Geometry d2 = ((Metric&)m2)(g0, g1);

    BOOST_REQUIRE_SMALL(cv::norm(d0.mat, d1.mat), EPSILON);
    BOOST_REQUIRE_SMALL(cv::norm(d1.mat, d2.mat), EPSILON);

    // test transform
    EuclideanTransform T;
    T.GetRotation().FromAngles(30, 45, 60);

    Metric::Own m3 = m2.Transform(T);
    BOOST_CHECK(m3);

    Geometry d3 = ((*m3)(T(g0), T(g1)));
    BOOST_REQUIRE_SMALL(cv::norm(d0.mat, d3.mat), EPSILON);

    // test filtering
    MahalanobisMetric& m4 = *static_cast<MahalanobisMetric*>(m3.get());
    
    const cv::Mat kal = m4.Update(m4);
    const cv::Mat cov = m4.GetFullCovMat();
    
    // the Kalman gain of self-updating has to be I/2
    for (int i = 0; i < kal.rows; i++)
    {
        const int DIMS = static_cast<int>(m4.dims);
        const cv::Mat K = kal.row(i).reshape(1, DIMS);

        for (int d0 = 0; d0 < DIMS; d0++)
        {
            for (int d1 = d0; d1 < DIMS; d1++)
            {
                const double Kij = K.at<double>(d0, d1);

                if (d0 == d1) BOOST_REQUIRE_CLOSE(Kij, 0.5f, EPSILON);
                else          BOOST_REQUIRE_SMALL(Kij, EPSILON);
            }
        }
    }

    // test transformation
    MahalanobisMetric m5(MahalanobisMetric::ANISOTROPIC_ROTATED, 3);
    MahalanobisMetric m6(MahalanobisMetric::ANISOTROPIC_ROTATED, 3);
    Geometry jac0(Geometry::ROW_MAJOR, tfrm0.row(0));
    Geometry jac1(Geometry::ROW_MAJOR, tfrm0);
    Geometry jac2(Geometry::ROW_MAJOR, tfrm1);

    BOOST_CHECK(m5.SetCovarianceMat(cvar3.row(0)));
    BOOST_CHECK(m6.SetCovarianceMat(cvar3));

    // many-via-one transformation
    Metric::Own m7 = m6.Transform(EuclideanTransform::Identity, jac0);

    // one-via-many transformation
    Metric::Own m8 = m5.Transform(EuclideanTransform::Identity, jac1);

    // many-via-many transformation
    Metric::Own m9 = m6.Transform(EuclideanTransform::Identity, jac1);

    BOOST_CHECK(cv::norm(static_cast<MahalanobisMetric*>(m7.get())->GetFullCovMat(), cvar4) == 0);
    BOOST_CHECK(cv::norm(static_cast<MahalanobisMetric*>(m8.get())->GetFullCovMat(), cvar5) == 0);
    BOOST_CHECK(cv::norm(static_cast<MahalanobisMetric*>(m9.get())->GetFullCovMat(), cvar6) == 0);

    // rotation
    Metric::Own m10 = m6.Transform(T)->Transform(T.GetInverse());
    BOOST_REQUIRE_SMALL(cv::norm(static_cast<MahalanobisMetric*>(m10.get())->GetFullCovMat(), cvar3), EPSILON);

    // rotation + Jacobian
    Metric::Own m11 = m6.Transform(EuclideanTransform::Identity, jac2);
    Metric::Own m12 = m6.Transform(T)->Transform(T.GetInverse(), jac2);
    BOOST_REQUIRE_SMALL(cv::norm(
        static_cast<MahalanobisMetric*>(m11.get())->GetFullCovMat(), 
        static_cast<MahalanobisMetric*>(m12.get())->GetFullCovMat()
    ), EPSILON);
}

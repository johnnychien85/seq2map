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
    Geometry g1 = g0.Reshape(Geometry::COL_MAJOR);
    Geometry g2 = g0.Reshape(Geometry::PACKED);
    Geometry g3 = g2.Reshape(Geometry::ROW_MAJOR);

    BOOST_CHECK(g0.GetDimension() == g1.GetDimension() && g0.GetElements() == g1.GetElements());
    BOOST_CHECK(g0.GetDimension() == g2.GetDimension() && g0.GetElements() == g2.GetElements());

    for (int i = 0; i < g0.mat.rows; i++)
    {
        for (int j = 0; j < g0.mat.cols; j++)
        {
            BOOST_CHECK(g0.mat.at<int>(i, j) == g1.mat.at<int>(j, i));
            BOOST_CHECK(g0.mat.at<int>(i, j) == g2.mat.at<cv::Vec3i>(i, 0)[j]);
            BOOST_CHECK(g0.mat.at<int>(i, j) == g3.mat.at<int>(i, j));
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
    cv::Mat gmat0 = (cv::Mat_<double>(4, 3) << 68, 76, 74, 39, 66, 17, 71,  3, 28,  5, 10, 82);
    cv::Mat gmat1 = (cv::Mat_<double>(4, 3) << 69, 32, 95,  3, 44, 38, 77, 80, 19, 49, 45, 65);
    cv::Mat cvar0 = (cv::Mat_<double>(4, 1) << 1, 3, 5, 7);
    cv::Mat cvar1 = (cv::Mat_<double>(4, 3) << 1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7);
    cv::Mat cvar2 = (cv::Mat_<double>(4, 6) << 1, 0, 0, 1, 0, 1, 3, 0, 0, 3, 0, 3, 5, 0, 0, 5, 0, 5, 7, 0, 0, 7, 0, 7);

    Geometry g0(Geometry::ROW_MAJOR, gmat0);
    Geometry g1(Geometry::ROW_MAJOR, gmat1);

    MahalanobisMetric m0(MahalanobisMetric::ISOTROPIC,              3);
    MahalanobisMetric m1(MahalanobisMetric::ANISOTROPIC_ORTHOGONAL, 3);
    MahalanobisMetric m2(MahalanobisMetric::ANISOTROPIC_ROTATED,    3);

    BOOST_CHECK( m0.SetCovarianceMat(cvar0));
    BOOST_CHECK(!m0.SetCovarianceMat(cvar1));
    BOOST_CHECK(!m0.SetCovarianceMat(cvar2));
    BOOST_CHECK(!m1.SetCovarianceMat(cvar0));
    BOOST_CHECK( m1.SetCovarianceMat(cvar1));
    BOOST_CHECK(!m1.SetCovarianceMat(cvar2));
    BOOST_CHECK(!m2.SetCovarianceMat(cvar0));
    BOOST_CHECK(!m2.SetCovarianceMat(cvar1));
    BOOST_CHECK( m2.SetCovarianceMat(cvar2));

    Geometry d0 = m0(g0, g1);
    Geometry d1 = m1(g0, g1);
    Geometry d2 = m2(g0, g1);

    BOOST_REQUIRE_SMALL(cv::norm(d0.mat, d1.mat), 1e-10);
    BOOST_REQUIRE_SMALL(cv::norm(d1.mat, d2.mat), 1e-10);

    EuclideanTransform T;
    T.GetRotation().FromAngles(30, 45, 60);

    Metric::Own m3 = m2.Transform(T);
    BOOST_CHECK(m3);

    Geometry d3 = T.GetInverse()(((*m3)(T(g0), T(g1))));
    BOOST_REQUIRE_SMALL(cv::norm(d0.mat, d3.mat), 1e-10);
}

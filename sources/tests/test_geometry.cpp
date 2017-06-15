#define BOOST_TEST_MODULE "Geometry"
#include <boost/test/unit_test.hpp>
#include <seq2map/geometry.hpp>

using namespace seq2map;

BOOST_AUTO_TEST_CASE(constructors)
{
    cv::Mat mat = (cv::Mat_<int>(3, 4) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    Geometry g(Geometry::ROW_MAJOR, mat);
    Geometry g1(g);
    Geometry g2(static_cast<const Geometry&>(g));

    BOOST_CHECK(g.mat.rows == g1.mat.rows && g.mat.cols == g1.mat.cols && g.mat.type() == g1.mat.type());
    BOOST_CHECK(g.mat.rows == g2.mat.rows && g.mat.cols == g2.mat.cols && g.mat.type() == g2.mat.type());

    for (int i = 0; i < g.mat.rows; i++)
    {
        g1.mat.row(i) = g1.mat.row((i + 1) % g.mat.rows) - g1.mat.row(i);
        g2.mat.row(i) = g2.mat.row(i) - g2.mat.row((i + 1) % g.mat.rows);
    }

    for (int i = 0; i < g.mat.rows; i++)
    {
        for (int j = 0; j < g.mat.cols; j++)
        {
            BOOST_CHECK(g.mat.at<int>(i, j) == g1.mat.at<int>(i, j));
            BOOST_CHECK(g.mat.at<int>(i, j) != g2.mat.at<int>(i, j));
        }
    }
}

BOOST_AUTO_TEST_CASE(reshape)
{
    cv::Mat mat = (cv::Mat_<int>(4, 3) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    const Geometry g(Geometry::ROW_MAJOR, mat);
    Geometry g1 = g.Reshape(Geometry::COL_MAJOR);
    Geometry g2 = g.Reshape(Geometry::PACKED);
    Geometry g3 = g2.Reshape(Geometry::ROW_MAJOR);

    BOOST_CHECK(g.GetDimension() == g1.GetDimension() && g.GetElements() == g1.GetElements());
    BOOST_CHECK(g.GetDimension() == g2.GetDimension() && g.GetElements() == g2.GetElements());

    for (int i = 0; i < g.mat.rows; i++)
    {
        for (int j = 0; j < g.mat.cols; j++)
        {
            BOOST_CHECK(g.mat.at<int>(i, j) == g1.mat.at<int>(j, i));
            BOOST_CHECK(g.mat.at<int>(i, j) == g2.mat.at<cv::Vec3i>(i, 0)[j]);
            BOOST_CHECK(g.mat.at<int>(i, j) == g3.mat.at<int>(i, j));
        }
    }
}
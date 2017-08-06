#define BOOST_TEST_MODULE "Features"
#include <boost/test/unit_test.hpp>
#include <seq2map/sequence.hpp>

using namespace seq2map;

BOOST_AUTO_TEST_CASE(matching)
{
    Sequence seq;
    ImageFeatureSet f0, f1;

    if (!seq.Restore("KITTI_ODOMETRY_00"))
    {
        BOOST_FAIL("error reading test sequence");
    }

    if (!seq.GetFeatureStore(0))
    {
        BOOST_FAIL("error getting feature store 0");
    }

    if (!seq.GetFeatureStore(0)->Retrieve(0, f0) ||
        !seq.GetFeatureStore(0)->Retrieve(1, f1))
    {
        BOOST_FAIL("error retrieving feature set");
    }

    FeatureMatcher matcher(true, false, false, 0.8f, true);
    const ImageStore& im = seq.GetFeatureStore(0)->GetCamera()->GetImageStore();

    E_INFO << "feature set 0: " << f0.GetSize() << " feature(s)";
    E_INFO << "feature set 1: " << f1.GetSize() << " feature(s)";

    matcher.SetUniqueness(false);
    ImageFeatureMap map0 = matcher(f0, f1);
    //cv::imshow("DMatching", map0.Draw(im[0].im, im[1].im));
    E_INFO << "without uniqueness and symmetric checks, we got " << map0.Select(FeatureMatch::INLIER).size() << " match(es)";

    matcher.SetUniqueness(true);
    ImageFeatureMap map1 = matcher(f0, f1);
    //cv::imshow("DMatching + Uniqueness", map1.Draw(im[0].im, im[1].im));
    E_INFO << "by enforcing uniqueness check, we got " << map1.Select(FeatureMatch::INLIER).size() << " match(es)";

    matcher.SetSymmetric(true);
    ImageFeatureMap map2 = matcher(f0, f1);
    //cv::imshow("DMatching + Uniqueness + Symmetry", map2.Draw(im[0].im, im[1].im));
    E_INFO << "by enforcing uniqueness and symmetric checks, we got " << map2.Select(FeatureMatch::INLIER).size() << " match(es)";

    //cv::waitKey(0);
}

#define BOOST_TEST_MODULE "Geometry Problems"
#include <boost/test/unit_test.hpp>
#include <seq2map/sequence.hpp>

using namespace seq2map;

BOOST_AUTO_TEST_CASE(disparity)
{
    Sequence seq;
    DisparityStore::ConstOwn dpStore;
    RectifiedStereo::ConstOwn stereo;

    if (!seq.Restore("KITTI_ODOMETRY_00"))
    {
        BOOST_FAIL("error reading test sequence");
    }

    dpStore = seq.GetDisparityStore(0);

    if (!dpStore)
    {
        BOOST_FAIL("missing dispariy store");
    }

    stereo = dpStore->GetStereoPair();

    if (!stereo)
    {
        BOOST_FAIL("missing stereo pair");
    }

    cv::Mat dp = (*dpStore)[0].im;

    if (dp.empty())
    {
        BOOST_FAIL("missing disparity map");
    }

    Geometry g = stereo->Backproject(dp);
    PersistentMat(g.mat).Store(Path("bp.bin"));
}

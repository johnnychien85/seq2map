#define BOOST_TEST_MODULE "Geometry Problems"
#include <boost/test/unit_test.hpp>
#include <seq2map/sequence.hpp>
#include <seq2map/geometry_problems.hpp>

using namespace seq2map;

BOOST_AUTO_TEST_CASE(photometric)
{
    Sequence seq;
    DisparityStore::ConstOwn dpStore;
    RectifiedStereo::ConstOwn stereo;
    Camera::ConstOwn cam;

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

    cam = stereo->GetPrimaryCamera();

    if (!cam)
    {
        BOOST_FAIL("missing primary camera");
    }

    EuclideanTransform tform[2];

    for (size_t pass = 0; pass < 2; pass++)
    {
        cv::Mat dp = (*dpStore)[1].im;

        if (dp.empty())
        {
            BOOST_FAIL("missing disparity map");
        }

        size_t src = pass == 0 ? 0 : 1;
        size_t dst = pass == 0 ? 1 : 0;

        cv::Mat I0 = cam->GetImageStore()[src].im;
        cv::Mat I1 = cam->GetImageStore()[dst].im;

        if (I0.empty() || I1.empty())
        {
            BOOST_FAIL("missing image(s)");
        }

        PhotometricObjective* obj = new PhotometricObjective(cam->GetIntrinsics(), I1);
        Geometry g = stereo->Backproject(dp);

        cv::Mat z = g.Reshape(Geometry::ROW_MAJOR).mat.col(2);

        if (!obj->SetData(g, I0))
        {
            BOOST_FAIL("error setting data");
        }

        VectorisableD::Vec x(6); // initial solution to help convergence
        x[5] = pass == 0 ? -1.0f : 1.0f;

        MultiObjectivePoseEstimation problem;
        problem.SetDifferentiationStep(1e-3);
        problem.AddObjective(AlignmentObjective::Own(obj));
        problem.GetPose().Restore(x); // set initial solution

        LevenbergMarquardtAlgorithm solver;
        solver.SetInitialDamp(1e-2);
        solver.SetVervbose(true);

        if (!solver.Solve(problem))
        {
            BOOST_FAIL("error solving ego-motion");
        }

        tform[pass] = problem.GetPose();
    }

    EuclideanTransform err = tform[0] >> tform[1];
    cv::Mat drift = err.GetTranslation();
    
    BOOST_CHECK(cv::norm(drift) < 0.1f);
    E_INFO << mat2string(drift, "drift");
}

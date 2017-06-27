#define BOOST_TEST_MODULE "Geometry Problems"
#include <boost/test/unit_test.hpp>
#include <seq2map/sequence.hpp>
#include <seq2map/geometry_problems.hpp>

using namespace seq2map;

BOOST_AUTO_TEST_CASE(ransac)
{
    Sequence seq;
    FeatureStore::ConstOwn f0;
    FeatureStore::ConstOwn f1;
    ProjectionModel::ConstOwn p0;
    ProjectionModel::ConstOwn p1;
    ImageFeatureSet fset0;
    ImageFeatureSet fset1;

    if (!seq.Restore("KITTI_ODOMETRY_00"))
    {
        BOOST_FAIL("error reading test sequence");
    }

    f0 = seq.GetFeatureStore(0);
    f1 = seq.GetFeatureStore(1);

    if (!f0 || !f1)
    {
        BOOST_FAIL("missing feature store(s)");
    }

    if (!f0->Retrieve(0, fset0) || !f1->Retrieve(0, fset1))
    {
        BOOST_FAIL("error retrieving feature set(s)");
    }

    p0 = f0->GetCamera() ? f0->GetCamera()->GetIntrinsics() : ProjectionModel::Own();
    p1 = f1->GetCamera() ? f1->GetCamera()->GetIntrinsics() : ProjectionModel::Own();

    if (!p0 || !p1)
    {
        BOOST_FAIL("missing projection model(s)");
    }

    FeatureMatcher matcher;
    ImageFeatureMap fmap = matcher(fset0, fset1);
    GeometricMapping::ImageToImageBuilder builder;
    AlignmentObjective::Own objective = AlignmentObjective::Own(new EpipolarObjective(p0, p1, EpipolarObjective::GEOMETRIC));
    std::vector<size_t> idmap;

    for (size_t i = 0; i < fmap.GetMatches().size(); i++)
    {
        const FeatureMatch& m = fmap[i];

        if (m.state & FeatureMatch::INLIER)
        {
            builder.Add(fset0[m.srcIdx].keypoint.pt, fset1[m.dstIdx].keypoint.pt, i);
            idmap.push_back(i);
        }
    }

    GeometricMapping mapping = builder.Build();

    if (!objective->SetData(mapping))
    {
        BOOST_FAIL("error setting point correspondences");
    }

    PoseEstimator::ConstOwn solver = PoseEstimator::Own(new EssentialMatrixDecomposer(p0, p1));
    PoseEstimator::Estimate estimate;

    ConsensusPoseEstimator estimator;
    estimator.AddSelector(objective->GetSelector(0.0015f));
    estimator.SetConfidence(0.5f);
    estimator.SetMaxIterations(100);
    estimator.SetMinInlierRatio(0.8f);
    estimator.SetSolver(solver);
    estimator.SetVerbose(true);
    estimator.EnableOptimisation();

    ConsensusPoseEstimator::IndexLists inliers, outliers;

    if (!estimator(mapping, estimate, inliers, outliers))
    {
        BOOST_FAIL("error solving camera pose");
    }

    BOOST_FOREACH (size_t idx, outliers[0])
    {
        fmap[idmap[idx]].Reject(FeatureMatch::GEOMETRIC_TEST_FAILED);
    }

    EuclideanTransform calib = f1->GetCamera()->GetExtrinsics() - f0->GetCamera()->GetExtrinsics();

    E_INFO << mat2string(calib.GetTransformMatrix(), "CALIB");
    E_INFO << mat2string(estimate.pose.GetTransformMatrix(), "RANSAC");

    // cv::Mat im = fmap.Draw(f0->GetCamera()->GetImageStore()[0].im, f1->GetCamera()->GetImageStore()[0].im);
    // cv::imshow("match", im);
    // cv::waitKey(0);
}

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
        problem.AddObjective(AlignmentObjective::ConstOwn(obj));
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

#include <seq2map/mapping.hpp>

using namespace seq2map;
namespace po = boost::program_options;

class MyApp : public App
{
public:
    MyApp(int argc, char* argv[]) : App(argc, argv) {}

protected:
    virtual void SetOptions(Options&, Options&, Positional&);
    virtual void ShowHelp(const Options&) const;
    virtual bool Init();
    virtual bool Execute();

private:
    String m_seqPath;
    String m_outPath;
    size_t m_kptsStoreId;
    size_t m_dispStoreId;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Ego-motion estimation using monocular vision." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <sequence_database>" << std::endl;
    std::cout << o << std::endl;
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ("f", po::value<size_t>(&m_kptsStoreId)->default_value(0),  "source feature store")
        ("d", po::value<size_t>(&m_kptsStoreId)->default_value(-1), "optional disparity store");

    h.add_options()
        ("seq", po::value<String>(&m_seqPath)->default_value(""), "Path to the input sequence.")
        ("out", po::value<String>(&m_outPath)->default_value(""), "Path to the output motion.");

    p.add("seq", 1).add("out", 1);
}

bool MyApp::Init()
{
    if (m_seqPath.empty())
    {
        E_ERROR << "missing input path";
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    Sequence seq;

    if (!seq.Restore(m_seqPath))
    {
        E_ERROR << "error restoring sequence from " << m_seqPath;
        return false;
    }

    FeatureStore::ConstOwn   F = seq.GetFeatureStore(m_kptsStoreId);
    DisparityStore::ConstOwn D = seq.GetDisparityStore(m_dispStoreId);
    Camera::ConstOwn cam = F ? F->GetCamera() : Camera::ConstOwn();
    RectifiedStereo::ConstOwn stereo = D ? D->GetStereoPair() : RectifiedStereo::ConstOwn();

    if (!F)
    {
        E_ERROR << "missing feature store " << m_kptsStoreId;
        return false;
    }

    if (!cam)
    {
        E_ERROR << "missing camera";
        return false;
    }

    if (!D)
    {
        E_INFO << "no disparity store available, the estimated motion will have no absolute scale";
    }
    else if (!stereo)
    {
        E_ERROR << "missing stereo camera";
        return false;
     }
    else if (stereo->GetPrimaryCamera() != cam)
    {
        E_ERROR << "the primary camera of " << stereo->ToString() << " has to be cam " << cam->GetIndex();
        return false;
    }

    Map map;
    FeatureTracker tracker(
        FeatureTracker::FramedStore(F, 0, D), // tracking from frame k
        FeatureTracker::FramedStore(F, 1, D)  // to frame k+1
    );

    for (size_t t = 0; t < seq.GetFrames(); t++)
    {
        if (!tracker(map, t))
        {
            E_ERROR << "error tracking frame " << t << " -> " << (t+1);
            return false;
        }
    }

    cv::Mat K;
    intrinsics->GetCameraMatrix().convertTo(K, CV_32F);

    float fu = K.at<float>(0, 0);
    float fv = K.at<float>(1, 1);
    float epsilon = 0.2f / ((fu + fv) * 0.5f);

    FeatureMatcher matcher;
    EssentialMatrixFilter ematFilter(epsilon, 0.995, true, true, K, K);
    //FundamentalMatrixFilter fmatFilter(epsilon, 0.995, false);
    SigmaFilter sigmaFilter(0.5f);

    matcher.AddFilter(ematFilter).AddFilter(sigmaFilter);
    //matcher.AddFilter(sigmaFilter).AddFilter(fmatFilter);

    size_t t0 = 2;
    size_t tn = cam.GetFrames() - 1;
    size_t frames = tn - t0 + 1;

    Motion motion;
    motion.Update(EuclideanTransform::Identity); // initialise t0 as the referenced frame

    Motion ground;
    ground.Restore("ground.txt");

    Points3F Gi, Gj;
    std::vector<double> Wi, Wj;
    std::vector<int>    Ui, Uj;
    int uid = 0;

    for (size_t t = t0; t < tn; t++)
    {
        size_t ti = t;
        size_t tj = t + 1;
        Frame Fi = cam[ti];
        Frame Fj = cam[tj];
        MotionEstimation egomo(K, 0.5f, true, false);
        EuclideanTransform Mij;

        //E_INFO << ti << " -> " << tj << " : " << matcher.Report();

        if (Fi.features.IsEmpty() || Fj.features.IsEmpty())
        {
            E_ERROR << "error reading features for frames " << ti << " -> " << tj;
        }

        if (ti == t0)
        {
            Gi = Points3F(Fi.features.GetSize(), cv::Point3f(0, 0, 0));
            Wi = std::vector<double>(Gi.size(), 0.0f);
            Ui = std::vector<int>(Gi.size(), 0);
        }

        Gj = Points3F(Fj.features.GetSize(), cv::Point3f(0, 0, 0));
        Wj = std::vector<double>(Gj.size(), 0.0f);
        Uj = std::vector<int>(Gj.size(), 0);

        ImageFeatureMap fmap = matcher.MatchFeatures(Fi.features, Fj.features);

        if (!Fi.im.empty() && !Fj.im.empty())
        {
            cv::imshow("Feature Matching", fmap.Draw(Fi.im, Fj.im));
            cv::waitKey(1);
        }

        BOOST_FOREACH (FeatureMatch& match, fmap.GetMatches())
        {
            if (!(match.state & FeatureMatch::INLIER)) continue;

            Point3F x3di = Gi[match.srcIdx];
            double  w3di = Wi[match.srcIdx];
            Point2F x2di = Fi.features[match.srcIdx].keypoint.pt;
            Point2F x2dj = Fj.features[match.dstIdx].keypoint.pt;
            size_t  uidi = Ui[match.srcIdx];
            size_t  uidj = uidi > 0 ? uidi : ++uid;

            Uj[match.dstIdx] = uidj;

            egomo.AddObservation(uidj, x2di, x2dj);

            if (x3di.z > 0 /*&& x3di.z < 50*/)
            {
                egomo.AddObservation(uidj, x3di, x2dj, w3di);
            }
        }

        if (t == t0)
        {
            if (ground.IsEmpty())
            {
                cv::Mat rmat, tvec;
                ematFilter.GetPose(rmat, tvec);

                Mij.SetRotationMatrix(rmat);
                Mij.SetTranslation(tvec);
            }
            else
            {
                Mij = ground.GetLocalTransform(ti, tj);
            }

            ematFilter.SetPoseRecovery(false);
        }
        else
        {
            LevenbergMarquardtAlgorithm levmar(10.0f, 0.0f, false);

            if (!levmar.Solve(egomo, egomo.Initialise()))
            {
                E_ERROR << "ego-motion estimation failed";
                return false;
            }

            Mij = egomo.GetTransform();
        }

        E_INFO << ti << " -> " << tj << " : " << " #PnP=" << egomo.GetReprojectionConds().GetSize() << " #epi=" << egomo.GetEpipolarConds().GetSize();

        //OptimalTriangulator
        //MidPointTriangulator
        MidPointTriangulator triangulator(
            intrinsics->MakeProjectionMatrix(EuclideanTransform::Identity),
            intrinsics->MakeProjectionMatrix(Mij));

        Points3D x3dj;
        std::vector<double> e3dj;
        triangulator.Triangulate(egomo.GetEpipolarConds(), x3dj, e3dj);

        std::stringstream ss;
        ss << "egomo_t" << ti << ".m";
        Path saveto(ss.str());
        egomo.Store(saveto);

        E_INFO << mat2string(Mij.GetTransformMatrix().t(), "M", 3);

        if (!ground.IsEmpty())
        {
            EuclideanTransform M0ij = ground.GetLocalTransform(ti, tj);
            EuclideanTransform diff = M0ij - Mij;

            double drift = (100.0f * cv::norm(diff.GetTranslation()) / cv::norm(M0ij.GetTranslation()));

            E_INFO << mat2string(M0ij.GetTransformMatrix().t(), "G", 3);
            E_INFO << "drift: " << drift << "% over " << cv::norm(M0ij.GetTranslation()) << "cm";

            if (drift > 2.5f)
            {
                //E_FATAL << "GAME OVER";
                //return -1;
            }
        }

        //cv::triangulatePoints(Pi, Pj, x2di, x2dj, x3dj);
        size_t i = 0;
        BOOST_FOREACH (FeatureMatch& match, fmap.GetMatches())
        {
            if (!(match.state & FeatureMatch::INLIER)) continue;

            Point3D gi = Gi[match.srcIdx];
            double  wi = Wi[match.srcIdx];

            Point3D gj = x3dj[i];
            double  wj = 1 / (1 + e3dj[i]);

            //double wj = wi == 0 ? 1 : (1 / (1 + cv::norm(cv::Mat(gi) - cv::Mat(gj))));

            Wj[match.dstIdx] = wi + wj;

            Mij.Apply(gi);
            Mij.Apply(gj);

            Gj[match.dstIdx] = Point3D(
                (wi * gi.x + wj * gj.x) / Wj[match.dstIdx],
                (wi * gi.y + wj * gj.y) / Wj[match.dstIdx],
                (wi * gi.z + wj * gj.z) / Wj[match.dstIdx]);

            i++;
        }

        motion.Update(Mij);
        Gi = Gj;
        Wi = Wj;
        Ui = Uj;
    }

    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

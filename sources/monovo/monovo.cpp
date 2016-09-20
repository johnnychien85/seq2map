#include <seq2map/sequence.hpp>

using namespace seq2map;
namespace po = boost::program_options;

struct Args
{
    Path seqPath;
    size_t camIdx;
    bool help;
};

bool init(int, char*[], Args&);
void showSynopsis(char*);

int main(int argc, char* argv[])
{
    Args args;

    if (!init(argc, argv, args)) return args.help ? EXIT_SUCCESS : EXIT_FAILURE;

    try
    {
        Sequence seq;
        
        if (!seq.Restore(args.seqPath))
        {
            E_FATAL << "error restoring sequence from " << fullpath(args.seqPath);
            return -1;
        }

        if (args.camIdx > seq.GetCameras().size())
        {
            E_FATAL << "camera " << args.camIdx << " selected while the sequence has only " << seq.GetCameras().size() << " camera(s)";
            return -1;
        }

        const Camera& cam = seq[args.camIdx];
        const BouguetModel::Ptr intrinsics = boost::dynamic_pointer_cast<BouguetModel>(cam.GetIntrinsics());

        if (!intrinsics)
        {
            E_FATAL << "the camera model is not Bouguet";
            return -1;
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
                if (!(match.state & FeatureMatch::Flag::INLIER)) continue;

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
            egomo.Store(Path(ss.str()));
            
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
                if (!(match.state & FeatureMatch::Flag::INLIER)) continue;

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
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caught in main loop: " << ex.what();
        return -1;
    }

    return 0;
}

bool init(int argc, char* argv[], Args& args)
{
    String seqPath;

    po::options_description o("General Options");
    o.add_options()
        ("cam,c", po::value<size_t>(&args.camIdx)->default_value(0), "Index of the selected camera")
        ("help,h", "Show this help message and exit.");

    po::options_description h("Hiddens");
    h.add_options()
        ("in", po::value<String>(&seqPath)->default_value(""), "Input sequence descriptor");

    po::positional_options_description p;
    p.add("in", 1);

    try
    {
        po::options_description a;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(a.add(o).add(h)).positional(p).run();
        po::variables_map vm;

        po::store(parsed, vm);
        po::notify(vm);

        args.help = vm.count("help") > 0;
    }
    catch (po::error& pe)
    {
        E_FATAL << "error parsing general arguments: " << pe.what();
        showSynopsis(argv[0]);

        return false;
    }
    catch (std::exception& ex)
    {
        E_FATAL << "exception caugth: " << ex.what();
        return false;
    }

    args.seqPath = seqPath;

    if (args.help)
    {
        std::cout << "Usage: " << argv[0] << " <seq_descriptor_path> [options]" << std::endl;
        std::cout << o << std::endl;

        return false;
    }

    if (seqPath.empty())
    {
        E_FATAL << "input sequence descriptor missing";
        showSynopsis(argv[0]);

        return false;
    }

    return true;
}

void showSynopsis(char* exec)
{
    std::cout << std::endl;
    std::cout << "Try \"" << exec << " -h\" for usage listing." << std::endl;
}

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
        float epsilon = 0.1f / ((fu + fv) * 0.5f);

        FeatureMatcher matcher;
        EssentialMatrixFilter ematFilter(epsilon, 0.995f, true, true, K, K);
        //FundamentalMatrixFilter fmatFilter;
        SigmaFilter sigmaFilter;
        //matcher.AddFilter(fmatFilter).AddFilter(sigmaFilter);
        matcher.AddFilter(ematFilter).AddFilter(sigmaFilter);

        size_t t0 = 0;
        size_t tn = cam.GetFrames() - 1;
        size_t frames = tn - t0 + 1;

        Motion motion;
        motion.Update(EuclideanTransform::Identity); // initialise t0 as the referenced frame

        Points3F Gi, Gj;

        for (size_t t = t0; t < tn; t++)
        {
            size_t ti = t;
            size_t tj = t + 1;

            Frame Fi = cam[ti];
            Frame Fj = cam[tj];

            if (Fi.features.IsEmpty() || Fj.features.IsEmpty())
            {
                E_ERROR << "error reading features for frames " << ti << " -> " << tj;
            }

            if (t == t0) Gi = Points3F(Fi.features.GetSize(), cv::Point3f(0, 0, 0));
            Gj = Points3F(Fj.features.GetSize(), cv::Point3f(0, 0, 0));

            ImageFeatureMap fmap = matcher.MatchFeatures(Fi.features, Fj.features);

            if (!Fi.im.empty() && !Fj.im.empty())
            {
                cv::imshow("Feature Matching", fmap.Draw(Fi.im, Fj.im));
                cv::waitKey(1);
            }

            E_INFO << ti << " -> " << tj << " : " << matcher.Report();

            MotionEstimation egomo(K);
            EuclideanTransform Mij;
            
            FeatureMatches knowns;

            BOOST_FOREACH (FeatureMatch& match, fmap.GetMatches())
            {
                if (!(match.state & FeatureMatch::Flag::INLIER)) continue;

                Point3F x3di = Gi[match.srcIdx];
                Point2F x2di = Fi.features[match.srcIdx].keypoint.pt;
                Point2F x2dj = Fj.features[match.dstIdx].keypoint.pt;

                egomo.AddObservation(x2di, x2dj);

                if (x3di.z > 0 && x3di.z < 50)
                {
                    egomo.AddObservation(x3di, x2dj);
                    knowns.push_back(match);
                }
            }

            if (t == t0)
            {
                cv::Mat rmat, tvec;
                ematFilter.GetPose(rmat, tvec);

                Mij.SetRotationMatrix(rmat);
                Mij.SetTranslation(tvec);

                ematFilter.SetPoseRecovery(false);
            }
            else
            {
                LevenbergMarquardtAlgorithm levmar;
                levmar.SetVervbose(true);
                bool solved = levmar.Solve(egomo, egomo.Initialise());

                if (!solved)
                {
                    E_ERROR << "egomotion estimation failed";
                    return false;
                }

                Mij = egomo.GetTransform();

                //cv::Mat rvec, tvec;
                //cv::solvePnP(pts3, pts2, K, cv::Mat(), rvec, tvec, false, CV_EPNP);

                //Mij.SetRotationVector(rvec);
                //Mij.SetTranslation(tvec);
            }

            OptimalTriangulator triangulator(
                intrinsics->MakeProjectionMatrix(EuclideanTransform::Identity),
                intrinsics->MakeProjectionMatrix(Mij));

            Points3F x3dj;
            triangulator.Triangulate(egomo.GetEpipolarConds(), x3dj);
            
            /*
            if (t == 6)
            {
                std::ofstream of("check.m", std::ofstream::out);
                of << mat2string(Mij.GetTransformMatrix(), "M") << std::endl;
                of << mat2string(K, "K") << std::endl;

                of << mat2string(cv::Mat(x3di).reshape(1), "x0") << std::endl;
                of << mat2string(cv::Mat(x2di).reshape(1), "y0") << std::endl;
                of << mat2string(cv::Mat(x2dj).reshape(1), "y1") << std::endl;
                of << mat2string(cv::Mat(x2dj2).reshape(1), "r1") << std::endl;

                //of << mat2string(cv::Mat(Gi).reshape(1), "Gi") << std::endl;
                //of << mat2string(Mij.GetTransformMatrix(), "M") << std::endl;
                //Mij.Apply(Gi);
                //of << mat2string(cv::Mat(Gi).reshape(1), "Gj") << std::endl;
            }
            else
            {
                //Mij.Apply(Gi);
            }
            */
            
            E_INFO << mat2string(Mij.GetTransformMatrix().t(), "", 2);

            //cv::triangulatePoints(Pi, Pj, x2di, x2dj, x3dj);
            size_t i = 0;
            BOOST_FOREACH (FeatureMatch& match, fmap.GetMatches())
            {
                if (!(match.state & FeatureMatch::Flag::INLIER)) continue;

                cv::Point3f gi = Gi[match.srcIdx];
                cv::Point3f gj = x3dj[i];

                bool initialised = gi.z > 0 && gi.z < 50;
                
                if (initialised) Mij.Apply(gi);
                
                Gj[match.dstIdx] = initialised ? cv::Point3f(
                    (gi.x + gj.x) * 0.5f,
                    (gi.y + gj.y) * 0.5f,
                    (gi.z + gj.z) * 0.5f) : gj;

                //if (initialised) E_INFO << (gi.z - gj.z);

                //E_INFO << "[" << x2dj[i].x << "," << x2dj[i].y << "] -> [" << x2dj2[i].x << "," << x2dj2[i].y << "]";
                //if (initialised) E_INFO << "[" << gi.x << "," << gi.y << "," << gi.z << "] -> [" << x3dj.at<float>(i, 0) << "," << x3dj.at<float>(i, 1) << "," << x3dj.at<float>(i, 2) << "]";

                i++;
            }

            motion.Update(Mij);
            Gi = Gj;
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

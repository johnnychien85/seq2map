#include <seq2map/app.hpp>
#include <seq2map/mapping.hpp>
#include <DBoW3/DBoW3.h>

using namespace seq2map;

class MyApp : public App
{
public:
    MyApp(int argc, char* argv[]) : App(argc, argv) {}

protected:
    virtual void SetOptions(Options&, Options&, Positional&);
    virtual void ShowHelp(const Options&) const;
    virtual bool Init();
    virtual bool Execute();

    String srcPath;
    String dstPath;
    Map src;
    Map dst;

    int levels;
    int degree;
    double minScore;
    double maxDist;
    int maxIters;
    double minInliers;
};

void MyApp::ShowHelp(const Options& o) const
{
    std::cout << "Multi-sequence integration." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << m_exec.string() << " [options] <source_map> <target_map>" << std::endl;
    std::cout << o << std::endl;
}

void MyApp::SetOptions(Options& o, Options& h, Positional& p)
{
    namespace po = boost::program_options;

    o.add_options()
        ("levels",      po::value<int>   (&levels    )->default_value(  10), "Height of the vocabulary tree.")
        ("degree",      po::value<int>   (&degree    )->default_value(   5), "Number of branches at each level in the vocabulary tree.")
        ("min-score",   po::value<double>(&minScore  )->default_value(.15f), "Threshold of similarity score to consider a positive match.")
        ("max-dist",    po::value<double>(&maxDist   )->default_value(1.0f), "Threshold to consider a 3D-to-2D correspondence an inlier.")
        ("max-iter",    po::value<int>   (&maxIters  )->default_value( 100), "Maximum number of RANSAC iterations")
        ("min-inliers", po::value<double>(&minInliers)->default_value(0.2f), "Minimum ratio inliers to consider a model fit")
        ;

    h.add_options()
        ("src",    po::value<String>(&srcPath)->default_value(""), "Path to the source map.")
        ("dst",    po::value<String>(&dstPath)->default_value(""), "Path to the target map.")
        ;

    p.add("src", 1).add("dst", 1);
}

bool MyApp::Init()
{
    if (srcPath.empty())
    {
        E_ERROR << "missing path to source map";
        return false;
    }

    if (dstPath.empty())
    {
        E_ERROR << "missing path to target map";
        return false;
    }

    if (!src.Restore(Path(srcPath)))
    {
        E_ERROR << "error restoring source map from \"" << srcPath << "\"";
        return false;
    }

    if (!dst.Restore(Path(dstPath)))
    {
        E_ERROR << "error restoring target map from \"" << dstPath << "\"";
        return false;
    }

    return true;
}

bool MyApp::Execute()
{
    typedef std::set<size_t> IndexSet;
    const IndexSet kfs0 = src.GetKeyframes();
    const IndexSet kfs1 = dst.GetKeyframes();
    FeatureStore::ConstOwn fs0 = src.GetSource(0).store;
    FeatureStore::ConstOwn fs1 = dst.GetSource(0).store;

    if (!fs0 || !fs1)
    {
        E_ERROR << "missing feature store(s)";
        return false;
    }

    const ImageStore& I0 = fs0->GetCamera()->GetImageStore();
    const ImageStore& I1 = fs1->GetCamera()->GetImageStore();
    FeatureDetextractor::ConstOwn dxtor = FeatureDetextractorFactory::GetInstance().Create("ORB", "ORB");

    DBoW3::Vocabulary voc(degree, levels);
    {
        E_INFO << "collecting features in " << kfs0.size() << " key frame(s) of the source sequence for vocabulary creation";

        std::vector<cv::Mat> samples;
        size_t features = 0;
        for (IndexSet::const_iterator kf0 = kfs0.cbegin(); kf0 != kfs0.cend(); kf0++)
        {
            //samples.push_back((*fs0)[*kf0].GetDescriptors());
            samples.push_back(dxtor->DetectAndExtractFeatures(I0[*kf0].im).GetDescriptors());
            features += static_cast<size_t>(samples.back().rows);
        }

        E_INFO << "creating vocabulary from " << features << " collected training features..";
        voc.create(samples);
        E_INFO << "vocabulary creation finished";
        E_INFO << voc;
    }

    DBoW3::Database db(voc, false, 0);
    std::vector<size_t> srcIdxMap;
    srcIdxMap.reserve(kfs0.size());
    {
        E_INFO << "creating database from source sequence..";
        for (IndexSet::const_iterator kf0 = kfs0.cbegin(); kf0 != kfs0.cend(); kf0++)
        {
            // DBoW3::BowVector v0;
            // ImageFeatureSet f0 = (*fs0)[*kf0];
            // f0.Append(src.GetFrame(*kf0).augmentedFeaturs[fs0->GetIndex()]);
            // voc.transform(f0.GetDescriptors(), v0);

            DBoW3::BowVector v0;

            voc.transform(dxtor->DetectAndExtractFeatures(I0[*kf0].im).GetDescriptors(), v0);
            db.add(v0);
            srcIdxMap.push_back(*kf0);
        }
        E_INFO << "database creation finished";
        E_INFO << db;
    }

    E_INFO << "evaluating cross-sequence similarity..";
    cv::Mat scores = cv::Mat::zeros(kfs0.size(), kfs1.size(), CV_64F);
    FeatureMatcher matcher;
    ConsensusPoseEstimator estimator;
    matcher.AddFilter(FeatureMatcher::Filter::Own(new FundamentalMatrixFilter(1.0f, 0.995f, true)));
    ProjectionModel::Own P0 = boost::dynamic_pointer_cast<PosedProjection, ProjectionModel>(fs0->GetCamera()->GetPosedProjection());
    ProjectionModel::Own P1 = boost::dynamic_pointer_cast<PosedProjection, ProjectionModel>(fs1->GetCamera()->GetPosedProjection());
    PoseEstimator::ConstOwn mainSolver;
    GeometricMapping mainMapping;
    size_t maxInliers = 0;
    {
        size_t j = 0;
        for (IndexSet::const_iterator kf1 = kfs1.cbegin(); kf1 != kfs1.cend(); kf1++)
        {
            // DBoW3::BowVector v1;
            // ImageFeatureSet f1 = (*fs1)[*kf1];
            // f1.Append(dst.GetFrame(*kf1).augmentedFeaturs[fs1->GetIndex()]);
            // voc.transform(f1.GetDescriptors(), v1);

            DBoW3::BowVector v1;
            DBoW3::QueryResults rs;

            voc.transform(dxtor->DetectAndExtractFeatures(I1[*kf1].im).GetDescriptors(), v1);
            db.query(v1, rs, 0);

            // double bstScore = 0.0f;
            // size_t bstMatch = INVALID_INDEX;

            for (size_t i = 0; i < rs.size(); i++)
            {
                scores.row(rs[i].Id).col(j) = rs[i].Score;
            }

            Frame& ti = src.GetFrame(srcIdxMap[rs[0].Id]);
            Frame& tj = dst.GetFrame(*kf1);

            E_INFO << "best match of frame " << tj.GetIndex() << " is " << ti.GetIndex();

            // false-positive rejection
            if (!rs.empty() && rs[0].Score > minScore)
            {
                ImageFeatureSet fi = (*fs0)[ti.GetIndex()];
                ImageFeatureSet fj = (*fs1)[tj.GetIndex()];

                fi.Append(ti.augmentedFeaturs[fs0->GetIndex()]);
                fj.Append(tj.augmentedFeaturs[fs1->GetIndex()]);

                const ImageFeatureMap fmap = matcher(fi, fj);
                const FeatureMatches matches = fmap.GetMatches();

                if (matches.size() < 8)
                {
                    E_INFO << "(" << ti.GetIndex() << "," << tj.GetIndex() << ") rejected due to insufficient inlier(s) of " << matches.size();
                    continue;
                }

                ProjectionModel::Own Pi = ProjectionModel::Own(new PosedProjection(ti.pose.pose, P0));
                ProjectionModel::Own Pj = ProjectionModel::Own(new PosedProjection(tj.pose.pose, P1));

                ProjectionObjective::Own Oj = ProjectionObjective::Own(new ProjectionObjective(Pj, false));

                GeometricMapping::WorldToImageBuilder forward, backward;

                BOOST_FOREACH (const FeatureMatch& m, matches)
                {
                    Landmark* li = ti.featureLandmarkLookup[fs0->GetIndex()][m.srcIdx];
                    Landmark* lj = tj.featureLandmarkLookup[fs1->GetIndex()][m.dstIdx];

                    if (li && li->position.z != 0)
                    {
                        forward.Add(li->position, fj[m.dstIdx].keypoint.pt, li->GetIndex());
                    }

                    if (lj && lj->position.z != 0)
                    {
                        backward.Add(lj->position, fi[m.srcIdx].keypoint.pt, lj->GetIndex());
                    }
                }

                if (forward.GetSize() > 0)
                {
                    ProjectionObjective::Own obj = ProjectionObjective::Own(new ProjectionObjective(Pj, true));
                    if (!obj->SetData(forward.Build()))
                    {
                        E_ERROR << "error building projective alignment data for " << ti.GetIndex() << " -> " << tj.GetIndex();
                        return false;
                    }

                    estimator.AddSelector(obj->GetSelector(maxDist));

                    if (forward.GetSize() > maxInliers)
                    {
                        mainSolver = PoseEstimator::Own(new PerspevtivePoseEstimator(Pj));
                        mainMapping = forward.Build();
                        maxInliers = forward.GetSize();

                        E_INFO << "main solver set to " << ti.GetIndex() << " -> " << tj.GetIndex();
                    }
                }
                
                if (backward.GetSize() > 0)
                {
                    ProjectionObjective::Own obj = ProjectionObjective::Own(new ProjectionObjective(Pi, false));
                    if (!obj->SetData(backward.Build()))
                    {
                        E_ERROR << "error building projective alignment data for " << tj.GetIndex() << " -> " << ti.GetIndex();
                        return false;
                    }
                }

                E_INFO << matches.size() << " found for " << ti.GetIndex() << " -> " << tj.GetIndex() << ": " << forward.GetSize() << " forward mapping(s) and " << backward.GetSize() << " backward mapping(s) inserted";
                // cv::imshow("matching", fmap.Draw(I0[ti.GetIndex()].im, I1[tj.GetIndex()].im));
                // cv::waitKey(0);
            }
            j++;
        }
    }

    PersistentMat(scores).Store(Path("hits.bin"));

    if (!mainSolver)
    {
        E_ERROR << "no solver available to solve for cross-sequence alignment";
        return false;
    }

    PoseEstimator::Estimate estimate;

    estimator.SetStrategy(ConsensusPoseEstimator::RANSAC);
    estimator.SetMaxIterations(maxIters);
    estimator.SetMinInlierRatio(minInliers);
    estimator.SetConfidence(0.995);
    estimator.SetSolver(mainSolver);
    estimator.SetVerbose(true);
    estimator.EnableOptimisation();

    if (!estimator(mainMapping, estimate))
    {
        return false;
    }

    E_INFO << mat2string(estimate.pose.GetTransformMatrix(), "E");

    return true;
}

int main(int argc, char* argv[])
{
    MyApp app(argc, argv);
    return app.Run();
}

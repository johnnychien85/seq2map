#include <boost/algorithm/string/join.hpp>
#include <seq2map/mapping.hpp>

using namespace seq2map;

//==[ Map ]===================================================================//

Source& Map::AddSource(FeatureStore::ConstOwn& store, DisparityStore::ConstOwn& dpm)
{
    const bool validDisp =
        dpm && dpm->GetStereoPair() && store->GetCamera()->GetIndex() == dpm->GetStereoPair()->GetPrimaryCamera()->GetIndex();

    Source& src = Dim2(m_newSourcId++);

    src.store = store;
    src.dpm = validDisp ? dpm : DisparityStore::Own();

    return src;
}

Landmark& Map::AddLandmark()
{
    return Dim0(m_newLandmarkId++);
}

void Map::RemoveLandmark(Landmark& l)
{
    // unlink all references in lookup tables
    for (Landmark::const_iterator h = l.cbegin(); h; h++)
    {
        const Hit& hit = *h;

        Frame& t = h.GetContainer<1, Frame>();
        const Source& c = h.GetContainer<2, Source>();

        t.featureLandmarkLookup[c.store->GetIndex()][hit.index] = NULL;
    }

    l.clear();
}

bool Map::IsJoinable(const Landmark& li, const Landmark& lj)
{
    // parallel traversal
    Landmark::const_iterator hi = li.cbegin();
    Landmark::const_iterator hj = lj.cbegin();

    while (hi && hj)
    {
        const Frame& ti = hi.GetContainer<1, Frame>();
        const Frame& tj = hj.GetContainer<1, Frame>();

        if (ti < tj)
        {
            hi++;
        }
        else if (tj < ti)
        {
            hj++;
        }
        else
        {
            const Source& ci = hi.GetContainer<2, Source>();
            const Source& cj = hj.GetContainer<2, Source>();

            if (ci < cj)
            {
                hi++;
            }
            else if (cj < ci)
            {
                hj++;
            }
            else
            {
                return false;
            }
        }
    }

    return true;
}

Landmark& Map::JoinLandmark(Landmark& li, Landmark& lj)
{
    for (Landmark::const_iterator hj = lj.cbegin(); hj; hj++)
    {
        const Hit& hit = *hj;

        Frame&  tj = hj.GetContainer<1, Frame>();
        Source& cj = hj.GetContainer<2, Source>();

        li.Hit(tj, cj, hit.index) = hit;

        // ID table rewriting
        Landmark::Ptrs& uj = tj.featureLandmarkLookup[cj.store->GetIndex()];
        uj[hit.index] = &li;
    }

    lj.clear(); // abondaned

    return li;
}

StructureEstimation::Estimate Map::GetStructure(const Landmark::Ptrs& u) const
{
    typedef cv::Vec6d Covar6D; // 6 coefficients of full 3D covariance matrix

    cv::Mat mat = cv::Mat(static_cast<int>(u.size()), 3, CV_64F).reshape(3);
    cv::Mat cov = cv::Mat(static_cast<int>(u.size()), 6, CV_64F).reshape(6);

    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL)
        {
            mat.at<Point3D>(static_cast<int>(k)) = Point3D(0, 0, 0);
            cov.at<Covar6D>(static_cast<int>(k)) = Covar6D(0, 0, 0, 0, 0, 0);
        }
        else
        {
            const Landmark& lk = *u[k];
            const Landmark::Covar3D& ck = lk.cov;

            mat.at<Point3D>(static_cast<int>(k)) = lk.position;
            cov.at<Covar6D>(static_cast<int>(k)) = Covar6D(ck.xx, ck.xy, ck.xz, ck.yy, ck.yz, ck.zz);
        }
    }

    return StructureEstimation::Estimate(
        Geometry(Geometry::ROW_MAJOR, mat.reshape(1)),
        Metric::Own(new MahalanobisMetric(MahalanobisMetric::ANISOTROPIC_ROTATED, 3, cov.reshape(1)))
    );

    /*
    cv::Mat mat = cv::Mat(static_cast<int>(u.size()), 3, CV_64F).reshape(3);
    cv::Mat icv = cv::Mat(static_cast<int>(u.size()), 1, CV_64F);

    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL)
        {
            mat.at<Point3D>(static_cast<int>(k)) = Point3D(0, 0, 0);
            icv.at <double>(static_cast<int>(k)) = 0;
        }
        else
        {
            const Landmark& lk = *u[k];
            const Landmark::Covar3D ck = lk.cov;

            mat.at<Point3D>(static_cast<int>(k)) = lk.position;
            icv.at<double> (static_cast<int>(k)) = lk.icv;
        }
    }

    return StructureEstimation::Estimate(
        Geometry(Geometry::ROW_MAJOR, mat.reshape(1)),
        Metric::Own(new WeightedEuclideanMetric(icv))
    );
    */
}

void Map::SetStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& structure)
{
    typedef cv::Vec6d Covar6D; // 6 coefficients of full 3D covariance matrix

    boost::shared_ptr<MahalanobisMetric> metric
        = boost::dynamic_pointer_cast<MahalanobisMetric, Metric>(structure.metric);

    if (!metric || metric->type != MahalanobisMetric::ANISOTROPIC_ROTATED)
    {
        E_ERROR << "structure has to be equipped with a full Mahalanobis metric";
        return;
    }

    const cv::Mat& mat = structure.structure.mat.reshape(3);
    const cv::Mat& cov = metric->GetCovariance().mat.reshape(6);

    // write back to the landmarks
    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL) continue;

        Landmark& lk = *u[k];

        lk.position = mat.at<Point3D>(static_cast<int>(k));
        lk.cov      = cov.at<Covar6D>(static_cast<int>(k));
    }

    /*
    const WeightedEuclideanMetric* metric =
        dynamic_cast<const WeightedEuclideanMetric*>(structure.metric.get());

    if (!metric)
    {
        E_ERROR << "structure has to be equipped with a weighted Euclidean metric";
        return;
    }

    const cv::Mat& mat = structure.structure.mat.reshape(3);
    const cv::Mat& icv = metric->GetWeights();

    // write back to the landmarks
    for (size_t k = 0; k < u.size(); k++)
    {
        if (u[k] == NULL) continue;

        Landmark& lk = *u[k];

        lk.position = mat.at<Point3D>(static_cast<int>(k));
        lk.icv      = icv.at<double> (static_cast<int>(k));
    }
    */
}

void Map::Clear()
{
    m_newSourcId = 0;
    m_newLandmarkId = 0;
    m_keyframes.clear();

    Map3::Clear();
}

StructureEstimation::Estimate Map::UpdateStructure(const Landmark::Ptrs& u, const StructureEstimation::Estimate& g, const EuclideanTransform& ref)
{
    if (u.empty())
    {
        return StructureEstimation::Estimate(Geometry::ROW_MAJOR);
    }

    if (g.structure.IsEmpty())
    {
        return GetStructure(u).Transform(ref);
    }

    assert(g.structure.shape == Geometry::ROW_MAJOR);
    assert(g.structure.GetElements() == u.size());
    assert(g.structure.GetDimension() == 3);

    StructureEstimation::Estimate g0 = GetStructure(u); // extract structure of the involved landmarks
    g0 += g.Transform(ref.GetInverse()); // state update, recursive Bayesian
    SetStructure(u, g0);                 // write back the extracted structure

    return g0.Transform(ref);
}

bool Map::Store(Path & path) const
{
    Path idxPath = path / "index.yml";
    Path lmkPath = path / "landmarks.dat";
    Path hitPath = path / "hits.dat";
    Path augPath = path / "aug";

    if (!makeOutDir(path))
    {
        E_ERROR << "error creating directory " << path;
        return false;
    }

    if (!makeOutDir(augPath))
    {
        E_ERROR << "error creating directory for augmented features " << augPath;
        return false;
    }

    cv::FileStorage fs(idxPath.string(), cv::FileStorage::WRITE);
    std::ofstream os;

    if (!fs.isOpened())
    {
        E_ERROR << "error opening " << idxPath << " for writing";
        return false;
    }

    // landmarks
    os.open(lmkPath.string().c_str(), std::ios::out | std::ios::binary);
    if (!os.is_open())
    {
        E_ERROR << "error writing landmarks to " << lmkPath;
        return false;
    }
    size_t landmarks = 0;
    for (S0::const_iterator itr = Begin0(); itr != End0(); itr++)
    {
        const Landmark& l = itr->second;

        os.write((char*)&l.position, sizeof l.position);
        os.write((char*)&l.cov,      sizeof l.cov);

        landmarks++;
    }
    assert(landmarks == GetSize0());
    E_TRACE << landmarks << " landmark(s) written to " << lmkPath;
    os.close();

    // hits
    os.open(hitPath.string(), std::ios::out | std::ios::binary);
    if (!os.is_open())
    {
        E_ERROR << "error writing hits to " << hitPath;
        return false;
    }
    size_t hits = 0;
    for (S0::const_iterator itr = Begin0(); itr != End0(); itr++)
    {
        const Landmark& l = itr->second;
        size_t i = l.GetIndex();

        for (Landmark::const_iterator h = l.cbegin(); h != l.cend(); h++)
        {
            size_t j = h.GetContainer<1, Frame> ().GetIndex();
            size_t k = h.GetContainer<2, Source>().GetIndex();

            os.write((char*)&i, sizeof i);
            os.write((char*)&j, sizeof j);
            os.write((char*)&k, sizeof k);
            os.write((char*)&h->index, sizeof h->index);
            os.write((char*)&h->proj,  sizeof h->proj );
        }
    }
    os.close();

    // augmented image feature sets

    fs << "sequence" << m_seqPath;
    fs << "sources" << "[";
    for (S2::const_iterator itr = Begin2(); itr != End2(); itr++)
    {
        const Source& s = itr->second;
        fs << "{";
        fs << "keypoints" << s.store->GetIndex();
        fs << "disparity" << (s.dpm ? s.dpm->GetIndex() : INVALID_INDEX);
        fs << "}";
    }
    fs << "]";
    fs << "frames" << "[";
    for (S1::const_iterator itr = Begin1(); itr != End1(); itr++)
    {
        const Frame& t = itr->second;
        fs << "{";
        fs << "index" << t.GetIndex();
        fs << "hits"  << t.size();
        fs << "pose"  << t.pose.pose.GetTransformMatrix();
        fs << "augmentedFeatureSet" << "{";
        for (std::map<size_t, ImageFeatureSet>::const_iterator aug = t.augmentedFeaturs.begin(); aug != t.augmentedFeaturs.end(); aug++)
        {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(5) << t.GetIndex() << "." << std::setfill('0') << std::setw(2) << aug->first << ".dat";

            Path saveTo = augPath / ss.str();

            if (!aug->second.Store(saveTo))
            {
                E_ERROR << "error writing augmented feature set to " << saveTo;
                return false;
            }

            fs << "store" << aug->first;
            fs << "items" << aug->second.GetSize();
            fs << "src" << saveTo;
        }
        fs << "}";
        fs << "}";
    }
    fs << "]";
    fs << "landmarks" << "{";
    fs << "src"   << lmkPath;
    fs << "items" << GetSize0();
    fs << "}";

    fs << "hits" << "{";
    fs << "items" << hits;
    fs << "src"   << hitPath;
    fs << "}";

    fs << "keyframes" << "[";
    BOOST_FOREACH (size_t kf, m_keyframes)
    {
        fs << kf;
    }
    fs << "]";

    return true;
}

bool Map::Restore(const Path & path)
{
    // reset everything first..
    Clear();

    const Path idxPath = path / "index.yml";
    cv::FileStorage fs(idxPath.string(), cv::FileStorage::READ);
    std::ifstream is;

    if (!fs.isOpened())
    {
        E_ERROR << "error reading from " << idxPath;
        return false;
    }

    fs["sequence"] >> m_seqPath;

    Sequence seq;
    if (!seq.Restore(m_seqPath))
    {
        E_ERROR << "error restoring sequence from " << m_seqPath;
        return false;
    }

    cv::FileNode sources = fs["sources"];
    for (cv::FileNodeIterator itr = sources.begin(); itr != sources.end(); itr++)
    {
        size_t keypoints, disparity;
        (*itr)["keypoints"] >> keypoints;
        (*itr)["disparity"] >> disparity;
        
        FeatureStore::ConstOwn store = seq.GetFeatureStore(keypoints);
        DisparityStore::ConstOwn dpm = disparity == INVALID_INDEX ? DisparityStore::ConstOwn() : seq.GetDisparityStore(disparity);

        if (!store)
        {
            E_ERROR << "missing feature store" << keypoints;
            return false;
        }

        if (!dpm && disparity != INVALID_INDEX)
        {
            E_ERROR << "missing disparity store " << disparity;
            return false;
        }

        AddSource(store, dpm);
    }

    cv::FileNode frames = fs["frames"];
    for (cv::FileNodeIterator itr = frames.begin(); itr != frames.end(); itr++)
    {
        size_t index;
        (*itr)["index"] >> index;

        Frame& t = GetFrame(index);
        cv::Mat tform;

        (*itr)["pose"] >> tform;

        if (!t.pose.pose.SetTransformMatrix(tform))
        {
            E_ERROR << "error setting pose matrix";
            return false;
        }

        cv::FileNode aug = fs["augmentedFeatureSet"];
        for (cv::FileNodeIterator a = aug.begin(); a != aug.end(); a++)
        {
            size_t store;
            Path from;

            fs["store"] >> store;
            fs["src"]   >> from;

            if (!t.augmentedFeaturs[store].Restore(from))
            {
                E_ERROR << "error restoring augmented feature store from " << from;
                return false;
            }
        }
    }

    // keyframes
    cv::FileNode kf = fs["keyframes"];
    for (cv::FileNodeIterator itr = kf.begin(); itr != kf.end(); itr++)
    {
        size_t index;
        (*itr) >> index;
        m_keyframes.insert(index);
    }

    // landmarks
    cv::FileNode landmarks = fs["landmarks"];
    Path lmkPath;
    landmarks["src"] >> lmkPath;

    is.open(lmkPath.string().c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open())
    {
        E_ERROR << "error loading landmarks from " << lmkPath;
        return false;
    }

    while (!is.eof())
    {
        Landmark& l = AddLandmark();
        is.read((char*)&l.position, sizeof l.position);
        is.read((char*)&l.cov,      sizeof l.cov);
    }

    is.close();
    E_INFO << GetLandmarks() << " landmark(s) loaded from " << fullpath(lmkPath);

    // hits
    cv::FileNode hits = fs["hits"];
    Path hitPath;
    
    hits["src"] >> hitPath;
    
    is.open(hitPath.string(), std::ios::in | std::ios::binary);
    if (!is.is_open())
    {
        E_ERROR << "error loading hits from " << hitPath;
        return false;
    }

    while (!is.eof())
    {
        size_t i, j, k;
        size_t index;
        Point2F proj;

        is.read((char*)&i, sizeof i);
        is.read((char*)&j, sizeof j);
        is.read((char*)&k, sizeof k);
        is.read((char*)&index, sizeof index);
        is.read((char*)&proj,  sizeof proj );

        Landmark& l = GetLandmark(i);
        Frame&    t = GetFrame(j);
        Source&   s = GetSource(k);
        
        l.Hit(t, s, index).proj = proj;
        Landmark::Ptrs& u = t.featureLandmarkLookup[s.store->GetIndex()];
        if (u.empty())
        {
            const size_t features = (*s.store)[k].GetSize() + t.augmentedFeaturs[s.store->GetIndex()].GetSize();
            u.resize(features, NULL);
        }

        u[index] = &l; // update the feature-landmark table
    }

    return true;
}

Landmark::Ptrs Map::GetLandmarks(std::vector<size_t> indices)
{
    Landmark::Ptrs u;
    u.reserve(indices.size());

    BOOST_FOREACH (size_t idx, indices)
    {
        u.push_back(&GetLandmark(idx));
    }

    return u;
}

//==[ Landmark ]==============================================================//

Hit& Landmark::Hit(Frame& frame, Source& src, size_t index)
{
    typedef Container* dnType;
    dnType d12[2];
    d12[0] = static_cast<dnType>(&frame);
    d12[1] = static_cast<dnType>(&src);

    return Insert(::Hit(index), d12);
}

//==[ Frame ]=================================================================//

double Frame::GetCovisibility(const Frame& tj) const
{
    const Frame& ti = *this;

    Frame::const_iterator hi = ti.cbegin();
    Frame::const_iterator hj = tj.cbegin();

    size_t qi = INVALID_INDEX;
    size_t unique = 0;
    size_t shared = 0;

    while (hi && hj)
    {
        const Landmark& li = hi.GetContainer<0, Landmark>();
        const Landmark& lj = hj.GetContainer<0, Landmark>();

        if (li < lj)
        {
            if (li.GetIndex() != qi)
            {
                qi = li.GetIndex();
                unique++;
            }

            hi++;
        }
        else if (lj < li)
        {
            hj++;
        }
        else
        {
            if (li.GetIndex() != qi)
            {
                qi = li.GetIndex();
                unique++;
                shared++;
            }

            hi++;
            hj++;
        }
    }

    return static_cast<double>(shared) / static_cast<double>(unique);
}

//==[ MultiObjectiveOutlierFilter ]==========================================//

bool MultiObjectiveOutlierFilter::operator() (ImageFeatureMap& fmap, IndexList& inliers)
{
    if (builders.empty())
    {
        E_WARNING << "no objective builder available";
        return false;
    }

    //
    // build objectives
    //
    const ImageFeatureSet& Fi = fmap.From();
    const ImageFeatureSet& Fj = fmap.To();

    struct Record
    {
        Record() : refs(0), hits(0) {}

        size_t refs; // number of referenced models
        size_t hits; // number of being an inlier to the referenced models
    };

    std::vector<Record> rec(inliers.size());

    size_t idx = 0;
    BOOST_FOREACH (size_t k, inliers)
    {
        const FeatureMatch& m = fmap[k];

        BOOST_FOREACH (ObjectiveBuilder::Own builder, builders)
        {
            if (builder->Prebuilt()) continue;

            if (builder->AddData(m.srcIdx, m.dstIdx, k, Fi[m.srcIdx], Fj[m.dstIdx], idx))
            {
                rec[idx].refs++;
            }
        }

        idx++;
    }

    //
    // apply outlier detection on the built models
    //
    ConsensusPoseEstimator estimator;
    PoseEstimator::Estimate estimate;
    PoseEstimator::ConstOwn solver;
    GeometricMapping solverData;

    // to trace the population and inliers in each model
    std::vector<AlignmentObjective::InlierSelector::Stats*> stats;

    if (motion.valid)
    {
        solver = PoseEstimator::ConstOwn(new DummyPoseEstimator(motion.pose));
    }

    BOOST_FOREACH (ObjectiveBuilder::Own builder, builders)
    {
        GeometricMapping data;
        AlignmentObjective::InlierSelector selector;
        
        if (!builder->Build(data, selector, sigma))
        {
            E_WARNING << "objective \"" << builder->ToString() << "\" building failed";
            continue;
        }

        E_TRACE << "objective \"" << builder->ToString() << "\" built from " << data.GetSize() << " correspondence(s)";

        estimator.AddSelector(selector);
        stats.push_back(&builder->stats);

        if (!solver)
        {
            E_TRACE << "main solver set to \"" << builder->ToString() << "\"";

            solver = builder->GetSolver();
            solverData = data;
        }
    }

    if (!solver)
    {
        E_WARNING << "no objective succesfully built";
        return false;
    }

    estimator.SetStrategy(ConsensusPoseEstimator::RANSAC);
    estimator.SetMaxIterations(motion.valid ? 1 : maxIterations);
    estimator.SetMinInlierRatio(minInlierRatio);
    estimator.SetConfidence(confidence);
    estimator.SetSolver(solver);
    estimator.SetVerbose(true);

    if (optimisation && !motion.valid)
    {
        estimator.EnableOptimisation();
    }
    else
    {
        estimator.DisableOptimisation();
    }

    std::vector<IndexList> survived, eliminated;

    if (!estimator(solverData, estimate, survived, eliminated))
    {
        E_ERROR << "consensus outlier detection failed";
    }

    // aggregate all inliers from all the selectors

    for (size_t s = 0; s < survived.size(); s++)
    {
        const AlignmentObjective::InlierSelector& sel = estimator.GetSelectors()[s];
        const std::vector<size_t>& idmap = sel.objective->GetData().indices;
        
        /////////////////////////////////////////////////////////////////////////////////
        // std::stringstream ss; ss << "M" << s << ".bin";
        // std::stringstream ss2; ss2 << "I" << s << ".bin";
        // PersistentMat(sel.objective->operator()(estimate.pose)).Store(Path(ss.str()));
        // std::vector<int> idmap2(idmap.size());
        // for (size_t k = 0; k < idmap.size(); k++) idmap2[k] = (int) idmap[k];
        // PersistentMat(cv::Mat(idmap2, false)).Store(Path(ss2.str()));
        /////////////////////////////////////////////////////////////////////////////////

        if (stats[s] != NULL)
        {
            stats[s]->population = sel.objective->GetData().GetSize(); // idmap.size();
            stats[s]->inliers = survived[s].size();
            stats[s]->secs += sel.metre.GetElapsedSeconds();
        }

        if (idmap.empty()) continue;

        BOOST_FOREACH (size_t j, survived[s])
        {
            rec[idmap[j]].hits++;
        }
    }

    IndexList::iterator itr = inliers.begin();
    for (std::vector<Record>::const_iterator h = rec.begin(); h != rec.end(); h++)
    {
        if (h->hits < h->refs)
        {
            fmap[*itr].Reject(FeatureMatch::GEOMETRIC_TEST_FAILED);
            inliers.erase(itr++);
        }
        else
        {
            itr++;
        }
    }

    // estimate.pose might be useful..
    if (estimate.valid && !motion.valid)
    {
        motion = estimate;
    }

    return estimate.valid;
}

//==[ FeatureTracker ]=======================================================//

bool FeatureTracker::StringToAlignment(const String& flag, int& model)
{
    model = 0;

    for (size_t i = 0; i < flag.length(); i++)
    {
        switch (flag[i])
        {
        case 'B': model |= OutlierRejectionScheme::BACKWARD_PROJ_ALIGN; break;
        case 'P': model |= OutlierRejectionScheme::PHOTOMETRIC_ALIGN;   break;
        case 'R': model |= OutlierRejectionScheme::RIGID_ALIGN;         break;
        case 'E': model |= OutlierRejectionScheme::EPIPOLAR_ALIGN;      break;
        default:
            E_ERROR << "unknown model \"" << flag[i] << "\"";
            return false;
        }
    }

    return true;
}

String FeatureTracker::AlignmentToString(int model)
{
    std::stringstream ss;
    if (model & OutlierRejectionScheme::BACKWARD_PROJ_ALIGN) ss << "B";
    if (model & OutlierRejectionScheme::PHOTOMETRIC_ALIGN)   ss << "P";
    if (model & OutlierRejectionScheme::RIGID_ALIGN)         ss << "R";
    if (model & OutlierRejectionScheme::EPIPOLAR_ALIGN)      ss << "E";

    return ss.str();
}

bool FeatureTracker::StringToFlow(const String& flow, int& type)
{
    if (flow.empty())
    {
        type = 0;
        return true;
    }

    if (flow.compare("FORWARD") == 0)
    {
        type = InlierInjectionScheme::FORWARD_FLOW;
        return true;
    }

    if (flow.compare("BACKWARD") == 0)
    {
        type = InlierInjectionScheme::BACKWARD_FLOW;
        return true;
    }

    if (flow.compare("BIDIRECTION") == 0)
    {
        type = InlierInjectionScheme::FORWARD_FLOW | InlierInjectionScheme::BACKWARD_FLOW;
        return true;
    }

    E_ERROR << "unknown flow option \"" << flow << "\"";
    return false;
}

String FeatureTracker::FlowToString(int scheme)
{
    bool forward  = (scheme & InlierInjectionScheme::FORWARD_FLOW ) != 0;
    bool backward = (scheme & InlierInjectionScheme::BACKWARD_FLOW) != 0;

    if (forward && backward) return "BIDIRECTION";
    else if (forward)        return "FORWARD";
    else if (backward)       return "BACKWARD";
    else                     return "";
}

bool FeatureTracker::StringToTriangulation(const String& triangulation, TriangulationMethod& method)
{
    if (triangulation.empty())
    {
        method = TriangulationMethod::DISABLED;
        return true;
    }

    if (triangulation.compare("MIDPOINT") == 0)
    {
        method = TriangulationMethod::MIDPOINT;
        return true;
    }

    if (triangulation.compare("OPTIMAL") == 0)
    {
        method = TriangulationMethod::OPTIMAL;
        return true;
    }

    E_ERROR << "unknown triangulation method \"" << triangulation << "\"";
    method = TriangulationMethod::DISABLED;

    return false;
}

String FeatureTracker::TriangulationToString(TriangulationMethod method)
{
    switch (method)
    {
    case TriangulationMethod::MIDPOINT: return "MIDPOINT"; break;
    case TriangulationMethod::OPTIMAL:  return "OPTIMAL";  break;
    case TriangulationMethod::DISABLED: return "";         break;
    }

    E_ERROR << "unknown method " << method;
    return TriangulationToString(TriangulationMethod::DISABLED);
}

void FeatureTracker::WriteParams(cv::FileStorage& fs) const
{
    fs << "name" << GetName();

    fs << "outlierRejection" << "{";
    {
        fs << "model" << AlignmentToString(outlierRejection.model);
        fs << "confidence" << outlierRejection.confidence;
        fs << "iterations" << outlierRejection.maxIterations;
        fs << "minInliers" << outlierRejection.minInlierRatio;
        fs << "sigma" << outlierRejection.sigma;
        // fs << "epipolarEps" << outlierRejection.epipolarEps;
        fs << "fastMetric" << outlierRejection.fastMetric;
    }
    fs << "}";

    fs << "inlierInjection" << "{";
    {
        fs << "flow" << FlowToString(inlierInjection.scheme);
        fs << "blockSize" << inlierInjection.blockSize;
        fs << "levels" << inlierInjection.levels;
        // fs << "epipolarEps" << inlierInjection.epipolarEps;
        fs << "bidirectionalTol" << inlierInjection.bidirectionalTol;
    }
    fs << "}";

    fs << "triangulation" << TriangulationToString(triangulation);
    fs << "epipolarEps"   << m_epipolarEps;
}

bool FeatureTracker::ReadParams(const cv::FileNode& fn)
{
    String name;
    const cv::FileNode oj = fn["outlierRejection"];
    const cv::FileNode ij = fn["inlierInjection"];

    oj["model"]       >> m_alignString;
    oj["confidence"]  >> outlierRejection.confidence;
    oj["iterations"]  >> outlierRejection.maxIterations;
    oj["minInliers"]  >> outlierRejection.minInlierRatio;
    oj["sigma"]       >> outlierRejection.sigma;
    oj["fastMetric"]  >> outlierRejection.fastMetric;

    ij["flow"]        >> m_flowString;
    ij["blockSize"]   >> inlierInjection.blockSize;
    ij["levels"]      >> inlierInjection.levels;
    ij["bidirectionalTol"] >> inlierInjection.bidirectionalTol;

    fn["name"] >> name;
    fn["epipolarEps"]   >> m_epipolarEps;
    fn["triangulation"] >> m_triangulation;

    SetName(name);
    ApplyParams();

    return true;
}

void FeatureTracker::ApplyParams()
{
    if (!StringToAlignment(m_alignString, outlierRejection.model))
    {
        E_ERROR << "error applying alignment models \"" << m_alignString << "\"";
    }

    if (!StringToFlow(m_flowString, inlierInjection.scheme))
    {
        E_ERROR << "error applying flow-based inlier injection setting";
    }

    if (!StringToTriangulation(m_triangulation, triangulation))
    {
        E_ERROR << "error applying triangulation setting";
    }

    outlierRejection.model |= OutlierRejectionScheme::FORWARD_PROJ_ALIGN;
    outlierRejection.epipolarEps = inlierInjection.epipolarEps = m_epipolarEps;

    Strings models;

    if (outlierRejection.model & OutlierRejectionScheme::FORWARD_PROJ_ALIGN)  models.push_back("FORWARD-PROJECTION");
    if (outlierRejection.model & OutlierRejectionScheme::BACKWARD_PROJ_ALIGN) models.push_back("BACKWARD-PROJECTION");
    if (outlierRejection.model & OutlierRejectionScheme::RIGID_ALIGN)         models.push_back("RIGID");
    if (outlierRejection.model & OutlierRejectionScheme::PHOTOMETRIC_ALIGN)   models.push_back("PHOTOMETRIC");
    if (outlierRejection.model & OutlierRejectionScheme::EPIPOLAR_ALIGN)      models.push_back("EPIPOLAR");

    E_INFO << "[" << GetName() << "]";
    E_INFO << "enabled alignment model : " << boost::algorithm::join(models, ", ");
    E_INFO << "optical flow direction  : " << (m_flowString.empty() ? "DISABLED" : m_flowString);
    E_INFO << "triangulation method    : " << TriangulationToString(triangulation);
    E_INFO << "RANSAC iterations       : " << outlierRejection.maxIterations;
    E_INFO << "RANSAC confidence       : " << (100.0f * outlierRejection.confidence) << "%";
    E_INFO << "minimum inlier ratio    : " << (100.0f * outlierRejection.minInlierRatio) << "%";
    E_INFO << "inlier sigma            : " << outlierRejection.sigma;
    E_INFO << "fast metric evaluation  : " << (outlierRejection.fastMetric ? "YES" : "NO");
    E_INFO << "flow bidirectional tol. : " << inlierInjection.bidirectionalTol << " pixel(s)";
    E_INFO << "epipolar tolerance      : " << (1/m_epipolarEps) << " normalised pixel(s)";
}

FeatureTracker::Options FeatureTracker::GetOptions(int flag)
{
    namespace po = boost::program_options;
    Options o;

    OutlierRejectionOptions& oj = outlierRejection;
    InlierInjectionOptions&  ij = inlierInjection;

    o.add_options()
        ("align-model,a",  po::value<String>(&m_alignString    )->default_value(   ""), "A string containning flags of alignment models to activate. \"B\" for backward projection, \"P\" for photometric, \"R\" for rigid, and \"E\" for epipolar alignment.")
        ("triangulation",  po::value<String>(&m_triangulation  )->default_value(   ""), "Pose-motion triangulation method; valid strings are \"MIDPOINT\" and \"OPTIMAL\". Set to empty string to disable the feature.")
        ("epipolar-eps",   po::value<double>(&m_epipolarEps    )->default_value( 1000), "Threshold for epipolar constraint, as the inverse of the tolerable epipolar distance in normalised pixel. Ths value must be positive.")
        ("sigma",          po::value<double>(&oj.sigma         )->default_value(1.00f), "Threshold for inlier selection. The value must be positive.")
        ("ransac-iter",    po::value<size_t>(&oj.maxIterations )->default_value(  100), "Max number of iterations for the RANSAC outlier rejection process.")
        ("ransac-conf",    po::value<double>(&oj.confidence    )->default_value(0.99f), "The confidence of obtaining a valid estimate at the end of the RANSAC process. The value is used to calculate a required iteration number.")
        ("ransac-ratio",   po::value<double>(&oj.minInlierRatio)->default_value(0.70f), "Minimum ratio of inliers required to consider a hypothesis valid.")
        ("flow",           po::value<String>(&m_flowString     )->default_value(   ""), "Optical flow computation option for missed features. Valid strings are \"FORWARD\", \"BACKWARD\" and \"BIDIRECTION\"")
        ("flow-level",     po::value<size_t>(&ij.levels        )->default_value(    3), "Level of pyramid for optical flow computation.")
        ("flow-bidir-tol", po::value<double>(&ij.bidirectionalTol)->default_value(  1), "Threshold of the forward-backward flow error, in image pixels. Set to a non-positive value to disable the test.")
        ("block-size",     po::value<size_t>(&ij.blockSize     )->default_value(    5), "Block size for optical flow computation and epipolar search.")
        ("fast-metric",    po::bool_switch  (&oj.fastMetric    )->default_value(false), "Apply metric reduction to accelerate error evaluation.")
        ;

    return o;
}

StructureEstimation::Estimate FeatureTracker::GetFeatureStructure(const ImageFeatureSet& f, const StructureEstimation::Estimate& structure)
{
    return structure[GetFeatureImageIndices(f, structure.structure.mat.size())];
}

IndexList FeatureTracker::GetFeatureImageIndices(const ImageFeatureSet& f, const cv::Size& imageSize) const
{
    IndexList indices;

    // convert subscripts to rounded integer indices
    for (size_t k = 0; k < f.GetSize(); k++)
    {
        const Point2F& sub = f[k].keypoint.pt;
        const int i = static_cast<int>(std::round(sub.y));
        const int j = static_cast<int>(std::round(sub.x));

        if (i < 0 || i >= imageSize.height || j < 0 || j >= imageSize.width)
        {
            indices.push_back(INVALID_INDEX);
            E_ERROR << "subscript (" << i << "," << j << ") out of bound";

            continue;
        }

        indices.push_back(static_cast<size_t>(i * imageSize.width + j));
    }

    return indices;
}

bool FeatureTracker::operator() (Map& map, Source& si, Frame& ti, Source& sj, Frame& tj)
{
    assert(si.store && sj.store);

    const FeatureStore& Fi = *si.store;
    const FeatureStore& Fj = *sj.store;
    ImageFeatureSet fi = Fi[ti.GetIndex()];
    ImageFeatureSet fj = Fj[tj.GetIndex()];
    Landmark::Ptrs& ui = ti.featureLandmarkLookup[Fi.GetIndex()];
    Landmark::Ptrs& uj = tj.featureLandmarkLookup[Fj.GetIndex()];

    Camera::ConstOwn ci = Fi.GetCamera();
    Camera::ConstOwn cj = Fj.GetCamera();
    boost::shared_ptr<PosedProjection> pi = ci ? ci->GetPosedProjection() : boost::shared_ptr<PosedProjection>();
    boost::shared_ptr<PosedProjection> pj = cj ? cj->GetPosedProjection() : boost::shared_ptr<PosedProjection>();
    cv::Mat Ii = ci ? ci->GetImageStore()[ti.GetIndex()].im : cv::Mat();
    cv::Mat Ij = cj ? cj->GetImageStore()[tj.GetIndex()].im : cv::Mat();
    cv::Mat Di = si.dpm ? (*si.dpm)[ti.GetIndex()].im : cv::Mat();
    cv::Mat Dj = sj.dpm ? (*sj.dpm)[tj.GetIndex()].im : cv::Mat();

    PoseEstimator::Estimate& mi = ti.pose;
    PoseEstimator::Estimate& mj = tj.pose;
    StructureEstimation::Estimate gi(Geometry::ROW_MAJOR);
    StructureEstimation::Estimate gj(Geometry::ROW_MAJOR);

    boost::shared_ptr<MultiObjectiveOutlierFilter> filter;
    GeometricMapping::ImageToImageBuilder flow;

    // initialise statistics
    stats = Stats();

    // append augmented feature sets
    fi.Append(ti.augmentedFeaturs[Fi.GetIndex()]);
    fj.Append(tj.augmentedFeaturs[Fj.GetIndex()]);

    // initialise frame's feature-landmark lookup table for first time access
    if (ui.empty()) ui.resize(fi.GetSize(), NULL);
    if (uj.empty()) uj.resize(fj.GetSize(), NULL);

    // get feature structure from depthmap and transform it to the reference camera's coordinates system
    if (!Di.empty()) gi = si.dpm->GetStereoPair()->Backproject(Di, cv::Mat(), GetFeatureImageIndices(fi, Di.size())).Transform(ci->GetExtrinsics().GetInverse());
    if (!Dj.empty()) gj = sj.dpm->GetStereoPair()->Backproject(Dj, cv::Mat(), GetFeatureImageIndices(fj, Dj.size())).Transform(cj->GetExtrinsics().GetInverse());

    // perform pre-motion structure update
    if (mi.valid) gi = map.UpdateStructure(ui, gi, mi.pose);
    if (mj.valid) gj = map.UpdateStructure(uj, gj, mj.pose);

    // build the outlier filter and models
    if (outlierRejection.model)
    {
        filter = boost::shared_ptr<MultiObjectiveOutlierFilter>(
            new MultiObjectiveOutlierFilter(outlierRejection.maxIterations, outlierRejection.minInlierRatio, outlierRejection.confidence, outlierRejection.sigma)
        );

        if (ti == tj)
        {
            filter->motion.pose = EuclideanTransform::Identity;
            filter->motion.valid = true;
        }
        else if (ti.pose.valid && tj.pose.valid)
        {
            filter->motion.pose = mi.pose.GetInverse() >> mj.pose;
            filter->motion.valid = true;

            ////////////////////////////////////////////////////////////////////////////////////////
            // E_TRACE << GetName() << " uses motion " << ti.GetIndex() << " -> " << tj.GetIndex()";
            // E_TRACE << mat2string(filter->motion.pose.GetTransformMatrix(),"M");
            ////////////////////////////////////////////////////////////////////////////////////////
        }

        if (outlierRejection.model & FORWARD_PROJ_ALIGN)
        {
            if (pj)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                    new PerspectiveObjectiveBuilder(pj, gi, true, outlierRejection.fastMetric, stats.objectives[FORWARD_PROJ_ALIGN])
                ));
            }
            else
            {
                E_WARNING << "error adding forward projection objective to the outlier filter due to missing projection model";
                E_WARNING << "forward projection-based outlier rejection deactivated"; // << ToString();

                outlierRejection.model &= ~FORWARD_PROJ_ALIGN;
            }
        }

        if (outlierRejection.model & BACKWARD_PROJ_ALIGN)
        {
            if (pi)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                    new PerspectiveObjectiveBuilder(pi, gj, false, outlierRejection.fastMetric, stats.objectives[BACKWARD_PROJ_ALIGN])
                ));
            }
            else
            {
                E_WARNING << "error adding backward projection objective to the outlier filter due to missing projection model";
                E_WARNING << "backward projection-based outlier rejection deactivated"; // << ToString();

                outlierRejection.model &= ~BACKWARD_PROJ_ALIGN;
            }
        }

        if (outlierRejection.model & PHOTOMETRIC_ALIGN)
        {
            if (pj)
            {
                MultiObjectiveOutlierFilter::ObjectiveBuilder::Own builder =
                    MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                        new PhotometricObjectiveBuilder(pj, gi, Ii, Ij, outlierRejection.fastMetric, stats.objectives[PHOTOMETRIC_ALIGN])
                    );

                for (size_t k = 0; k < fi.GetSize(); k++)
                {
                    ImageFeature fi_k = fi[k];
                    builder->AddData(
                        k,
                        INVALID_INDEX, // not used
                        INVALID_INDEX, // not used
                        fi_k,
                        fi_k,          // not used
                        INVALID_INDEX  // not used
                    );
                }

                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(builder));
            }
            else
            {
                E_WARNING << "error adding photometric objective to the outlier filter due to missing projection model";
                E_WARNING << "photometric outlier rejection deactivated"; // << ToString();

                outlierRejection.model &= ~PHOTOMETRIC_ALIGN;
            }
        }

        if (outlierRejection.model & RIGID_ALIGN)
        {
            filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                new RigidObjectiveBuilder(gi, gj, outlierRejection.fastMetric, stats.objectives[RIGID_ALIGN])
            ));
        }

        if (outlierRejection.model & EPIPOLAR_ALIGN)
        {
            if (pi && pj)
            {
                filter->builders.push_back(MultiObjectiveOutlierFilter::ObjectiveBuilder::Own(
                    new EpipolarObjectiveBuilder(pi, pj, outlierRejection.epipolarEps, stats.objectives[EPIPOLAR_ALIGN])
                ));
            }
            else
            {
                E_WARNING << "error adding epipolar objective to the outlier filter due to missing projection model(s)";
                E_WARNING << "epipolar-based outlier rejection deactivated"; // << ToString();

                outlierRejection.model &= ~EPIPOLAR_ALIGN;
            }
        }

        matcher.GetFilters().push_back(FeatureMatcher::Filter::Own(filter));
    }

    ImageFeatureMap fmap = matcher(fi, fj);

    // dispose the outlier filter and apply the solved ego-motion whenever useful
    if (outlierRejection.model)
    {
        matcher.GetFilters().pop_back();

        if (!filter->motion.valid)
        {
            E_ERROR << "tracker \"" << GetName() << "\" failed egomotion estimation";
            return false;
        }

        ////////////////////////////////////////////////////////////////////////////////////////
        // E_TRACE << GetName() << " solved motion " << ti.GetIndex() << " -> " << tj.GetIndex();
        // E_TRACE << mat2string(filter->motion.pose.GetTransformMatrix(), "M");
        ////////////////////////////////////////////////////////////////////////////////////////

        stats.motion = filter->motion;

        if (ti != tj)
        {
            if (mi.valid && !mj.valid)
            {
                mj.pose  = mi.pose >> filter->motion.pose;
                mj.valid = true;
            }

            if (mj.valid && !mi.valid) // i.e. "else if (mj.valid) .."
            {
                mi.pose  = mj.pose >> filter->motion.pose.GetInverse();
                mi.valid = true;
            }
        }
    }

    // insert the observations into the map
    std::vector<bool> qi(fi.GetSize());
    std::vector<bool> qj(fj.GetSize());
    const FeatureMatches& matches = fmap.GetMatches();

    // E_INFO << matches.size() << " matches..";

    for (size_t k = 0; k < matches.size(); k++)
    {
        const FeatureMatch& m = matches[k];

        if (!(m.state & FeatureMatch::INLIER))
        {
            continue;
        }

        qi[m.srcIdx] = qj[m.dstIdx] = true;

        Landmark*& ui_k = ui[m.srcIdx];
        Landmark*& uj_k = uj[m.dstIdx];

        bool bi = (ui_k == NULL);
        bool bj = (uj_k == NULL);

        bool firstHit = bi == true && bj == true;
        bool converge = bi != true && bj != true && ui_k != uj_k;

        if (!converge)
        {
            Landmark& lk = firstHit ? map.AddLandmark() : (bj ? *ui_k : *uj_k);

            if (bi) lk.Hit(ti, si, m.srcIdx).proj = fi[m.srcIdx].keypoint.pt;
            if (bj) lk.Hit(tj, sj, m.dstIdx).proj = fj[m.dstIdx].keypoint.pt;

            ui_k = uj_k = &lk;

            if (firstHit)
            {
                stats.spawned++;
            }
            else
            {
                stats.tracked++;
            }

            flow.Add(fi[m.srcIdx].keypoint.pt, fj[m.dstIdx].keypoint.pt, lk.GetIndex());

            continue;
        }

        if (policy == ConflictResolution::NO_MERGE)
        {
            continue;
        }

        Landmark& li = *ui_k;
        Landmark& lj = *uj_k;

        if (policy == ConflictResolution::KEEP_BOTH || map.IsJoinable(li, lj))
        {
            map.JoinLandmark(li, lj);
            stats.joined++;

            continue;
        }

        switch (policy)
        {
        case ConflictResolution::KEEP_BEST:

            throw std::exception("not implemented");
            break;

        case ConflictResolution::KEEP_LONGEST:

            throw std::exception("not implemented");
            break;

        case ConflictResolution::KEEP_SHORTEST:

            throw std::exception("not implemented");
            break;

        case ConflictResolution::REMOVE_BOTH:

            map.RemoveLandmark(li);
            map.RemoveLandmark(lj);

            stats.removed += 2;

            break;
        }
    }

    // initialise structure for the newly spawned landmarks
    // if (mi.valid && !gi.structure.IsEmpty()) map.UpdateStructure(ai.u, gi[ai.f], mi.pose);
    // if (mj.valid && !gj.structure.IsEmpty()) map.UpdateStructure(aj.u, gj[aj.f], mj.pose);

    // post-motion correspondence recovery
    if (inlierInjection.scheme && stats.motion.valid)
    {
        GeometricMapping forward, backward, epipolar;

        if (inlierInjection.scheme & InlierInjectionScheme::FORWARD_FLOW)
        {
            if (pi && pj)
            {
                AlignmentObjective::Own eval(new EpipolarObjective(pi, pj));
                forward = FindFeaturesFlow(Ii, Ij, fi, eval, stats.motion.pose, qi);
            }
            else
            {
                E_WARNING << "projection(s) missing for optical flow feature recovery";
                E_WARNING << "forward flow-based inlier injection now deactivated.";

                inlierInjection.scheme &= ~InlierInjectionScheme::FORWARD_FLOW;
            }
        }

        if (inlierInjection.scheme & InlierInjectionScheme::BACKWARD_FLOW)
        {
            if (pi && pj)
            {
                AlignmentObjective::Own eval(new EpipolarObjective(pj, pi));
                backward = FindFeaturesFlow(Ij, Ii, fj, eval, stats.motion.pose.GetInverse(), qj);
            }
            else
            {
                E_WARNING << "projection(s) missing for optical flow feature recovery";
                E_WARNING << "backward flow-based inlier injection now deactivated.";

                inlierInjection.scheme &= ~InlierInjectionScheme::BACKWARD_FLOW;
            }
        }

        if (inlierInjection.scheme & InlierInjectionScheme::EPIPOLAR_SEARCH)
        {
            // do something..
        }

        if (!AugmentFeatures(forward, fi, fj, map, ti, tj, si, sj, ui, uj, tj.augmentedFeaturs[Fj.GetIndex()], stats.spawned, stats.tracked))
        {
            E_ERROR << "error augmenting feature set from forward flow";
            return false;
        }

        if (!AugmentFeatures(backward, fj, fi, map, tj, ti, sj, si, uj, ui, ti.augmentedFeaturs[Fi.GetIndex()], stats.spawned, stats.tracked))
        {
            E_ERROR << "error augmenting feature set from backward flow";
            return false;
        }

        for (size_t i = 0; i < forward.GetSize(); i++)
        {
            flow.Add(
                forward.src.mat.reshape(2).at<Point2D>(static_cast<int>(i)),
                forward.dst.mat.reshape(2).at<Point2D>(static_cast<int>(i)),
                ui[forward.indices[i]]->GetIndex()
            );
        }

        for (size_t j = 0; j < backward.GetSize(); j++)
        {
            flow.Add(
                backward.dst.mat.reshape(2).at<Point2D>(static_cast<int>(j)),
                backward.src.mat.reshape(2).at<Point2D>(static_cast<int>(j)),
                uj[backward.indices[j]]->GetIndex()
            );
        }

        stats.injected += forward.GetSize() + backward.GetSize();
    }

    stats.flow = flow.Build();
    stats.accumulated = map.GetLandmarks();

    if (triangulation != DISABLED && mi.valid && stats.motion.valid)
    {
        map.UpdateStructure(
            map.GetLandmarks(stats.flow.indices),
            triangulation == TriangulationMethod::OPTIMAL ? 
                OptimalTriangulation (*pi, *pj, stats.motion)(stats.flow) :
                MidPointTriangulation(*pi, *pj, stats.motion)(stats.flow),
            mi.pose
        );
    }

    if (rendering)
    {
        stats.Render(imfuse(Ii, Ij), GetName(), map, mi.pose);

        std::stringstream ss;
        cv::Mat im;

        ss << GetName() << "." << std::setw(5) << std::setfill('0') << ti.GetIndex() << ".jpg";
        cv::imwrite(ss.str(), stats.im);
        //cv::vconcat(stats.im, fmap.Draw(Ii, Ij), im);
        //cv::imwrite(ss.str(), im);

        cv::imshow("Feature Tracking [" + GetName() + "]", stats.im);
        cv::waitKey(1);
    }

    return true;
}

GeometricMapping FeatureTracker::FindFeaturesFlow(const cv::Mat& Ii, const cv::Mat& Ij, const ImageFeatureSet& fi, AlignmentObjective::Own& eval, const EuclideanTransform& pose, std::vector<bool>& tracked)
{
    assert(fi.GetSize() == tracked.size());
    assert(eval);

    // cv::Mat im = imfuse(Ii, Ij);

    struct Flow
    {
        Points2F xi;
        Points2F xj;
        std::vector<size_t> idx;
        std::vector<uchar> found;
        cv::Mat err;
    };

    Flow forward;

    for (size_t k = 0; k < tracked.size(); k++)
    {
        if (tracked[k]) continue;

        // cv::circle(im, fi[k].keypoint.pt, 2, cv::Scalar(0, 0, 255), -1);

        forward.xi.push_back(fi[k].keypoint.pt);
        forward.idx.push_back(k);
    }

    const int bs = static_cast<int>(inlierInjection.blockSize);
    cv::Size win(bs, bs);
    cv::calcOpticalFlowPyrLK(Ii, Ij, forward.xi, forward.xj, forward.found, forward.err, win, static_cast<int>(inlierInjection.levels));

    for (size_t i = 0; i < forward.found.size(); i++)
    {
        const double x = round(forward.xj[i].x);
        const double y = round(forward.xj[i].y);

        if (x < 0 || x >= Ij.cols || y < 0 || y >= Ij.rows)
        {
            forward.found[i] = false;
        }
    }

    if (inlierInjection.bidirectionalTol > 0)
    {
        Flow backward;

        for (size_t i = 0; i < forward.found.size(); i++)
        {
            if (!forward.found[i]) continue;

            backward.xj.push_back(forward.xj[i]);
            backward.idx.push_back(i);

            // cv::line(im, forward.xi[i], forward.xj[i], cv::Scalar(127, 127, 127));
            // cv::circle(im, forward.xj[i], 2, cv::Scalar(255, 0, 0), -1);
        }

        cv::calcOpticalFlowPyrLK(Ij, Ii, backward.xj, backward.xi, backward.found, backward.err, win, static_cast<int>(inlierInjection.levels));
        const double tol2 = inlierInjection.bidirectionalTol * inlierInjection.bidirectionalTol;

        for (size_t j = 0; j < backward.found.size(); j++)
        {
            const size_t i = backward.idx[j];

            if (!backward.found[j])
            {
                forward.found[i] = false;
                continue;
            }

            // cv::circle(im, backward.xi[j], 2, cv::Scalar(0, 127, 255), -1);

            const double dx = forward.xi[i].x - backward.xi[j].x;
            const double dy = forward.xi[i].y - backward.xi[j].y;
            
            forward.found[i] = (dx * dx + dy * dy) < tol2;

            // cv::line(im, forward.xj[i], backward.xi[j], forward.found[i] ? cv::Scalar(192, 64, 64) : cv::Scalar(64, 64, 192));
        }
    }

    GeometricMapping::ImageToImageBuilder builder;
    GeometricMapping mapping;
    IndexList inliers;
    // std::vector<size_t> trace;

    for (size_t i = 0; i < forward.found.size(); i++)
    {
        if (!forward.found[i]) continue;

        builder.Add(forward.xi[i], forward.xj[i], forward.idx[i]);

        // trace.push_back(i);
        // cv::line(im, forward.xi[i], forward.xj[i], cv::Scalar(255, 255, 255));
    }

    mapping = builder.Build();
    mapping.metric = Metric::Own(new EuclideanMetric(inlierInjection.epipolarEps));
    
    if (!eval->SetData(mapping))
    {
        E_ERROR << "error setting geometric mapping for epipolar alignment verification";
        return GeometricMapping();
    }
   
    if (!eval->GetSelector(outlierRejection.sigma)(pose, inliers))
    {
        E_ERROR << "error evaluating epipolar error of the computed flow";
        return GeometricMapping();
    }

    BOOST_FOREACH (size_t idx, inliers)
    {
        tracked[mapping.indices[idx]] = true;
        // cv::line(im, forward.xi[trace[idx]], forward.xj[trace[idx]], cv::Scalar(0, 255, 0));
    }

    // cv::imshow("Lost Feature Flow", im);
    // cv::waitKey(0);

    return mapping[inliers];
}

bool FeatureTracker::AugmentFeatures(
    const GeometricMapping flow, const ImageFeatureSet& fi, const ImageFeatureSet& fj,
    Map& map, Frame& ti, Frame& tj, Source& si, Source& sj,
    Landmark::Ptrs& ui, Landmark::Ptrs& uj, ImageFeatureSet& aj, size_t& spawned, size_t& tracked)
{
    if (flow.GetSize() == 0)
    {
        return true;
    }

    const size_t n = uj.size();
    KeyPoints keypoints;
    cv::Mat descriptors;

    const cv::Mat xj = flow.dst.mat.reshape(2);

    uj.resize(n + flow.GetSize(), NULL);

    for (size_t k = 0; k < flow.GetSize(); k++)
    {
        const size_t i = flow.indices[k];
        const size_t j = n + k;

        Landmark*& ui_k = ui[i];
        Landmark*& uj_k = uj[j];
        cv::KeyPoint kp = fi[i].keypoint;

        if (ui_k == NULL)
        {
            (ui_k = &map.AddLandmark())->Hit(ti, si, i).proj = kp.pt;
            spawned++;
        }
        else
        {
            tracked++;
        }

        kp.pt = xj.at<Point2D>(static_cast<int>(k));
        (uj_k = ui_k)->Hit(tj, sj, j).proj = kp.pt;

        keypoints.push_back(kp);
    }

    FeatureExtractor::ConstOwn xtor = boost::dynamic_pointer_cast<const FeatureExtractor, const FeatureDetextractor>(si.store->GetFeatureDetextractor());
    const bool copyDesc = true;

    if (inlierInjection.extractDescriptor && xtor)
    {
        // ImageFeatureSet aj = xtor->ExtractFeatures(Ij, keypoints);
        //
        // if (aj.GetSize() == forward.GetSize())
        // {
        //    descriptors = aj.GetDescriptors().clone();
        //    copyDesc = false;
        // }
        // else
        // {
        //   E_ERROR << "descriptor extraction returns " << aj.GetSize() << " element(s), while given " << keypoints.size();
        // }
    }

    if (copyDesc)
    {
        const cv::Mat src = fi.GetDescriptors();
        descriptors = cv::Mat(static_cast<int>(flow.GetSize()), src.cols, src.type());

        for (size_t k = 0; k < flow.GetSize(); k++)
        {
            src.row(static_cast<int>(flow.indices[k])).copyTo(descriptors.row(static_cast<int>(k)));
        }
    }

    if (!aj.Append(ImageFeatureSet(keypoints, descriptors, fj.GetNormType())))
    {
        E_FATAL << "error augmenting feature set for frame " << tj.GetIndex() << " store " << sj.store->GetIndex();
        return false;
    }

    return true;
}

//==[ FeatureTracker::EpipolarOutlierModel ]=================================//

String FeatureTracker::Stats::ToString() const
{
    std::stringstream ss;
    ss << spawned     << " spawned, "
       << tracked     << " tracked, "
       << injected    << " injected, "
       << removed     << " removed, "
       << joined      << " joined, "
       << accumulated << " accumulated";

    return ss.str();
}

void FeatureTracker::Stats::Render(const cv::Mat& canvas, String& tracker, Map& map, const EuclideanTransform& tform)
{
    cv::Mat overlay = cv::Mat::zeros(canvas.rows, canvas.cols, canvas.type());
    ColourMap cmap(255);
    const int fontFace = cv::FONT_HERSHEY_PLAIN;
    const double fontScale = 0.8f;
    const cv::Scalar fontColour(230, 230, 230);
    const int thickness = 1;

    const cv::Mat src = flow.src.mat.reshape(2);
    const cv::Mat dst = flow.dst.mat.reshape(2);

    for (size_t i = 0; i < flow.GetSize(); i++)
    {
        static const double zmin = 0;
        static const double zmax = 100;
        const size_t k = flow.indices[i];

        const Landmark& lk = map.GetLandmark(k);
        Point3D gk = lk.position;
        double  wk = sqrt(lk.cov.xx + lk.cov.yy + lk.cov.zz);

        double zk = tform(gk).z;
        Point2D xk = src.at<Point2D>(static_cast<int>(i));
        Point2D yk = dst.at<Point2D>(static_cast<int>(i));
        int mk = wk > 0 ? 5 * (1 / (1 + wk)) + 1 : 1;

        const double d = std::sqrt((xk.x - yk.x) * (xk.x - yk.x) + (xk.y - yk.y) * (xk.y - yk.y));
        cv::line(overlay, xk, yk, cmap.GetColour(zk, zmin, zmax), std::min(mk, 5));
    }

    cv::add(canvas, 0.9f * overlay, im);

    std::stringstream ss;
    cv::Point pt(2, 16);

    // ss << "[" << tracker << "]";
    // PutTextLine(im, ss.str(), fontFace, fontScale, fontColour, thickness, pt);
    // ss.str("");

    ss << "Stats: " << ToString();
    PutTextLine(im, ss.str(), fontFace, fontScale, fontColour, thickness, pt);
    ss.str("");

    BOOST_FOREACH (FeatureTracker::Stats::ObjectiveStats::value_type pair, objectives)
    {
        ss << "Outlier model " << std::setw(2) << pair.first << ": "
            << std::setw(5) << pair.second.inliers << " / " << std::setw(5) << pair.second.population
            << " (" << pair.second.secs << " secs)";
        PutTextLine(im, ss.str(), fontFace, fontScale, fontColour, thickness, pt);
        ss.str("");
    }
}

void FeatureTracker::Stats::PutTextLine(cv::Mat im, const String& text, int face, double scale, const cv::Scalar& colour, int thickness, cv::Point& pt)
{
    const int margin = 2;
    int baseline;

    cv::Size box = cv::getTextSize(text, face, scale, thickness, &baseline);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(box.width, -box.height-margin), cv::Scalar(0, 0, 96), cv::FILLED);
    cv::putText(im, text, pt, face, scale, colour);

    pt.y += box.height + margin * 2;
}

//==[ FeatureTracker::EpipolarOutlierModel ]=================================//

bool FeatureTracker::EpipolarObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    m_builder.Add(fi.keypoint.pt, fj.keypoint.pt, localIdx);
    return true;
}

bool FeatureTracker::EpipolarObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new EpipolarObjective(pi, pj));
    
    data = m_builder.Build();
    data.metric = Metric::Own(new EuclideanMetric(epsilon));
    
    if (!objective->SetData(data))
    {
        E_WARNING << "error setting epipolar constraints for \"" << ToString() << "\"";
        return false;
    }

    selector = objective->GetSelector(sigma);
    return true;
}

//==[ FeatureTracker::ProjectiveOutlierModel ]===============================//

bool FeatureTracker::PerspectiveObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    if (g.structure.IsEmpty()) return false;

    const Point3D& gk = g.structure.mat.reshape(3).at<Point3D>(static_cast<int>(forward ? i : j));
    const Point2F& pk = forward ? fj.keypoint.pt : fi.keypoint.pt;

    if (gk.z > 0)
    {
        m_builder.Add(gk, pk, localIdx);
        m_idx.push_back(forward ? i : j);

        return true;
    }

    return false;
}

bool FeatureTracker::PerspectiveObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new ProjectionObjective(p, forward));

    data = m_builder.Build();
    data.metric = g.metric ? (*g.metric)[m_idx] : Metric::Own();

    if (reduceMetric && data.metric)
    {
        data.metric = data.metric->Reduce();
    }

    if (!objective->SetData(data))
    {
        E_WARNING << "error setting perspective constraints for \"" << ToString() << "\"";
        return false;
    }

    selector = objective->GetSelector(sigma);
    return true;
}

PoseEstimator::Own FeatureTracker::PerspectiveObjectiveBuilder::GetSolver() const
{
    PoseEstimator::Own estimator = PoseEstimator::Own(new PerspevtivePoseEstimator(p));
    return forward ? estimator : PoseEstimator::Own(new InversePoseEstimator(estimator));
}

//==[ FeatureTracker::PhotometricObjectiveBuilder ]=========================//

bool FeatureTracker::PhotometricObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    const Point3D& gi_k = gi.structure.mat.reshape(3).at<Point3D>(static_cast<int>(i));

    if (gi_k.z > 0)
    {
        m_idx.push_back(i);
        m_localIdx.push_back(localIdx);
        m_imagePoints.push_back(fi.keypoint.pt);

        return true;
    }

    return false;
}

bool FeatureTracker::PhotometricObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    boost::shared_ptr<PhotometricObjective> objective =
        boost::shared_ptr<PhotometricObjective>(new PhotometricObjective(pj, Ij));

    StructureEstimation::Estimate g = gi[m_idx];
    Geometry p(Geometry::PACKED, cv::Mat(m_imagePoints, false));

    if (!objective->SetData(g.structure, p, Ii, false, reduceMetric && g.metric ? g.metric->Reduce() : g.metric/*, m_localIdx*/))
    {
        E_WARNING << "error setting photometric constraints for \"" << ToString() << "\"";
        return false;
    }

    data = objective->GetData();
    selector = objective->GetSelector(sigma);

    return true;
}

//==[ FeatureTracker::RigidObjectiveBuilder ]===============================//

bool FeatureTracker::RigidObjectiveBuilder::AddData(size_t i, size_t j, size_t k, const ImageFeature& fi, const ImageFeature& fj, size_t localIdx)
{
    if (gi.structure.IsEmpty() || gj.structure.IsEmpty()) return false;

    const Point3D& gi_k = gi.structure.mat.reshape(3).at<Point3D>(static_cast<int>(i));
    const Point3D& gj_k = gj.structure.mat.reshape(3).at<Point3D>(static_cast<int>(j));

    if (gi_k.z > 0 && gj_k.z > 0)
    {
        m_builder.Add(gi_k, gj_k, localIdx);
        m_idx0.push_back(i);
        m_idx1.push_back(j);

        return true;
    }

    return false;
}

bool FeatureTracker::RigidObjectiveBuilder::Build(GeometricMapping& data, AlignmentObjective::InlierSelector& selector, double sigma)
{
    AlignmentObjective::Own objective = AlignmentObjective::Own(new RigidObjective());

    data = m_builder.Build();
    //data.metric = gi.metric && gj.metric ?  Metric::Own(new DualMetric((*gi.metric)[m_idx0], (*gj.metric)[m_idx1])) : Metric::Own();
    data.metric = gi.metric ?  Metric::Own((*gi.metric)[m_idx0]) : Metric::Own();

    if (!objective->SetData(data))
    {
        E_WARNING << "error setting rigid constraints for \"" << ToString() << "\"";
        return false;
    }

    selector = objective->GetSelector(sigma);

    return true;
}

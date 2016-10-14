#include <fstream>
#include <queue>
#include <seq2map/solve.hpp>
#include "calibgraph.hpp"
#include "calibgraphbundler.hpp"

// a helper class to find the maximum spanning tree from a given cost matrix
class MinimumSpanningTree
{
public:
    struct Edge
    {
        Edge(size_t p = INVALID_INDEX, size_t q = INVALID_INDEX, float cost = -1);

        size_t p;
        size_t q;
        float  cost;
    };

    struct CompareEdges
    {
        bool operator()(Edge const& e0, Edge const& e1) { return e0.cost > e1.cost; }
    };

    //struct Trace
    //{
    //    Trace(size_t edgeIdx, bool reverse) : edgeIdx(edgeIdx), reverse(reverse) {}
    //    size_t edgeIdx;
    //    bool   reverse;
    //};

    typedef std::vector<Edge> Edges;
    //typedef std::list<Trace>  Path;

    MinimumSpanningTree(const cv::Mat& costs, size_t start);
    virtual ~MinimumSpanningTree() {}

    std::vector<Indices> traces;

    Edges edges;
    bool valid;
};

MinimumSpanningTree::Edge::Edge(size_t p, size_t q, float cost)
{
    if (p < q)
    {
        this->p = p;
        this->q = q;
    }
    else
    {
        this->p = q;
        this->q = p;
    }

    this->cost = cost;
}

MinimumSpanningTree::MinimumSpanningTree(const cv::Mat& costs, size_t start)
{
    assert(costs.rows == costs.cols); // TODO: check symmetry of the cost matrix

    size_t n = costs.rows;

    edges.reserve(n - 1);
    traces.clear();
    traces.resize(n);

    E_INFO << "calculating minimum spanning tree from " << n << "-camera calibration graph";
    E_INFO << mat2string(costs, "cost");

    //
    // Prim's algorithm
    //
    const size_t NiL = INVALID_INDEX;
    std::priority_queue<Edge, Edges, CompareEdges> Q; // priority queue storing the explored edges
    std::vector<size_t> pre(n, NiL);

    // establish the initial edge set
    for (size_t q = 0; q < n; q++)
    {
        if (q != start) Q.push(Edge(start, q, costs.at<float>(start, q)));
    }

    // discover and build the spanning tree
    while (!Q.empty())
    {
        Edge e = Q.top();
        Q.pop();

        bool p_visited = (e.p == start || pre[e.p] != NiL);
        bool q_visited = (e.q == start || pre[e.q] != NiL);

        assert(p_visited || q_visited);

        // ignore the edge if adding it leads to loop
        if (p_visited && q_visited) continue;

        // make sure we traverse in the right direction of p->q
        size_t p, q;
        
        if (p_visited)
        {
            p = e.p;
            q = e.q;
        }
        else
        {
            p = e.q;
            q = e.p;
        }

        edges.push_back(e);
        pre[q] = p;

        for (size_t r = 0; r < n; r++)
        {
            if (r != q) Q.push(Edge(q, r, costs.at<float>(q, r)));
        }
    }

    // backtracing to bulid traversal sequence starting at cam1 to cam2, cam1 to cam3..
    valid = true;
    for (size_t q = 0; q < n; q++)
    {
        size_t p = pre[q];

        if (p == NiL && q != start) // this camera is non-reacheable from cam0
        {
            E_WARNING << "camera " << q << " is not reacheable from " << p;
            valid = false;
        }

        std::stringstream ss;
        ss << q;

        while (p != NiL)
        {
            ss << " -> " << p;
            traces[q].push_back(p);
            p = pre[p];
        }

        E_INFO << "trace " << q << ": " << ss.str();
    }
}

bool CalibGraph::Create(size_t cams, size_t views, size_t refCamIdx)
{
    if (cams == 0 || views == 0)
    {
        E_ERROR << "number of cameras and views have to be greater than zero";
        return false;
    }

    if (refCamIdx > cams - 1)
    {
        E_ERROR << "reference camera index (" << refCamIdx << ") is out of bound (" << (cams - 1) << ")";
        return false;
    }

    m_cameraVtx = CameraVertex::Ptrs(cams);
    m_viewVtx   = ViewVertex::Ptrs(views);

    for (size_t c = 0; c < cams;  c++) m_cameraVtx[c] = CameraVertex::Ptr(new CameraVertex(c));
    for (size_t v = 0; v < views; v++) m_viewVtx[v]   = ViewVertex::Ptr  (new ViewVertex  (v));

    m_refCamIdx = refCamIdx;
    m_observations = Observations(cams * views);

    m_cams = cams;
    m_views = views;

    return true;
}

bool CalibGraph::Calibrate(bool pairwiseOptim)
{
    bool monocular = m_cams == 1;
    cv::TermCriteria fastTerm = cv::TermCriteria(cv::TermCriteria::COUNT, 0, 0);
    cv::TermCriteria pairTerm = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-6);

    // calibrate each camera individually
    E_INFO << "starting initialisation of intrinsics";
    for (size_t cam = 0; cam < m_cams; cam++)
    {
        CameraVertex::Ptr vxcam = m_cameraVtx[cam];
        std::vector<Points3F> objectPoints;
        std::vector<Points2F> imagePoints;
        std::vector<size_t> viewIdx;

        if (!vxcam->initialised)
        {
            E_ERROR << "camera " << cam << " not initialised";
            return false;
        }

        for (size_t view = 0; view < m_views; view++)
        {
            Observation& o = GetObservation(cam, view);
            ViewVertex::Ptr vxview = m_viewVtx[view];

            if (!o.IsActive()) continue; // observation suppressed or not available

            if (!vxview->initialised)
            {
                E_ERROR << "view " << view << " not initialised";
                return false;
            }

            objectPoints.push_back(vxview->objectPoints);
            imagePoints.push_back(o.imagePoints);
            viewIdx.push_back(view);
        }

        try
        {
            cv::Mat cameraMatrix, distCoeffs;
            std::vector<cv::Mat> rvecs, tvecs;

            double rpe = cv::calibrateCamera(objectPoints, imagePoints, vxcam->imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0, fastTerm);
        
            vxcam->intrinsics.SetCameraMatrix(cameraMatrix);
            vxcam->intrinsics.SetDistCoeffs(distCoeffs);

            for (size_t i = 0; i < viewIdx.size(); i++)
            {
                Observation& o = GetObservation(cam, viewIdx[i]);
                o.pose.SetRotationVector(rvecs[i]);
                o.pose.SetTranslation(tvecs[i]);
                o.rpe = rpe;
                //E_INFO << "view " << i << ": " << mat2string(rvecs[i], "rvec") << " " << mat2string(tvecs[i], "tvec");
            }

            E_INFO << "camera " << cam << " intrinsics initalised, rpe: " << rpe << " px";
            E_INFO << mat2string(cameraMatrix, "K");
            E_INFO << mat2string(distCoeffs,   "D");
        }
        catch (std::exception& ex)
        {
            E_ERROR << "error calibrating intrinsics for camera " << cam;
            E_ERROR << ex.what();

            return false;
        }
    }

    // then we do pairwise extrinsics extimation from the maximum spanning tree
    // of the cost matrix, which contains the number of shared views between
    // the i-th and the j-th cameras at entry (i,j)
    E_INFO << "starting pairwise extrinsic calibration";

    cv::Mat vmat = GetVisibilityMatrix(); // visibility matrix for camera pairing
    vmat.convertTo(vmat, CV_32F); // costs = #views - vmat * vmat^\top
    MinimumSpanningTree spanningTree(m_views - vmat * vmat.t(), m_refCamIdx);

    if (!spanningTree.valid)
    {
        E_ERROR << "error building maximum spanning tree";
        return false;
    }

    std::vector<EuclideanTransform> transforms(m_cams * m_cams);

    for (size_t i = 0; i < spanningTree.edges.size(); i++)
    {
        const MinimumSpanningTree::Edge& e = spanningTree.edges[i];
        std::vector<Points3F> objectPoints;
        std::vector<Points2F> imagePoints1;
        std::vector<Points2F> imagePoints2;

        E_INFO << "calibrating local transforms of camera pair (" << e.p << "," << e.q << ")";

        for (size_t view = 0; view < m_views; view++)
        {
            const ViewVertex::Ptr viewvtx = m_viewVtx[view];
            const Observation& op = GetObservation(e.p, view);
            const Observation& oq = GetObservation(e.q, view);

            if (!op.IsActive() || !oq.IsActive()) continue;

            objectPoints.push_back(viewvtx->objectPoints);
            imagePoints1.push_back(op.imagePoints);
            imagePoints2.push_back(oq.imagePoints);

            viewvtx->connected = true;
        }

        const CameraVertex::Ptr camvtx1 = m_cameraVtx[e.p];
        const CameraVertex::Ptr camvtx2 = m_cameraVtx[e.q];

        if (camvtx1->imageSize != camvtx2->imageSize)
        {
            E_ERROR << "camera " << e.p << " and " << e.q << " have inconsistent image size";
            return false;
        }

        assert(objectPoints.size() > 0);

        camvtx1->connected = camvtx2->connected = true;

        cv::Mat cameraMatrix1 = camvtx1->intrinsics.GetCameraMatrix();
        cv::Mat cameraMatrix2 = camvtx2->intrinsics.GetCameraMatrix();
        cv::Mat distCoeffs1   = camvtx1->intrinsics.GetDistCoeffs();
        cv::Mat distCoeffs2   = camvtx2->intrinsics.GetDistCoeffs();
        cv::Mat rmat, tvec, emat, fmat;

        double rpe = cv::stereoCalibrate(objectPoints, imagePoints1, imagePoints2,
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, camvtx1->imageSize,
            rmat, tvec, emat, fmat, CV_CALIB_FIX_INTRINSIC, pairwiseOptim ? pairTerm : fastTerm);

        E_INFO << "camera pair (" << e.p << "," << e.q << ") calibrated, rpe: " << rpe << " px";
        E_INFO << mat2string(rmat, "R");
        E_INFO << mat2string(tvec, "t");

        size_t pq = e.p * m_cams + e.q;
        size_t qp = e.q * m_cams + e.p;

        transforms[pq].SetRotationMatrix(rmat);
        transforms[pq].SetTranslation(tvec);

        transforms[qp] = transforms[pq].GetInverse();
    }

    // calculate the extrinsics for each camera by back-tracing and concatenation
    E_INFO << "combining pair-wise transformations to initialise camera extrinsics";
    for (size_t cam = 0; cam < m_cams; cam++)
    {
        CameraVertex::Ptr camvtx = m_cameraVtx[cam];
        EuclideanTransform& extrinsics = camvtx->extrinsics;
        extrinsics = EuclideanTransform::Identity;

        size_t p = cam;
        BOOST_FOREACH(size_t q, spanningTree.traces[cam])
        {
            extrinsics = transforms[p * m_cams + q] >> extrinsics;
            p = q;
        }

        E_INFO << "camera " << cam << (cam == m_refCamIdx ? " (referenced)" : "") << " extrinsics derived";
        E_INFO << mat2string(extrinsics.GetRotationMatrix(), "R");
        E_INFO << mat2string(extrinsics.GetTranslation(),    "t");
    }

    // assigning the best observed pose to each view, taking into account
    // the calibrated extrinsics
    E_INFO << "estimating target pose(s)";
    for (size_t view = 0; view < m_views; view++)
    {
        ViewVertex::Ptr viewvtx = m_viewVtx[view];

        // find the best observation
        const double NiL = -1;
        double bestRpe = NiL;
        size_t bestCam = 0;

        for (size_t cam = 0; cam < m_cams; cam++)
        {
            const Observation& o = GetObservation(cam, view);
            if (o.IsActive() && (o.rpe < bestRpe || bestRpe == NiL))
            {
                bestCam = cam;
                bestRpe = o.rpe;
            }
        }

        if (bestRpe == NiL)
        {
            E_WARNING << "view " << view << " has no valid observation";
            continue;
        }

        const Observation& o = GetObservation(bestCam, view);
        CameraVertex::Ptr camvtx = m_cameraVtx[bestCam];
        cv::Mat rvec = o.pose.GetRotationVector();
        cv::Mat tvec = o.pose.GetTranslation();

        bool solved = cv::solvePnP(
            viewvtx->objectPoints, o.imagePoints,
            camvtx->intrinsics.GetCameraMatrix(), camvtx->intrinsics.GetDistCoeffs(),
            rvec, tvec, true, CV_ITERATIVE);

        if (!solved)
        {
            E_ERROR << "error estimating target pose of view " << view;
            return false;
        }

        viewvtx->pose = EuclideanTransform(rvec, tvec) >> camvtx->extrinsics.GetInverse();

        E_INFO << "view " << view << " target pose estimated";
        E_INFO << mat2string(viewvtx->pose.GetRotationMatrix(), "R");
        E_INFO << mat2string(viewvtx->pose.GetTranslation(),    "t");
    }

    return true;
}

bool CalibGraph::Optimise(size_t iter, double eps, size_t threads)
{    
    Indices camIdx, viewIdx;

    // find all the connected cameras
    if (!m_cameraVtx[m_refCamIdx]->connected)
    {
        E_ERROR << "the referenced camera vertex is not connected";
        return false;
    }

    camIdx.push_back(m_refCamIdx);

    for (size_t cam = 0; cam < m_cams; cam++)
    {
        if (m_cameraVtx[cam]->connected && cam != m_refCamIdx)
        {
            camIdx.push_back(cam);
        }
    }

    // find all the connected views
    for (size_t view = 0; view < m_views; view++)
    {
        if (m_viewVtx[view]->connected) viewIdx.push_back(view);
    }

    CalibGraphBundler::Ptr bundler = CalibGraphBundler::Create(*this, camIdx, viewIdx);

    VectorisableD::Vec x = bundler->Initialise();
    //E_INFO << "initial RMSE: " << rms(cv::Mat(y));

    // solve the bundle adjustment problem by LM algorithm
    LevenbergMarquardtAlgorithm solver;

    solver.SetVervbose(true);
    
    if (!solver.Solve(*bundler, x))
    {
        E_ERROR << "error optimising calibration graph";
        return false;
    }

    bundler->Apply(*this, camIdx, viewIdx);

    return true;
}

bool CalibGraph::WriteReport(const Path& reportPath)
{
    return false;
}

bool CalibGraph::WriteParams(const Path& calPath) const
{
    return false;
}

bool CalibGraph::WriteMFile(const Path& mfilePath) const
{
    std::ofstream of(mfilePath.string().c_str());

    if (!of.is_open())
    {
        return false;
    }

    for (size_t cam = 0; cam < m_cameraVtx.size(); cam++)
    {
        const CameraVertex::Ptr camvtx = m_cameraVtx[cam];
        of << "% camera " << cam << std::endl;
        of << "cam(" << (cam + 1) << ").index     = " << (camvtx->GetIndex() + 1) << ";" << std::endl;
        of << "cam(" << (cam + 1) << ").connected = " << (camvtx->connected ? "true" : "false") << ";" << std::endl;
        of << "cam(" << (cam + 1) << ").imageSize = " << "[" << camvtx->imageSize.height << ", " << camvtx->imageSize.width << "];" << std::endl;
        of << "cam(" << (cam + 1) << ").K         = " << mat2string(camvtx->intrinsics.GetCameraMatrix()) << std::endl;
        of << "cam(" << (cam + 1) << ").D         = " << mat2string(camvtx->intrinsics.GetDistCoeffs()) << std::endl;
        of << "cam(" << (cam + 1) << ").R         = " << mat2string(camvtx->extrinsics.GetRotationMatrix()) << std::endl;
        of << "cam(" << (cam + 1) << ").t         = " << mat2string(camvtx->extrinsics.GetTranslation()) << std::endl;
    }
    
    for (size_t view = 0; view < m_viewVtx.size(); view++)
    {
        const ViewVertex::Ptr viewvtx = m_viewVtx[view];
        of << "% view " << view << std::endl;
        of << "view(" << (view + 1) << ").index        = " << (viewvtx->GetIndex() + 1) << ";" << std::endl;
        of << "view(" << (view + 1) << ").connected    = " << (viewvtx->connected ? "true" : "false") << ";" << std::endl;
        of << "view(" << (view + 1) << ").R            = " << mat2string(viewvtx->pose.GetRotationMatrix()) << std::endl;
        of << "view(" << (view + 1) << ").t            = " << mat2string(viewvtx->pose.GetTranslation()) << std::endl;
        of << "view(" << (view + 1) << ").objectPoints = " << mat2string(cv::Mat(viewvtx->objectPoints).reshape(1, viewvtx->objectPoints.size())) << std::endl;
    }

    size_t i = 0;
    for (size_t view = 0; view < m_viewVtx.size(); view++)
    {
        for (size_t cam = 0; cam < m_cameraVtx.size(); cam++)
        {
            const Observation& o = GetObservation(cam, view);

            if (!o.IsActive()) continue;

            of << "% observation " << i << std::endl;
            of << "observation(" << (i + 1) << ").cam         = " << (cam + 1) << ";" << std::endl;
            of << "observation(" << (i + 1) << ").view        = " << (view + 1) << ";" << std::endl;
            of << "observation(" << (i + 1) << ").src         = '" << o.source.string() << "';" << std::endl;
            of << "observation(" << (i + 1) << ").imagePoints = " << mat2string(cv::Mat(o.imagePoints).reshape(1, o.imagePoints.size())) << std::endl;

            i++;
        }
    }

    return true;
}

void CalibGraph::Summary() const
{
    E_INFO << "summary of the " << m_cams << "-camera " << m_views << "-view calibration graph: ";

    for (size_t cam = 0; cam < m_cams; cam++)
    {
        CameraVertex::Ptr camvtx = m_cameraVtx[cam];
        E_INFO << "camera " << cam;
        E_INFO << mat2string(cv::Mat(camvtx->intrinsics.ToVector()), "intrinsics");
        E_INFO << mat2string(cv::Mat(camvtx->extrinsics.ToVector()), "extrinsics");
    }

    for (size_t view = 0; view < m_views; view++)
    {
        ViewVertex::Ptr viewvtx = m_viewVtx[view];
        E_INFO << "view " << view;
        E_INFO << mat2string(cv::Mat(viewvtx->pose.ToVector()), "pose");
    }
}

bool CalibGraph::Store(Path& path) const
{
    return false;
}

bool CalibGraph::Restore(const Path& path)
{
    return false;
}

cv::Mat CalibGraph::GetVisibilityMatrix() const
{
    cv::Mat vmat = cv::Mat::zeros(m_cams, m_views, CV_16U);
    for (size_t cam = 0; cam < m_cams; cam++)
    {
        for (size_t view = 0; view < m_views; view++)
        {
            vmat.at<ushort>(cam, view) = GetObservation(cam, view).IsActive() ? 1 : 0;
        }
    }
    return vmat;
}

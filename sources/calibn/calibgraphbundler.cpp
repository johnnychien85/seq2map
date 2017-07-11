#include <iomanip>
#include "calibgraphbundler.hpp"

template<typename T0, typename T1>
void castTo64F(const T0& pts32, T1& pts64)
{
    pts64 = T1(pts32.size());

    cv::Mat src(pts32), dst(pts64), tmp;    
    src.convertTo(tmp, CV_64F);
    tmp.copyTo(dst);
}

bool CalibGraphBundler::BundleParams::Store(Vec& v) const
{
    assert(intrinsics.size() == extrinsics.size());

    v.reserve(GetDimension());

    for (size_t cam = 0; cam < intrinsics.size(); cam++)
    {
        Vec vi, ve;;

        if (!intrinsics[cam].Store(vi) ||
            !extrinsics[cam].Store(ve))
        {
            E_ERROR << "error vectorising camera parameters";
            return false;
        }

        v.insert(v.end(), vi.begin(), vi.end());
        v.insert(v.end(), ve.begin(), ve.end());
    }

    for (size_t view = 0; view < poses.size(); view++)
    {
        Vec vp;

        if (!poses[view].Store(vp))
        {
            E_ERROR << "error vectorising pose parameters";
            return false;
        }

        v.insert(v.end(), vp.begin(), vp.end());
    }

    return true;
}

bool CalibGraphBundler::BundleParams::Restore(const Vec& v)
{
    if (v.size() != GetDimension())
    {
        E_ERROR << "given vector has " << v.size() << " element(s), while " << GetDimension() << " expected";
        return false;
    }

    Vec::const_iterator i = v.begin();

    for (size_t cam = 0; cam < intrinsics.size(); cam++)
    {
        Vec vi(i, i + intrinsics[cam].GetDimension()); i += vi.size();
        Vec ve(i, i + extrinsics[cam].GetDimension()); i += ve.size();

        intrinsics[cam].Restore(vi);
        extrinsics[cam].Restore(ve);

        //E_INFO << "intrinsics " << cam << ": " << mat2string(cv::Mat(intrinsics[cam].ToVector()));
        //E_INFO << "extrinsics " << cam << ": " << mat2string(cv::Mat(extrinsics[cam].ToVector()));
    }

    for (size_t view = 0; view < poses.size(); view++)
    {
        Vec vp(i, i + poses[view].GetDimension()); i += vp.size();
        poses[view].Restore(vp);

        //E_INFO << "view " << view << ": " << mat2string(cv::Mat(poses[view].ToVector()));
    }

    if (i != v.end())
    {
        E_ERROR << "residual parameter(s) found";
        return false;
    }

    return true;
}

size_t CalibGraphBundler::BundleParams::GetDimension() const
{
    size_t d = 0;

    BOOST_FOREACH(const BouguetModel& x,       intrinsics) { d += x.GetDimension(); }
    BOOST_FOREACH(const EuclideanTransform& x, extrinsics) { d += x.GetDimension(); }
    BOOST_FOREACH(const EuclideanTransform& x, poses)      { d += x.GetDimension(); }

    return d;
}

Indices CalibGraphBundler::BundleParams::MakeVarList() const
{
    Indices vars;

    size_t n = GetDimension();
    size_t i0 = intrinsics[0].GetDimension();
    size_t i1 = i0 + extrinsics[0].GetDimension();

    for (size_t var = 0; var < n; var++)
    {
        if (var < i0 || var >= i1) vars.push_back(var);
    }

    return vars;
}

CalibGraphBundler::Ptr CalibGraphBundler::Create(const CalibGraph& graph, const Indices& camIdx, const Indices& viewIdx)
{
    size_t cams  = camIdx.size();
    size_t views = viewIdx.size();

    BundleParams params(camIdx.size(), viewIdx.size());
    Views dataset(views);

    E_INFO << "building calibration bundle for camera(s) " << indices2string(camIdx) << " from view(s) " << indices2string(viewIdx);

    // copy camera parameters (intrinsics + extrinsics) from the involved camera vertices
    {
        size_t c = 0;
        BOOST_FOREACH(size_t cam, camIdx)
        {
            const CalibGraph::CameraVertex::Ptr camvtx = graph.m_camvtx[cam];

            assert(camvtx->connected);

            params.intrinsics[c] = camvtx->intrinsics;
            params.extrinsics[c] = camvtx->extrinsics;
            c++;
        }
    }

    // copy target poses from the involved views and gather observations
    {
        size_t v = 0;
        BOOST_FOREACH(size_t view, viewIdx)
        {
            const CalibGraph::ViewVertex::Ptr viewvtx = graph.m_viewvtx[view];
            
            assert(viewvtx->connected);

            params.poses[v] = graph.m_viewvtx[view]->pose;
            castTo64F<Points3F, Points3D>(viewvtx->objectPoints, dataset[v].objectPoints);

            size_t c = 0;
            BOOST_FOREACH(size_t cam, camIdx)
            {
                Projections proj;
                const CalibGraph::Observation& o = graph.GetObservation(cam, view);
                if (o.IsActive())
                {
                    proj.cam = c;
                    castTo64F<Points2F, Points2D>(o.imagePoints, proj.imagePoints);
                    dataset[v].projections.push_back(proj);
                }
                c++;
            }

            assert(!dataset[v].projections.empty());
            v++;
        }
    }

    return Ptr(new CalibGraphBundler(params, dataset));
}

void CalibGraphBundler::Apply(CalibGraph& graph, const Indices& camIdx, const Indices& viewIdx) const
{
    // apply camera parameters
    size_t c = 0;
    BOOST_FOREACH(size_t cam, camIdx)
    {
        const CalibGraph::CameraVertex::Ptr camvtx = graph.m_camvtx[cam];
        camvtx->intrinsics = m_params.intrinsics[c];
        camvtx->extrinsics = m_params.extrinsics[c];
        c++;
    }

    // apply target pose(s)
    size_t v = 0;
    BOOST_FOREACH(size_t view, viewIdx)
    {
        const CalibGraph::ViewVertex::Ptr viewvtx = graph.m_viewvtx[view];
        graph.m_viewvtx[view]->pose = m_params.poses[v];
        v++;
    }
}

size_t CalibGraphBundler::GetPoints(const Views& views)
{
    size_t n = 0;
    BOOST_FOREACH(const View& view, views)
    {
        n += view.objectPoints.size() * view.projections.size();
    }

    return n;
}

VectorisableD::Vec CalibGraphBundler::operator() (const VectorisableD::Vec& x) const
{
    size_t cams = m_params.intrinsics.size(); // = m_params.extrinsics.size()
    BundleParams params(cams, m_views.size());

    if (!params.Restore(x))
    {
        E_ERROR << "error devectorising parameters";
        return VectorisableD::Vec(m_conds, 0);
    }

    VectorisableD::Vec y;
    y.reserve(m_conds);

    for (size_t v = 0; v < m_views.size(); v++)
    {
        const View& view = m_views[v];
        BOOST_FOREACH(const Projections& proj, view.projections)
        {
            EuclideanTransform tform = params.poses[v] << params.extrinsics[proj.cam];
            Points3D p = view.objectPoints;

            params.intrinsics[proj.cam](tform(p));

            // assert(pts2d.size() == proj.imagePoints.size());

            for (size_t i = 0; i < p.size(); i++)
            {
                y.push_back(proj.imagePoints[i].x - p[i].x);
                y.push_back(proj.imagePoints[i].y - p[i].y);
            }
        }
    }

    return y;
}

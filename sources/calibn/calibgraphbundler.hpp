#ifndef CALIBGRAPHBUNDLER_HPP
#define CALIBGRAPHBUNDLER_HPP
#include <seq2map/solve.hpp>
#include "calibgraph.hpp"

using namespace seq2map;

/**
 * CalibGraphBundler gathers camera intrinsics, extrinsics and target poese
 * from a given CalibGraph and constructs an instance of bundle adjustment
 * problem based on the observations of the calibration target.
 */
class CalibGraphBundler : public LeastSquaresProblem
{
public:
    typedef boost::shared_ptr<CalibGraphBundler> Ptr;

    static Ptr Create(const CalibGraph& graph, const Indices& camIdx, const Indices& viewIdx);
    virtual ~CalibGraphBundler() {}
    void Apply(CalibGraph& graph, const Indices& camIdx, const Indices& viewIdx) const;

    // Inherited via LeastSquaresProblem
    virtual VectorisableD::Vec Initialise() { return m_params.ToVector(); }
    virtual VectorisableD::Vec operator() (const VectorisableD::Vec& x) const;
    virtual bool SetSolution(const VectorisableD::Vec& x) { return m_params.FromVector(x); }

private:
    class BundleParams : public VectorisableD
    {
    public:
        BundleParams(size_t cams, size_t views) :
            intrinsics(std::vector<BouguetModel>(cams)),
            extrinsics(EuclideanTransforms(cams)),
            poses     (EuclideanTransforms(views)) {}

        virtual Vec ToVector() const;
        virtual bool FromVector(const Vec& v);
        virtual size_t GetDimension() const;
        Indices MakeVarList() const;

        std::vector<BouguetModel> intrinsics;
        EuclideanTransforms extrinsics;
        EuclideanTransforms poses;
    };

    struct Projections
    {
        size_t cam;
        Points2D imagePoints;
    };

    struct View
    {
        Points3D objectPoints;
        std::vector<Projections> projections;
    };

    typedef std::vector<View> Views;

    static size_t GetPoints(const Views& views);
    CalibGraphBundler(const BundleParams& params, const Views& views)
    : m_params(params), m_views(views),
      LeastSquaresProblem(GetPoints(views) * 2, params.GetDimension(), params.MakeVarList()) {}

    BundleParams m_params;
    Views        m_views;
};

#endif // CALIBGRAPHBUNDLER_HPP

#include <seq2map/mapping.hpp>

using namespace seq2map;

namespace seq2map
{
    Mapper::Capability MfiMapper::GetCapability(const Sequence& seq) const
    {
        Capability capability;

        size_t featureStores = 0;
        size_t dispStores = 0;

        BOOST_FOREACH (const Camera& cam, seq.GetCameras())
        {
            featureStores += cam.GetFeatureStores().size();
        }

        BOOST_FOREACH (const RectifiedStereo& pair, seq.GetRectifiedStereo())
        {
            dispStores += pair.GetDisparityStores().size();
        }

        capability.motion = featureStores > 0;
        capability.metric = dispStores > 0;
        capability.dense = false;

        return capability;
    }

    bool MfiMapper::Map(const Sequence& seq, size_t t0, size_t tn)
    {
        for (size_t t = t0; t < tn; t++)
        {

        }


        return false;
    }
}

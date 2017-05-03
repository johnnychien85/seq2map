#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <seq2map/sequence.hpp>

namespace seq2map
{
    class Mapper
    {
    public:
        struct Capability
        {
            bool motion; // ego-motion estimation
            bool metric; // metric reconstruction
            bool dense;  // dense or semi-dense reconstruction
        };

        virtual Capability GetCapability(const Sequence& seq) const = 0;
        virtual bool Map(const Sequence& seq, size_t t0, size_t tn) = 0;
        inline bool Map(const Sequence& seq, size_t t0 = 0) { return Map(seq, t0, seq.GetFrames() - 1); }
    };

    /**
     * Implementation of a generalised multi-frame feature integration algorithm
     * based on the paper "Visual Odometry by Multi-frame Feature Integration"
     */
    class MfiMapper : public Mapper
    {
    public:
        virtual Capability GetCapability(const Sequence& seq) const;
        virtual bool Map(const Sequence& seq, size_t t0, size_t t1);
    };
}
#endif // MAPPING_HPP

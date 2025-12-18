#pragma once

#include <pomdp/belief.hpp>
namespace pomdp{
    template <typename StateT>
    class BeliefSampler {
    public:
        virtual ~BeliefSampler() = default;
        virtual StateT sample_state(const Belief& belief) const = 0;
    };

}
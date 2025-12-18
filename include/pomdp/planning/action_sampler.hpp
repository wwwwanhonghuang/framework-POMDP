#pragma once

#include <pomdp/action.hpp>
#include <pomdp/belief.hpp>

namespace pomdp {
class ActionSampler {
public:
    virtual ~ActionSampler() = default;

    // Required: used for expansion / root decisions
    virtual Action sample_action(
        const Belief& belief
    ) const = 0;

};


} // namespace pomdp

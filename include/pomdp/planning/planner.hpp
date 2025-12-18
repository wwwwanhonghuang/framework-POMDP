#pragma once

#include <pomdp/action.hpp>
#include <pomdp/belief.hpp>
#include <pomdp/history.hpp>

namespace pomdp {

class Planner {
public:
    virtual ~Planner() = default;

    /// Decide the next action given current belief and history
    virtual Action decide(
        const Belief& belief,
        const History& history
    ) = 0;
};

} // namespace pomdp

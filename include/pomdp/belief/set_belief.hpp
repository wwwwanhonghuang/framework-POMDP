#pragma once

#include <pomdp/belief.hpp>

namespace pomdp {

/**
 * Set-based belief:
 *   x ∈ feasible_set
 *
 * No probability, only feasibility.
 */
template <typename StateT>
class SetBelief : public Belief {
public:
    StateT lower_bound;
    StateT upper_bound;
};

} // namespace pomdp

#pragma once

#include <pomdp/belief.hpp>

namespace pomdp {

/**
 * Hybrid belief:
 *   coarse global + local particle detail
 */
template <typename GlobalBeliefT, typename LocalBeliefT>
class HybridBelief : public Belief {
public:
    GlobalBeliefT global;
    LocalBeliefT local;
};

} // namespace pomdp

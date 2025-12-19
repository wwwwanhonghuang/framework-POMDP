#pragma once

#include <vector>
#include <pomdp/belief.hpp>

namespace pomdp {

/**
 * Discrete belief over finite state space.
 *
 * states[i] with probability probs[i]
 */
template <typename StateT>
class DiscreteBelief : public Belief {
public:
    std::vector<StateT> states;
    std::vector<double> probs;
};

} // namespace pomdp

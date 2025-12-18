#pragma once

#include <pomdp/action.hpp>

namespace pomdp {

/**
 * Transition kernel:
 *   p(x' | x, a)
 *
 * This is a mathematical object, not a simulator.
 */
template <typename StateT>
class TransitionKernel {
public:
    virtual ~TransitionKernel() = default;

    /// log p(x' | x, a)
    virtual double transition_log_prob(
        const StateT& next,
        const StateT& prev,
        const Action& action
    ) const = 0;
};

} // namespace pomdp

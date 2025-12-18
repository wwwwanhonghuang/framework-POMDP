#pragma once

#include <cstddef>

#include <pomdp/action.hpp>
#include <pomdp/planning/simulator.hpp>

namespace pomdp::mcst {

/**
 * @brief Abstract rollout policy for Monte-Carlo planners.
 *
 * A rollout policy:
 *  - simulates future rewards from a given state and action
 *  - does NOT modify any tree or statistics
 *  - does NOT update belief
 *
 * Typical implementations:
 *  - random rollout
 *  - heuristic rollout
 *  - learned (NN-based) rollout
 *  - batched / SIMD rollout
 */
template <typename StateT>
class RolloutPolicy {
public:
    virtual ~RolloutPolicy() = default;

    /**
     * @brief Estimate discounted return starting from (state, first_action).
     *
     * @param state          Hypothetical current state
     * @param first_action  Action to apply at this step
     * @param simulator     Generative model
     * @param horizon       Remaining horizon
     * @param discount      Discount factor
     *
     * @return Estimated cumulative reward
     */
    virtual double rollout(
        StateT state,
        const Action& first_action,
        const Simulator<StateT>& simulator,
        std::size_t horizon,
        double discount
    ) const = 0;
};

} // namespace pomdp::mcst

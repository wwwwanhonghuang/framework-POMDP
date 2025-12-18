#pragma once

#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>

namespace pomdp {

/**
 * Proposal kernel for particle filters.
 *
 * Defines how new state hypotheses are proposed
 * given the previous state, action, and observation.
 *
 * This is a solver-level abstraction.
 */
template <typename StateT>
class ProposalKernel {
public:
    virtual ~ProposalKernel() = default;

    /**
     * Sample a proposed next state.
     */
    virtual StateT sample(
        const StateT& prev_state,
        const Action& action,
        const Observation& observation
    ) const = 0;
};

} // namespace pomdp

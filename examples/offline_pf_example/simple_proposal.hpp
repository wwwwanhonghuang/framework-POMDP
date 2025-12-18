#pragma once

#include <pomdp/particle_filter/proposal_kernel.hpp>
#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>

namespace example {

/**
 * A simple prior-based proposal.
 *
 * This proposal ignores the observation and
 * uses a trivial action-conditioned transition.
 *
 * It is intentionally naive and serves only
 * as an example.
 */
template <typename StateT>
class SimplePriorProposal : public pomdp::ProposalKernel<StateT> {
public:
    StateT sample(
        const StateT& prev_state,
        const pomdp::Action& action,
        const pomdp::Observation& /*observation*/
    ) const override
    {
        // Example logic:
        // user defines how action influences state
        return transition(prev_state, action);
    }

private:
    StateT transition(
        const StateT& x,
        const pomdp::Action& a
    ) const
    {
        // Placeholder: user-defined dynamics
        // For discrete StateT=int, this might be:
        return x; // identity by default
    }
};

} // namespace example

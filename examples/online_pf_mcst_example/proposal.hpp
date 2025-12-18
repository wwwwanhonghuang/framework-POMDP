#pragma once

#include <pomdp/particle_filter/proposal_kernel.hpp>
#include "model.hpp"

namespace online_example {

/**
 * @brief Bootstrap proposal kernel.
 *
 * Proposes next state by sampling from the transition model:
 *
 *   q(x_t | x_{t-1}, a_t, o_t) = p(x_t | x_{t-1}, a_t)
 *
 * Weighting is handled by transition + observation likelihoods
 * inside ParticleFilterUpdater.
 */
template <typename StateT>
class BootstrapProposal : public pomdp::ProposalKernel<StateT> {
public:
    explicit BootstrapProposal(const ContinuousModel& model)
        : model_(model)
    {}

    StateT sample(
        const StateT& prev,
        const pomdp::Action& action,
        const pomdp::Observation& obs
    ) const override
    {
        // observation is ignored for bootstrap proposal
        (void)obs;
        return model_.step(prev, action).next_state;
    }

private:
    const ContinuousModel& model_;
};

} // namespace online_example

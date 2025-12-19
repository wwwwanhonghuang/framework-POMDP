#pragma once

#include <bayesian_filter/particle_filter/proposal_kernel.hpp>
#include "model.hpp"

namespace online_example {

/**
 * @brief Bootstrap proposal kernel.
 *
 * Proposes next state by sampling from the transition model:
 *
 *   q(x_t | x_{t-1}, a_t, o_t) = p(x_t | x_{t-1}, a_t)
 *
 * Observation is ignored in sampling.
 */


template <
    typename StateT,
    typename ActionT,
    typename ObservationT
>
class BootstrapProposal
    : public bayesian_filter::ProposalKernel<
          StateT,
          ActionT,
          ObservationT
      >
{
public:
    explicit BootstrapProposal(const ContinuousModel& model)
        : model_(model)
    {}

    StateT sample(
        const StateT& prev,
        const ActionT& action,
        const ObservationT& /*obs*/
    ) const override
    {
        return model_.step(prev, action).next_state;
    }

    double probability(
        const StateT& next,
        const StateT& prev,
        const ActionT& action,
        const ObservationT& /*obs*/
    ) const override
    {
        // Bootstrap proposal: q = p_transition
        return std::exp(
            model_.transition_log_prob(next, prev, action)
        );
    }

private:
    const ContinuousModel& model_;
};

} // namespace online_example

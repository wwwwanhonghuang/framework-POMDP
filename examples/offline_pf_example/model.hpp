#pragma once

#include <pomdp/pomdp_model.hpp>
#include <pomdp/dynamics/transition_kernel.hpp>
#include <pomdp/dynamics/observation_kernel.hpp>

namespace example {

/**
 * A simple discrete POMDP model with integer states.
 */
class DiscreteModel
    : public pomdp::POMDPModel<int>
    , public pomdp::TransitionKernel<int>
    , public pomdp::ObservationKernel<int>
{
public:
    // --- POMDPModel interface ---
    const pomdp::TransitionKernel<int>& transition() const override {
        return *this;
    }

    const pomdp::ObservationKernel<int>& observation() const override {
        return *this;
    }

    // --- TransitionKernel interface ---
    double transition_log_prob(
        const int& next,
        const int& prev,
        const pomdp::Action& action
    ) const override;

    // --- ObservationKernel interface ---
    double observation_log_prob(
        const pomdp::Observation& obs,
        const int& state
    ) const override;

    // --- Example utility ---
    int num_actions() const;
};

} // namespace example

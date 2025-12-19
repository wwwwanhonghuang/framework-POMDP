#pragma once

namespace bayesian_filter {

/**
 * @brief Proposal kernel q(x_t | x_{t-1}, u_t, y_t)
 */
template <typename StateT, typename ControlT, typename ObservationT>
class ProposalKernel {
public:
    virtual ~ProposalKernel() = default;

    virtual StateT sample(
        const StateT& prev_state,
        const ControlT& control,
        const ObservationT& observation
    ) const = 0;

    virtual double probability(
        const StateT& next_state,
        const StateT& prev_state,
        const ControlT& control,
        const ObservationT& observation
    ) const = 0;
};

} // namespace bayesian_filter

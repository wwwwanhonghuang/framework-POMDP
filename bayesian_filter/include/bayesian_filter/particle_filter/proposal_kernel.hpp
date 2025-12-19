#pragma once

namespace bayesian_filter {

/**
 * @brief Proposal kernel for particle filters.
 *
 * Represents a proposal distribution:
 *   q(x_t | x_{t-1}, u_t, y_t)
 *
 * This abstraction is algorithmic (importance sampling),
 * not a generative model of the system.
 */
template <
    typename StateT,
    typename ControlT,
    typename ObservationT
>
class ProposalKernel {
public:
    virtual ~ProposalKernel() = default;

    /**
     * @brief Sample a proposed next state.
     */
    virtual StateT sample(
        const StateT& prev_state,
        const ControlT& control,
        const ObservationT& observation
    ) const = 0;

    /**
     * @brief Evaluate proposal probability or density.
     *
     * Required for importance weight correction.
     */
    virtual double probability(
        const StateT& next_state,
        const StateT& prev_state,
        const ControlT& control,
        const ObservationT& observation
    ) const = 0;
};

} // namespace bayesian_filter

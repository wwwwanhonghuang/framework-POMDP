#pragma once

namespace bayesian_filter {

/**
 * @brief State transition model p(x_t | x_{t-1}, u_t)
 */
template <typename StateT, typename ControlT>
class TransitionModel {
public:
    virtual ~TransitionModel() = default;

    virtual double probability(
        const StateT& next_state,
        const StateT& current_state,
        const ControlT& control
    ) const = 0;
};

} // namespace bayesian_filter

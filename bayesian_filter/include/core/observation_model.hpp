#pragma once

namespace bayesian_filter {

/**
 * @brief Observation model p(y_t | x_t)
 */
template <typename StateT, typename ObservationT>
class ObservationModel {
public:
    virtual ~ObservationModel() = default;

    virtual double probability(
        const ObservationT& observation,
        const StateT& state
    ) const = 0;
};

} // namespace bayesian_filter

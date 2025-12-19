#pragma once

#include <memory>
#include <bayesian_filter/core/state_distribution.hpp>

namespace bayesian_filter {

/**
 * @brief Abstract Bayesian filtering interface.
 */
template <typename StateT, typename ControlT, typename ObservationT>
class BayesianFilter {
public:
    virtual ~BayesianFilter() = default;

    virtual std::unique_ptr<StateDistribution> predict(
        const StateDistribution& prior,
        const ControlT& control
    ) const = 0;

    virtual std::unique_ptr<StateDistribution> update(
        const StateDistribution& predicted,
        const ObservationT& observation
    ) const = 0;

    virtual std::unique_ptr<StateDistribution> step(
        const StateDistribution& prior,
        const ControlT& control,
        const ObservationT& observation
    ) const {
        return update(*predict(prior, control), observation);
    }
};

} // namespace bayesian_filter

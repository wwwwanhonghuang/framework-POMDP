#pragma once

namespace bayesian_filter {

/**
 * @brief Probabilistic model over control / input.
 *
 * Optional abstraction.
 */
template <typename ControlT>
class ControlModel {
public:
    virtual ~ControlModel() = default;

    virtual double probability(
        const ControlT& control
    ) const = 0;
};

} // namespace bayesian_filter

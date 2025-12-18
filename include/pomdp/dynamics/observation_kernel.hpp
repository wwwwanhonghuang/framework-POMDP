#pragma once

#include <pomdp/observation.hpp>

namespace pomdp {

/**
 * Observation kernel:
 *   p(o | x')
 */
template <typename StateT>
class ObservationKernel {
public:
    virtual ~ObservationKernel() = default;

    /// log p(o | x')
    virtual double observation_log_prob(
        const Observation& obs,
        const StateT& state
    ) const = 0;
};

} // namespace pomdp

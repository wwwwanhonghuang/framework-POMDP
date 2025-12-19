#pragma once

#include <memory>

namespace bayesian_filter {

/**
 * @brief Abstract probability distribution over latent state.
 *
 * Pure mathematical object. No agent or POMDP semantics.
 */
class StateDistribution {
public:
    virtual ~StateDistribution() = default;
    virtual std::unique_ptr<StateDistribution> clone() const = 0;
};

} // namespace bayesian_filter

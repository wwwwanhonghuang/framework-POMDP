#pragma once

namespace bayesian_filter {

/**
 * @brief Generic weighted particle.
 */
template <typename StateT>
struct Particle {
    StateT state;
    double weight = 1.0;
};

} // namespace bayesian_filter

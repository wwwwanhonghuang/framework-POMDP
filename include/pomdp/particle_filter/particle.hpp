#pragma once

namespace pomdp {

/**
 * A weighted particle representing a single hypothesis
 * about the latent state.
 *
 * StateT is intentionally unconstrained:
 * - int (discrete state)
 * - vector<double>
 * - struct
 * - symbolic object
 */
template <typename StateT>
struct Particle {
    StateT x;
    double weight;
};

} // namespace pomdp

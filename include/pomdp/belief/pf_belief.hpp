#pragma once

#include <vector>

#include <pomdp/belief.hpp>
#include <pomdp/particle_filter/particle.hpp>

namespace pomdp {

/**
 * Particle-based belief representation.
 *
 * Purely representational:
 *  - stores particles
 *  - provides minimal numerical utilities
 */
template <typename StateT>
class ParticleBelief : public Belief {
public:
    using ParticleT = Particle<StateT>;

    std::vector<ParticleT> particles;

    /**
     * Normalize particle weights:
     *   sum_i w_i = 1
     */
    void normalize();

    /**
     * Effective Sample Size (ESS):
     *   ESS = 1 / sum_i (w_i^2)
     */
    double effective_sample_size() const;
};

} // namespace pomdp

// ---- template implementation ----
#include "pf_belief.tpp"

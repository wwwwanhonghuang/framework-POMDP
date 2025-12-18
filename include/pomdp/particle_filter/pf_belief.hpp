#pragma once

#include <vector>
#include <cmath>
#include <pomdp/belief.hpp>
#include <pomdp/particle_filter/particle.hpp>

namespace pomdp {

/**
 * Particle-based belief representation.
 *
 * This class is purely representational:
 * it stores particles and provides minimal
 * numerical utilities required by particle solvers.
 */
template <typename StateT>
class ParticleBelief : public Belief {
public:
    using ParticleT = Particle<StateT>;

    std::vector<ParticleT> particles;

    /**
     * Normalize particle weights so that
     * sum_i w_i = 1.
     */
    void normalize() {
        double sum = 0.0;
        for (const auto& p : particles) {
            sum += p.weight;
        }
        if (sum <= 0.0) return;

        for (auto& p : particles) {
            p.weight /= sum;
        }
    }

    /**
     * Effective Sample Size (ESS):
     *   ESS = 1 / sum_i (w_i^2)
     *
     * Used by solvers to decide when to resample.
     */
    double effective_sample_size() const {
        double sq_sum = 0.0;
        for (const auto& p : particles) {
            sq_sum += p.weight * p.weight;
        }
        return (sq_sum > 0.0) ? 1.0 / sq_sum : 0.0;
    }
};

} // namespace pomdp

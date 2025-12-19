#pragma once

#include <vector>
#include <memory>

#include <bayesian_filter/core/state_distribution.hpp>

namespace bayesian_filter {

/**
 * @brief Particle-based state distribution.
 *
 * This class represents a probability distribution over latent
 * state using weighted particles.
 *
 * It is purely representational:
 *  - stores particles and weights
 *  - provides minimal numerical utilities
 *
 * It contains NO filtering logic.
 */
template <typename StateT>
class ParticleBelief : public StateDistribution {
public:
    struct Particle {
        StateT state;
        double weight;
    };

    using ParticleT = Particle;

    std::vector<ParticleT> particles;

    /**
     * @brief Normalize particle weights so that sum_i w_i = 1.
     */
    void normalize();

    /**
     * @brief Effective Sample Size (ESS).
     *
     * ESS = 1 / sum_i (w_i^2)
     */
    double effective_sample_size() const;

    /**
     * @brief Polymorphic copy.
     */
    std::unique_ptr<StateDistribution> clone() const override {
        return std::make_unique<ParticleBelief<StateT>>(*this);
    }
};

} // namespace bayesian_filter

// ---- template implementation ----
#include "particle_belief.tpp"

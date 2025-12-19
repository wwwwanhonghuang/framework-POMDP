#pragma once

#include <random>

#include <pomdp/planning/belief_sampler.hpp>
#include <pomdp/belief/pf_belief.hpp>

namespace online_example {

/**
 * @brief BeliefSampler for particle-filter beliefs.
 *
 * Samples one particle (state) from the PF belief
 * according to its weight distribution.
 *
 * This class:
 *  - does NOT update the belief
 *  - does NOT modify particles
 *  - is read-only and thread-safe
 */
template <typename StateT>
class PFBeliefSampler : public pomdp::BeliefSampler<StateT> {
public:
    PFBeliefSampler() = default;

    StateT sample_state(
        const pomdp::Belief& belief
    ) const override
    {
        // Expect PF belief
        const auto& pb =
            dynamic_cast<const pomdp::ParticleBelief<StateT>&>(belief);

        const auto& particles = pb.particles;

        const std::size_t n = particles.size();

        // Sample index proportional to weights
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        static thread_local std::mt19937 gen{std::random_device{}()};

        double r = dist(gen);
        double accum = 0.0;

        for (const auto& p : particles) {
            accum += p.weight;
            if (r <= accum) {
                return p.x;
            }
        }

        // Fallback (numerical safety)
        return particles.back().x;
    }
};

} // namespace online_example

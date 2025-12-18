#pragma once

#include <random>
#include <vector>
#include <numeric>
#include <pomdp/particle_filter/resampler.hpp>

namespace pomdp {

/**
 * Systematic resampling for particle filters.
 *
 * This resampler assumes the belief is already normalized.
 * After resampling, all particles will have equal weights.
 */
template <typename StateT>
class SystematicResampler : public Resampler<StateT> {
public:
    void resample(ParticleBelief<StateT>& belief) const override {
        const std::size_t N = belief.particles.size();
        if (N == 0) return;

        // Cumulative distribution
        std::vector<double> cdf(N);
        cdf[0] = belief.particles[0].weight;
        for (std::size_t i = 1; i < N; ++i) {
            cdf[i] = cdf[i - 1] + belief.particles[i].weight;
        }

        std::vector<Particle<StateT>> new_particles;
        new_particles.reserve(N);

        // Systematic draw
        static thread_local std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<double> uni(0.0, 1.0 / N);
        double r = uni(gen);

        std::size_t i = 0;
        for (std::size_t m = 0; m < N; ++m) {
            double u = r + static_cast<double>(m) / N;
            while (u > cdf[i] && i + 1 < N) {
                ++i;
            }
            new_particles.push_back(
                Particle<StateT>{belief.particles[i].x, 1.0 / N}
            );
        }

        belief.particles = std::move(new_particles);
    }
};

} // namespace pomdp

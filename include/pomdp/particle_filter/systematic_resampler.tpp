#pragma once

#include <random>
#include <vector>

namespace pomdp {

template <typename StateT>
void SystematicResampler<StateT>::resample(
    ParticleBelief<StateT>& belief
) const
{
    const std::size_t N = belief.particles.size();
    if (N == 0) {
        return;
    }

    // --- cumulative distribution ---
    std::vector<double> cdf(N);
    cdf[0] = belief.particles[0].weight;
    for (std::size_t i = 1; i < N; ++i) {
        cdf[i] = cdf[i - 1] + belief.particles[i].weight;
    }

    std::vector<Particle<StateT>> new_particles;
    new_particles.reserve(N);

    // --- systematic draw ---
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

} // namespace pomdp

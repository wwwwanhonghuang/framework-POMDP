#pragma once

#include <numeric>

namespace bayesian_filter {

template <typename StateT>
std::vector<typename SystematicResampler<StateT>::ParticleT>
SystematicResampler<StateT>::resample(
    const std::vector<ParticleT>& particles
) const
{
    const std::size_t N = particles.size();
    std::vector<ParticleT> new_particles;
    new_particles.reserve(N);

    if (N == 0) {
        return new_particles;
    }

    // Step size
    const double step = 1.0 / static_cast<double>(N);

    // Random offset in [0, step)
    std::uniform_real_distribution<double> dist(0.0, step);
    double u = dist(rng_);

    double cumulative_weight = particles[0].weight;
    std::size_t i = 0;

    for (std::size_t m = 0; m < N; ++m) {
        const double threshold = u + m * step;

        while (threshold > cumulative_weight && i + 1 < N) {
            ++i;
            cumulative_weight += particles[i].weight;
        }

        ParticleT p = particles[i];
        p.weight = step;  // equal weight = 1/N
        new_particles.push_back(p);
    }

    return new_particles;
}

} // namespace bayesian_filter

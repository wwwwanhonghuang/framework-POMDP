#pragma once

#include <vector>

#include <bayesian_filter/particle_filter/particle.hpp>

namespace bayesian_filter {

/**
 * @brief Abstract resampler for particle filters.
 *
 * Resampling transforms a weighted particle set into a new
 * (typically equally-weighted) particle set in order to
 * reduce degeneracy.
 *
 * This interface is purely algorithmic.
 */
template <typename StateT>
class Resampler {
public:
    using ParticleT = Particle<StateT>;

    virtual ~Resampler() = default;

    /**
     * @brief Resample particles.
     *
     * @param particles Input weighted particles.
     * @return Resampled particle set.
     */
    virtual std::vector<ParticleT> resample(
        const std::vector<ParticleT>& particles
    ) const = 0;
};

} // namespace bayesian_filter

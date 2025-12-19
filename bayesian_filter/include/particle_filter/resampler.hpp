#pragma once

#include <vector>
#include <bayesian_filter/particle_filter/particle.hpp>

namespace bayesian_filter {

/**
 * @brief Abstract resampler interface.
 */
template <typename StateT>
class Resampler {
public:
    using ParticleT = Particle<StateT>;
    virtual ~Resampler() = default;

    virtual std::vector<ParticleT> resample(
        const std::vector<ParticleT>& particles
    ) const = 0;
};

} // namespace bayesian_filter

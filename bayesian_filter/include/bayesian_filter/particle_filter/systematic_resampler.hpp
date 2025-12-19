#pragma once

#include <random>
#include <vector>

#include <bayesian_filter/particle_filter/resampler.hpp>

namespace bayesian_filter {

/**
 * @brief Systematic resampling algorithm.
 *
 * Assumes input particle weights are normalized.
 *
 * Produces equally-weighted particles using a single
 * random offset and cumulative weight traversal.
 */
template <typename StateT>
class SystematicResampler : public Resampler<StateT> {
public:
    using ParticleT = typename Resampler<StateT>::ParticleT;

    /**
     * @param rng Random number generator used for sampling.
     */
    explicit SystematicResampler(std::mt19937& rng)
        : rng_(rng) {}

    std::vector<ParticleT> resample(
        const std::vector<ParticleT>& particles
    ) const override;

private:
    std::mt19937& rng_;
};

} // namespace bayesian_filter

// ---- template implementation ----
#include "systematic_resampler.tpp"

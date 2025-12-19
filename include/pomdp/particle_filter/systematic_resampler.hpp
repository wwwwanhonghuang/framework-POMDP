#pragma once

#include <pomdp/particle_filter/resampler.hpp>

namespace pomdp {

/**
 * Systematic resampling for particle filters.
 *
 * Assumes the belief is already normalized.
 * After resampling, all particles have equal weights.
 */
template <typename StateT>
class SystematicResampler : public Resampler<StateT> {
public:
    void resample(ParticleBelief<StateT>& belief) const override;
};

} // namespace pomdp

// ---- template implementation ----
#include "systematic_resampler.tpp"

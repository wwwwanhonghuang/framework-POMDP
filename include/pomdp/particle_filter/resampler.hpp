#pragma once

#include <pomdp/belief/pf_belief.hpp>

namespace pomdp {

/**
 * Resampling strategy for particle filters.
 *
 * A resampler transforms a particle belief to mitigate
 * weight degeneracy, typically by duplicating high-weight
 * particles and discarding low-weight ones.
 *
 * This is a solver-level abstraction.
 */
template <typename StateT>
class Resampler {
public:
    virtual ~Resampler() = default;

    /**
     * Resample the given particle belief in-place.
     *
     * Implementations may assume the belief has been
     * normalized before resampling.
     */
    virtual void resample(
        ParticleBelief<StateT>& belief
    ) const = 0;
};

} // namespace pomdp

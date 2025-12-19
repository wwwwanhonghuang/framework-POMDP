#pragma once

#include <bayesian_filter/particle_filter/particle_distribution.hpp>
#include <pomdp/belief.hpp>

namespace pomdp {

/**
 * @brief Particle-based belief for POMDP.
 *
 * This class:
 *  - reuses bayesian_filter::ParticleBelief for all particle math
 *  - tags it as a POMDP Belief (epistemic state)
 *
 * NO particle logic is reimplemented here.
 */
template <typename StateT>
class POMDPParticleBelief
    : public Belief
    , public bayesian_filter::ParticleDistribution<StateT>
{
public:
    using Base = bayesian_filter::ParticleDistribution<StateT>;
    using Base::Base; // inherit constructors

    // No normalize(), no ESS here.
    // All particle logic lives in the Bayesian layer.
};

} // namespace pomdp

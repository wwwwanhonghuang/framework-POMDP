#pragma once

#include <cmath>

namespace pomdp {

template <typename StateT>
ParticleFilterUpdater<StateT>::ParticleFilterUpdater(
    const POMDPModel<StateT>& model,
    const ProposalKernel<StateT>& proposal,
    const Resampler<StateT>& resampler,
    double ess_threshold
)
    : T_(model.transition())
    , O_(model.observation())
    , Q_(proposal)
    , R_(resampler)
    , ess_threshold_(ess_threshold)
{}

template <typename StateT>
std::unique_ptr<Belief>
ParticleFilterUpdater<StateT>::update(
    const Belief& prev,
    const Action& action,
    const Observation& observation,
    const History& /*history*/
) const
{
    const auto& b_prev =
        static_cast<const ParticleBelief<StateT>&>(prev);

    auto b_new = std::make_unique<ParticleBelief<StateT>>();
    b_new->particles.reserve(b_prev.particles.size());

    // --- propagate & weight ---
    for (const auto& p : b_prev.particles) {
        StateT x_new = Q_.sample(p.x, action, observation);

        double log_w =
            T_.transition_log_prob(x_new, p.x, action) +
            O_.observation_log_prob(observation, x_new);

        b_new->particles.push_back(
            Particle<StateT>{x_new, p.weight * std::exp(log_w)}
        );
    }

    // --- normalize ---
    b_new->normalize();

    // --- resample if ESS too small ---
    if (b_new->effective_sample_size()
        < ess_threshold_ * b_new->particles.size())
    {
        R_.resample(*b_new);
    }

    return b_new;
}

} // namespace pomdp

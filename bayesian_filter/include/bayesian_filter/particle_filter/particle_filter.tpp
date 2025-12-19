#pragma once

#include <cassert>
#include <cmath>

namespace bayesian_filter {

template <
    typename StateT,
    typename ControlT,
    typename ObservationT
>
ParticleFilter<StateT, ControlT, ObservationT>::ParticleFilter(
    const TransitionModel<StateT, ControlT>& transition_model,
    const ObservationModel<StateT, ObservationT>& observation_model,
    const ProposalKernel<StateT, ControlT, ObservationT>& proposal_kernel,
    const Resampler<StateT>& resampler,
    double ess_threshold
)
    : transition_model_(transition_model),
      observation_model_(observation_model),
      proposal_kernel_(proposal_kernel),
      resampler_(resampler),
      ess_threshold_(ess_threshold)
{
    assert(ess_threshold_ > 0.0 && ess_threshold_ <= 1.0);
}

template <
    typename StateT,
    typename ControlT,
    typename ObservationT
>
std::unique_ptr<StateDistribution>
ParticleFilter<StateT, ControlT, ObservationT>::predict(
    const StateDistribution& prior,
    const ControlT& control
) const
{
    const auto& prev =
        dynamic_cast<const BeliefT&>(prior);

    auto predicted = std::make_unique<BeliefT>();
    predicted->particles.reserve(prev.particles.size());

    for (const auto& p : prev.particles) {
        StateT next_state =
            proposal_kernel_.sample(
                p.state, control, ObservationT{}
            );

        double q =
            proposal_kernel_.probability(
                next_state, p.state, control, ObservationT{}
            );

        double p_trans =
            transition_model_.probability(
                next_state, p.state, control
            );

        ParticleT new_particle;
        new_particle.state  = next_state;
        new_particle.weight = p.weight * (p_trans / q);

        predicted->particles.push_back(new_particle);
    }

    predicted->normalize();
    return predicted;
}

template <
    typename StateT,
    typename ControlT,
    typename ObservationT
>
std::unique_ptr<StateDistribution>
ParticleFilter<StateT, ControlT, ObservationT>::update(
    const StateDistribution& predicted,
    const ObservationT& observation
) const
{
    auto updated =
        std::make_unique<BeliefT>(
            dynamic_cast<const BeliefT&>(predicted)
        );

    for (auto& p : updated->particles) {
        double likelihood =
            observation_model_.probability(
                observation, p.state
            );
        p.weight *= likelihood;
    }

    updated->normalize();

    const double ess = updated->effective_sample_size();
    const double n   = static_cast<double>(updated->particles.size());

    if (ess < ess_threshold_ * n) {
        updated->particles =
            resampler_.resample(updated->particles);

        const double w = 1.0 / updated->particles.size();
        for (auto& p : updated->particles) {
            p.weight = w;
        }
    }

    return updated;
}

} // namespace bayesian_filter

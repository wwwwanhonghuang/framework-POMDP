#pragma once

#include <memory>

#include <bayesian_filter/core/bayesian_filter.hpp>
#include <bayesian_filter/core/transition_model.hpp>
#include <bayesian_filter/core/observation_model.hpp>

#include <bayesian_filter/belief/particle_belief.hpp>
#include <bayesian_filter/particle_filter/proposal_kernel.hpp>
#include <bayesian_filter/particle_filter/resampler.hpp>

namespace bayesian_filter {

/**
 * @brief Generic Particle Filter (Sequential Monte Carlo).
 *
 * Implements recursive Bayesian filtering using importance
 * sampling and resampling.
 *
 * This class is model-agnostic and reusable across domains.
 */
template <
    typename StateT,
    typename ControlT,
    typename ObservationT
>
class ParticleFilter
    : public BayesianFilter<StateT, ControlT, ObservationT> {
public:
    using BeliefT   = ParticleBelief<StateT>;
    using ParticleT = typename BeliefT::ParticleT;

    ParticleFilter(
        const TransitionModel<StateT, ControlT>& transition_model,
        const ObservationModel<StateT, ObservationT>& observation_model,
        const ProposalKernel<StateT, ControlT, ObservationT>& proposal_kernel,
        const Resampler<StateT>& resampler,
        double ess_threshold = 0.5
    );

    std::unique_ptr<StateDistribution> predict(
        const StateDistribution& prior,
        const ControlT& control
    ) const override;

    std::unique_ptr<StateDistribution> update(
        const StateDistribution& predicted,
        const ObservationT& observation
    ) const override;

private:
    const TransitionModel<StateT, ControlT>& transition_model_;
    const ObservationModel<StateT, ObservationT>& observation_model_;
    const ProposalKernel<StateT, ControlT, ObservationT>& proposal_kernel_;
    const Resampler<StateT>& resampler_;
    double ess_threshold_;
};

} // namespace bayesian_filter

// ---- template implementation ----
#include "particle_filter.tpp"

#pragma once

#include <memory>

#include <pomdp/updater.hpp>
#include <pomdp/pomdp_model.hpp>
#include <pomdp/belief/pf_belief.hpp>
#include <pomdp/particle_filter/proposal_kernel.hpp>
#include <pomdp/particle_filter/resampler.hpp>

namespace pomdp {

/**
 * Particle Filter belief updater.
 *
 * Coordinates:
 *  - proposal (sampling)
 *  - weighting (model likelihoods)
 *  - normalization
 *  - optional resampling
 */
template <typename StateT>
class ParticleFilterUpdater : public BeliefUpdater {
public:
    ParticleFilterUpdater(
        const POMDPModel<StateT>& model,
        const ProposalKernel<StateT>& proposal,
        const Resampler<StateT>& resampler,
        double ess_threshold = 0.5
    );

    std::unique_ptr<Belief> update(
        const Belief& prev,
        const Action& action,
        const Observation& observation,
        const History& history
    ) const override;

private:
    const TransitionKernel<StateT>& T_;
    const ObservationKernel<StateT>& O_;
    const ProposalKernel<StateT>& Q_;
    const Resampler<StateT>& R_;
    double ess_threshold_;
};

} // namespace pomdp

// ---- template implementation ----
#include "particle_filter_updater.tpp"

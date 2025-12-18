#pragma once

#include <memory>
#include <pomdp/updater.hpp>
#include <pomdp/pomdp_model.hpp>
#include <pomdp/particle_filter/pf_belief.hpp>
#include <pomdp/particle_filter/proposal_kernel.hpp>
#include <pomdp/particle_filter/resampler.hpp>

namespace pomdp {

/**
 * Particle Filter belief updater.
 *
 * This class coordinates:
 *  - proposal (sampling)
 *  - weighting (via model kernels)
 *  - normalization
 *  - optional resampling
 *
 * It is templated on StateT because it operates
 * on particle-based beliefs.
 */
template <typename StateT>
class ParticleFilterUpdater : public BeliefUpdater {
public:
    ParticleFilterUpdater(
        const POMDPModel<StateT>& model,
        const ProposalKernel<StateT>& proposal,
        const Resampler<StateT>& resampler,
        double ess_threshold = 0.5
    )
        : T_(model.transition())
        , O_(model.observation())
        , Q_(proposal)
        , R_(resampler)
        , ess_threshold_(ess_threshold)
    {}

    std::unique_ptr<Belief> update(
        const Belief& prev,
        const Action& action,
        const Observation& observation,
        const History& /*history*/
    ) const override
    {
        const auto& b_prev =
            static_cast<const ParticleBelief<StateT>&>(prev);

        auto b_new = std::make_unique<ParticleBelief<StateT>>();
        b_new->particles.reserve(b_prev.particles.size());

        // --- Propagate and weight ---
        for (const auto& p : b_prev.particles) {
            StateT x_new = Q_.sample(p.x, action, observation);

            double log_w =
                T_.transition_log_prob(x_new, p.x, action) +
                O_.observation_log_prob(observation, x_new);

            b_new->particles.push_back(
                Particle<StateT>{x_new, p.weight * std::exp(log_w)}
            );
        }

        // --- Normalize ---
        b_new->normalize();

        // --- Resample if needed ---
        if (b_new->effective_sample_size()
            < ess_threshold_ * b_new->particles.size())
        {
            R_.resample(*b_new);
        }

        return b_new;
    }

private:
    const TransitionKernel<StateT>& T_;
    const ObservationKernel<StateT>& O_;
    const ProposalKernel<StateT>& Q_;
    const Resampler<StateT>& R_;
    double ess_threshold_;
};

} // namespace pomdp

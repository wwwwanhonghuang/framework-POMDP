#include <iostream>
#include <memory>
#include <vector>

#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>
#include <pomdp/history.hpp>
#include <pomdp/belief.hpp>

#include <pomdp/belief/pf_belief.hpp>
#include <pomdp/particle_filter/systematic_resampler.hpp>
#include <pomdp/updater/particle_filter_updater.hpp>

#include "model.hpp"
#include "simple_proposal.hpp"

int main() {
    using StateT = int;

    // ------------------------------------------------------------
    // Configuration (large scale)
    // ------------------------------------------------------------
    const int num_states    = 1000;
    const int num_particles = 5000;
    const int horizon       = 50;

    // ------------------------------------------------------------
    // Construct model
    // ------------------------------------------------------------
    example::DiscreteModel model;

    // ------------------------------------------------------------
    // Construct proposal and resampler
    // ------------------------------------------------------------
    example::SimplePriorProposal<StateT> proposal;
    pomdp::SystematicResampler<StateT> resampler;

    // ------------------------------------------------------------
    // Construct particle filter updater
    // ------------------------------------------------------------
    pomdp::ParticleFilterUpdater<StateT> updater(
        model,
        proposal,
        resampler,
        0.4   // ESS threshold (40%)
    );

    // ------------------------------------------------------------
    // Initialize belief (concrete PF belief)
    // ------------------------------------------------------------
    auto pf_belief = std::make_unique<pomdp::ParticleBelief<StateT>>();
    pf_belief->particles.reserve(num_particles);

    for (int i = 0; i < num_particles; ++i) {
        pf_belief->particles.push_back(
            pomdp::Particle<StateT>{
                i % num_states,
                1.0 / num_particles
            }
        );
    }

    pf_belief->normalize();

    // Upcast once: abstract ownership
    std::unique_ptr<pomdp::Belief> belief = std::move(pf_belief);

    // ------------------------------------------------------------
    // Generate offline action / observation sequence
    // ------------------------------------------------------------
    std::vector<pomdp::Action> actions;
    std::vector<pomdp::Observation> observations;

    actions.reserve(horizon);
    observations.reserve(horizon);

    for (int t = 0; t < horizon; ++t) {
        actions.emplace_back(t % model.num_actions());
        observations.emplace_back((t * 37) % num_states);
    }

    pomdp::History history;

    // ------------------------------------------------------------
    // Offline particle filtering loop
    // ------------------------------------------------------------
    for (int t = 0; t < horizon; ++t) {
        belief = updater.update(
            *belief,
            actions[t],
            observations[t],
            history
        );

        // PF-specific inspection (explicit downcast)
        const auto& pb =
            static_cast<const pomdp::ParticleBelief<StateT>&>(*belief);

        std::cout
            << "t=" << t
            << " obs=" << observations[t].id
            << " particles=" << pb.particles.size()
            << " ESS=" << pb.effective_sample_size()
            << " action=" << actions[t].id
            << std::endl;
    }

    return 0;
}

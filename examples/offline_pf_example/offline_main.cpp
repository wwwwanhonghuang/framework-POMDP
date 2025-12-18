#include <iostream>
#include <memory>
#include <vector>

#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>
#include <pomdp/history.hpp>
#include <pomdp/belief.hpp>

#include <pomdp/particle_filter/pf_belief.hpp>
#include <pomdp/particle_filter/systematic_resampler.hpp>
#include <pomdp/updater/particle_filter_updater.hpp>

#include "model.hpp"
#include "simple_proposal.hpp"

int main() {
    using StateT = int;

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
        0.5   // ESS threshold (50%)
    );

    // ------------------------------------------------------------
    // Initialize belief (concrete PF belief)
    // ------------------------------------------------------------
    const int num_particles = 1000;

    auto pf_belief = std::make_unique<pomdp::ParticleBelief<StateT>>();
    pf_belief->particles.reserve(num_particles);

    for (int i = 0; i < num_particles; ++i) {
        pf_belief->particles.push_back(
            pomdp::Particle<StateT>{
                i % 10,                       // initial state
                1.0 / num_particles           // uniform weight
            }
        );
    }

    pf_belief->normalize();

    // Upcast once: abstract ownership from here on
    std::unique_ptr<pomdp::Belief> belief = std::move(pf_belief);

    // ------------------------------------------------------------
    // Offline action / observation sequence
    // ------------------------------------------------------------
    std::vector<pomdp::Action> actions = {
        pomdp::Action{0},
        pomdp::Action{1},
        pomdp::Action{2}
    };

    std::vector<pomdp::Observation> observations = {
        pomdp::Observation{1},
        pomdp::Observation{2},
        pomdp::Observation{3}
    };

    pomdp::History history;

    // ------------------------------------------------------------
    // Offline particle filtering loop
    // ------------------------------------------------------------
    for (std::size_t t = 0; t < observations.size(); ++t) {
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
            << " ESS=" << pb.effective_sample_size()
            << std::endl;
    }

    return 0;
}

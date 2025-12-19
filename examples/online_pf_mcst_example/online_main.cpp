#include <iostream>
#include <chrono>

#include <Eigen/Dense>

#include <pomdp/belief.hpp>
#include <pomdp/history.hpp>
#include <pomdp/history/sequence_history.hpp>

#include <pomdp/belief/pf_belief.hpp>
#include <pomdp/updater/particle_filter_updater.hpp>

#include <pomdp/planning/planner_runner.hpp>

#include <pomdp/planning/mcst/mcst_planner.hpp>
#include <pomdp/planning/mcst/pomcp_planner.hpp>
#include <pomdp/planning/mcst/ucb_selection.hpp>
#include <pomdp/planning/mcst/progressive_widening.hpp>
#include <pomdp/planning/mcst/composite_selection.hpp>
#include <pomdp/particle_filter/proposal_kernel.hpp>

#include "proposal.hpp"
#include <pomdp/particle_filter/systematic_resampler.hpp>

#include "model.hpp"
#include "action_sampler.hpp"
#include "belief_sampler.hpp"
#include "rollout.hpp"



using namespace online_example;



// [真实系统]
//    ↓ step()
// [observation]
//    ↓
// [Particle Filter belief update]   ← 外部完成
//    ↓
// [belief + history]
//    ↓
// [POMCP / MCST planner]
//    ↓
// [action]



// 在 POMCP / MCST 中，树是 两类节点交替出现的：

// History Node (belief node)
//     ↓ choose action
// Action Node
//     ↓ sample transition
// History Node
//     ↓ choose action
// Action Node
//     ...

// UCB 只在“同一个 history node 下的 action nodes”之间选

int main() {
    // ------------------------------------------------------------
    // 3. Samplers
    // ------------------------------------------------------------
    ContinuousActionSampler action_sampler(/*a_max=*/1.0);
    PFBeliefSampler<State> belief_sampler;


    // ------------------------------------------------------------
    // 1. Model (ground truth + generative)
    // ------------------------------------------------------------
    ContinuousModel model(action_sampler, /*dt=*/0.1);

    // ------------------------------------------------------------
    // 2. Initial belief (particle filter)
    // ------------------------------------------------------------
    constexpr std::size_t NUM_PARTICLES = 500;


    pomdp::ParticleBelief<State> belief;
    belief.particles.reserve(NUM_PARTICLES);

    for (std::size_t i = 0; i < NUM_PARTICLES; ++i) {
        State x(4);
        x.setZero();

        pomdp::Particle<State> p;
        p.x = x;
        p.weight = 1.0 / NUM_PARTICLES;

        belief.particles.push_back(p);
    }

    BootstrapProposal<State> proposal(model);
    pomdp::SystematicResampler<State> resampler;

    // PF updater (uses model's kernels)
    pomdp::ParticleFilterUpdater<State> updater(
        model,
        proposal,
        resampler,
        /*ess_threshold=*/0.5
    );

 

    // ------------------------------------------------------------
    // 4. Rollout policy
    // ------------------------------------------------------------
    Rollout rollout([&]() {
        return action_sampler.sample_random_action();
    });
    // ------------------------------------------------------------
    // 5. Selection strategy (PW + UCB)
    // ------------------------------------------------------------
    auto pw = std::make_shared<pomdp::mcst::ProgressiveWidening>(
        action_sampler,
        /*k=*/1.0,
        /*alpha=*/0.5
    );

    auto ucb = std::make_shared<pomdp::mcst::UCBSelection>(
        /*c=*/1.4
    );

    pomdp::mcst::CompositeSelection selection;
    selection.add_strategy(pw);
    selection.add_strategy(ucb);

    // ------------------------------------------------------------
    // 6. Planner (MCST or POMCP)
    // ------------------------------------------------------------
    constexpr std::size_t HORIZON = 15;
    constexpr double DISCOUNT = 0.95;

    pomdp::mcst::MCSTPlanner<State> planner(
        belief_sampler,
        action_sampler,
        model,
        selection,
        rollout,
        HORIZON,
        DISCOUNT
    );

    // Online / anytime runner
    pomdp::PlannerRunner runner(planner);

    // ------------------------------------------------------------
    // 7. Execution loop (online)
    // ------------------------------------------------------------
    pomdp::SequenceHistory history;

    pomdp::Action action = action_sampler.sample_random_action();


    State true_state(4);
    true_state.setZero();

    constexpr std::size_t NUM_STEPS = 100;

    for (std::size_t t = 0; t < NUM_STEPS; ++t) {
        // --------------------------------------------------------
        // Observe environment
        // --------------------------------------------------------
        auto sim_result = model.step(true_state, action);
        true_state = sim_result.next_state;
        const pomdp::Observation& obs = sim_result.observation;

        // --------------------------------------------------------
        // Belief update (external to planner)
        // --------------------------------------------------------
        const pomdp::Action prev_action = action;

        auto new_belief = updater.update(
            belief,
            prev_action,
            obs,
            history
        );
        belief = *dynamic_cast<pomdp::ParticleBelief<State>*>(new_belief.release());

        // --------------------------------------------------------
        // Online planning (time-bounded)
        // --------------------------------------------------------
        runner.run_for_duration(
            belief,
            history,
            std::chrono::milliseconds(10)
        );

        // --------------------------------------------------------
        // Select best action anytime
        // --------------------------------------------------------
        action = planner.best_action();

        // --------------------------------------------------------
        // Logging
        // --------------------------------------------------------
        const auto& u = action_sampler.action_value(action);

        std::cout
            << "t=" << t
            << "  true_pos=("
            << true_state[0] << ", "
            << true_state[1] << ")"
            << " action=("
            << u[0] << ", "
            << u[1] << ") "
            << std::endl;

        // --------------------------------------------------------
        // Update history
        // --------------------------------------------------------
        
        history.append(action, obs);
    }

    return 0;
}

#include <iostream>
#include <chrono>
#include <random>
#include <Eigen/Dense>

#include <pomdp/belief.hpp>
#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>

#include <pomdp/history.hpp>
#include <pomdp/history/sequence_history.hpp>

#include <pomdp/belief/pf_belief.hpp>
#include <pomdp/adapters/bayesian_filter_adapter.hpp>

#include <pomdp/planning/planner_runner.hpp>
#include <bayesian_filter/particle_filter/particle_filter.hpp>
#include <bayesian_filter/particle_filter/systematic_resampler.hpp>

#include <pomdp/planning/mcst/mcst_planner.hpp>
#include <pomdp/planning/mcst/pomcp_planner.hpp>
#include <pomdp/planning/mcst/ucb_selection.hpp>
#include <pomdp/planning/mcst/progressive_widening.hpp>
#include <pomdp/planning/mcst/composite_selection.hpp>
#include <bayesian_filter/particle_filter/proposal_kernel.hpp>

#include "proposal.hpp"

#include "model.hpp"
#include "action_sampler.hpp"
#include "belief_sampler.hpp"
#include "rollout.hpp"



using namespace online_example;


// 真实系统（state 不可见）
//     ↓ 执行动作
// 观测 o_t
//     ↓
// Belief 更新（Particle Filter）
//     ↓
// 给定 (belief, history)
//     ↓
// Monte Carlo Tree Search 规划
//     ↓
// 选一个动作

class TransitionModelAdapter
    : public bayesian_filter::TransitionModel<
          State,
          pomdp::Action
      >
{
public:
    explicit TransitionModelAdapter(
        const pomdp::TransitionKernel<State>& kernel
    )
        : kernel_(kernel) {}

    double probability(
        const State& next,
        const State& prev,
        const pomdp::Action& action
    ) const override
    {
        // ParticleFilter expects probability, kernel gives log-prob
        return std::exp(
            kernel_.transition_log_prob(next, prev, action)
        );
    }

private:
    const pomdp::TransitionKernel<State>& kernel_;
};

class ObservationModelAdapter
    : public bayesian_filter::ObservationModel<
          State,
          pomdp::Observation
      >
{
public:
    explicit ObservationModelAdapter(
        const pomdp::ObservationKernel<State>& kernel
    )
        : kernel_(kernel) {}

    double probability(
        const pomdp::Observation& obs,
        const State& state
    ) const override
    {
        return std::exp(
            kernel_.observation_log_prob(obs, state)
        );
    }

private:
    const pomdp::ObservationKernel<State>& kernel_;
};

class ProposalKernelAdapter
    : public bayesian_filter::ProposalKernel<
          State,
          pomdp::Action,
          pomdp::Observation
      >
{
public:
    ProposalKernelAdapter(
        const online_example::BootstrapProposal<State, pomdp::Action, pomdp::Observation>& proposal,
        const bayesian_filter::TransitionModel<State, pomdp::Action>& transition
    )
        : proposal_(proposal),
          transition_(transition) {}

    State sample(
        const State& prev_state,
        const pomdp::Action& action,
        const pomdp::Observation& obs
    ) const override
    {
        // Bootstrap proposal ignores observation
        return proposal_.sample(prev_state, action, obs);
    }

    double probability(
        const State& next_state,
        const State& prev_state,
        const pomdp::Action& action,
        const pomdp::Observation& /*obs*/
    ) const override
    {
        // Bootstrap proposal: q = p_transition
        return transition_.probability(next_state, prev_state, action);
    }

private:
    const online_example::BootstrapProposal<State, pomdp::Action, pomdp::Observation>& proposal_;
    const bayesian_filter::TransitionModel<State, pomdp::Action>& transition_;
};


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


    pomdp::POMDPParticleBelief<State> belief;
    belief.particles.reserve(NUM_PARTICLES);

    for (std::size_t i = 0; i < NUM_PARTICLES; ++i) {
        State x(4);
        x.setZero();

        bayesian_filter::ParticleDistribution<State>::Particle p;
        p.state = x;
        p.weight = 1.0 / NUM_PARTICLES;

        belief.particles.push_back(p);
    }

    online_example::BootstrapProposal<State, pomdp::Action, pomdp::Observation> proposal(model);

    std::mt19937 rng(42);
    bayesian_filter::SystematicResampler<State> resampler(rng);

    // PF updater (uses model's kernels)

    double ess_ratio = 0.5;  // or 0.3, 0.7, etc.


    TransitionModelAdapter transition_model(model.transition());
    ObservationModelAdapter observation_model(model.observation());
    ProposalKernelAdapter proposal_adapter(proposal, transition_model);

    bayesian_filter::ParticleFilter<
        State,
        pomdp::Action,
        pomdp::Observation
    > pf(
        transition_model,
        observation_model,
        proposal_adapter,
        resampler,
        ess_ratio
    );


    //pomdp::ParticleBeliefAdapter updater(pf);


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

        // auto new_belief = updater.update(
        //     belief,
        //     prev_action,
        //     obs,
        //     history
        // );
        // belief = *dynamic_cast<pomdp::POMDPParticleBelief<State>*>(new_belief.release());

        auto next = pf.step(belief, action, obs);

        belief = *static_cast<
            pomdp::POMDPParticleBelief<State>*
        >(next.release());

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

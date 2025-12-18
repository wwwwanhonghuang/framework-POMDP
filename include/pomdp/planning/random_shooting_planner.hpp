#pragma once

#include <limits>
#include <memory>

#include <pomdp/planning/planner.hpp>
#include <pomdp/planning/belief_sampler.hpp>
#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/simulator.hpp>


// Random Shooting
//    ↓
// One-step lookahead tree (depth-1 MCTS)
//    ↓
// UCT (deterministic state)
//    ↓
// POMCP (belief → particles)
//    ↓
// POMCPOW (continuous actions)
namespace pomdp {

template <typename StateT>
class RandomShootingPlanner : public Planner {
public:
    RandomShootingPlanner(
        const BeliefSampler<StateT>& belief_sampler,
        const ActionSampler& action_sampler,
        const Simulator<StateT>& simulator,
        std::size_t num_action_samples,
        std::size_t horizon,
        double discount = 1.0
    )
        : belief_sampler_(belief_sampler)
        , action_sampler_(action_sampler)
        , simulator_(simulator)
        , num_action_samples_(num_action_samples)
        , horizon_(horizon)
        , discount_(discount)
    {}

    Action decide(
        const Belief& belief,
        const History& /*history*/
    ) override
    {
        Action best_action;
        double best_value = -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < num_action_samples_; ++i) {
            Action a = action_sampler_.sample_action(belief);

            StateT x = belief_sampler_.sample_state(belief);
            double value = rollout(x, a);

            if (value > best_value) {
                best_value = value;
                best_action = a;
            }
        }

        return best_action;
    }

private:
    double rollout(StateT state, const Action& first_action) const {
        double total_reward = 0.0;
        double gamma = 1.0;

        Action a = first_action;

        for (std::size_t t = 0; t < horizon_; ++t) {
            auto result = simulator_.step(state, a);

            total_reward += gamma * result.reward;
            gamma *= discount_;

            state = result.next_state;

            // future actions are resampled (random shooting)
            a = action_sampler_.sample_action_from_state(state);
        }

        return total_reward;
    }

private:
    const BeliefSampler<StateT>& belief_sampler_;
    const ActionSampler& action_sampler_;
    const Simulator<StateT>& simulator_;

    std::size_t num_action_samples_;
    std::size_t horizon_;
    double discount_;
};

} // namespace pomdp

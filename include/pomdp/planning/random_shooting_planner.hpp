#pragma once

#include <cstddef>

#include <pomdp/planning/planner.hpp>
#include <pomdp/planning/belief_sampler.hpp>
#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/simulator.hpp>

namespace pomdp {

/**
 * Random Shooting Planner
 *
 * One-step lookahead planner:
 *  - sample actions from belief
 *  - sample states from belief
 *  - perform fixed-horizon rollouts
 *
 * This is the minimal ancestor of:
 *   Random Shooting → UCT → POMCP → POMCPOW
 */
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
    );

    Action decide(
        const Belief& belief,
        const History& history
    ) override;

private:
    double rollout(StateT state, const Action& first_action) const;

private:
    const BeliefSampler<StateT>& belief_sampler_;
    const ActionSampler& action_sampler_;
    const Simulator<StateT>& simulator_;

    std::size_t num_action_samples_;
    std::size_t horizon_;
    double discount_;
};

} // namespace pomdp

// ---- template implementation ----
#include "random_shooting_planner.tpp"

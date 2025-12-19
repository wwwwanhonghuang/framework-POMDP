#pragma once

#include <cstddef>

#include <pomdp/planning/planner.hpp>
#include <pomdp/planning/belief_sampler.hpp>
#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/simulator.hpp>

#include <pomdp/planning/mcst/mcst_selection.hpp>
#include <pomdp/planning/mcst/rollout_policy.hpp>
#include <pomdp/planning/mcst/pomcp_node.hpp>

namespace pomdp::mcst {

/**
 * @brief POMCP planner adapter with pluggable rollout and selection.
 *
 * Responsibilities:
 *  - perform ONE POMCP simulation per call
 *  - handle action → observation branching
 *  - delegate rollout to RolloutPolicy
 *
 * Does NOT:
 *  - update belief
 *  - own execution loop
 *  - hard-code UCB / PW / rollout
 */
template <typename StateT>
class PomcpPlanner : public Planner {
public:
    PomcpPlanner(
        const BeliefSampler<StateT>& belief_sampler,
        const ActionSampler& action_sampler,
        const Simulator<StateT>& simulator,
        const SelectionStrategy& selection,
        const RolloutPolicy<StateT>& rollout_policy,
        std::size_t horizon,
        double discount = 1.0
    );

    /**
     * @brief Perform ONE POMCP simulation.
     */
    void run_simulation(const Belief& belief);

    /**
     * @brief Anytime best action at root.
     */
    Action best_action() const;

    /**
     * @brief Planner interface adapter.
     *
     * Default behavior: one simulation per call.
     */
    Action decide(
        const Belief& belief,
        const History& history
    ) override;

private:
    /**
     * @brief Recursive POMCP simulation.
     */
    double simulate(
        PomcpNode& node,
        StateT& state,
        std::size_t depth
    );

    Action select_or_expand(PomcpNode& node);

private:
    const BeliefSampler<StateT>& belief_sampler_;
    const ActionSampler& action_sampler_;
    const Simulator<StateT>& simulator_;

    const SelectionStrategy& selection_;
    const RolloutPolicy<StateT>& rollout_policy_;

    std::size_t horizon_;
    double discount_;

    PomcpNode root_;

    // Placeholder belief for action sampling only
    Belief dummy_belief_;
};

} // namespace pomdp::mcst

// ---- template implementation ----
#include "pomcp_planner.tpp"

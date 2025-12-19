#pragma once

#include <cstddef>
#include <memory>

#include <pomdp/planning/planner.hpp>
#include <pomdp/planning/belief_sampler.hpp>
#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/simulator.hpp>

#include <pomdp/planning/mcst/mcst_node.hpp>
#include <pomdp/planning/mcst/mcst_selection.hpp>
#include <pomdp/planning/mcst/rollout_policy.hpp>

namespace pomdp::mcst {

/**
 * @brief MCST-based planner adapter (pluggable selection & rollout).
 *
 * Responsibilities:
 *  - perform ONE MCST simulation per call
 *  - delegate action choice to SelectionStrategy
 *  - delegate rollout to RolloutPolicy
 *
 * Does NOT:
 *  - own execution loop
 *  - perform belief update
 *  - hard-code UCB / PW / rollout
 */
template <typename StateT>
class MCSTPlanner : public Planner {
public:
    MCSTPlanner(
        const BeliefSampler<StateT>& belief_sampler,
        const ActionSampler& action_sampler,
        const Simulator<StateT>& simulator,
        const SelectionStrategy& selection,
        const RolloutPolicy<StateT>& rollout_policy,
        std::size_t horizon,
        double discount = 1.0
    );

    /**
     * @brief Perform ONE MCST simulation.
     */
    void run_simulation(const Belief& belief);

    /**
     * @brief Anytime best action.
     */
    Action best_action() const;

    /**
     * @brief Planner interface adapter.
     *
     * Default: one simulation per call.
     */
    Action decide(
        const Belief& belief,
        const History& history
    ) override;

private:
    Action select_or_expand_root(const Belief& belief);

private:
    const BeliefSampler<StateT>& belief_sampler_;
    const ActionSampler& action_sampler_;
    const Simulator<StateT>& simulator_;

    const SelectionStrategy& selection_;
    const RolloutPolicy<StateT>& rollout_policy_;

    std::size_t horizon_;
    double discount_;

    std::unique_ptr<Node> root_;
};

} // namespace pomdp::mcst

// ---- template implementation ----
#include "mcst_planner.tpp"

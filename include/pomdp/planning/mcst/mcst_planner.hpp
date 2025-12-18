#pragma once

#include <limits>

#include <pomdp/planning/planner.hpp>
#include <pomdp/planning/belief_sampler.hpp>
#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/simulator.hpp>

#include <pomdp/planning/mcst/mcst_node.hpp>
#include <pomdp/planning/mcst/mcst_selection.hpp>
#include <pomdp/planning/mcst/rollout_policy.hpp>

namespace pomdp::mcst {

/**
 * @brief MCST-based planner adapter (with pluggable rollout & selection).
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
    )
        : belief_sampler_(belief_sampler)
        , action_sampler_(action_sampler)
        , simulator_(simulator)
        , selection_(selection)
        , rollout_policy_(rollout_policy)
        , horizon_(horizon)
        , discount_(discount)
    {}

    /**
     * @brief Perform ONE MCST simulation.
     */
    void run_simulation(const Belief& belief) {
        // 1. sample hypothetical state
        StateT state = belief_sampler_.sample_state(belief);

        // 2. select or expand root action
        Action action = select_or_expand_root(belief);

        // 3. rollout via injected policy
        double value = rollout_policy_.rollout(
            state,
            action,
            simulator_,
            horizon_,
            discount_
        );

        // 4. update statistics
        auto& entry = root_->ensure_action(action);
        entry.stats.update(value);
        root_->update(value);

    }

    /**
     * @brief Anytime best action.
     */
    Action best_action() const {
        Action best;
        double best_value = -std::numeric_limits<double>::infinity();

        for (const auto& [action, entry] : root_->actions) {
            if (entry.stats.visits == 0) continue;

            double mean = entry.stats.mean();
            if (mean > best_value) {
                best_value = mean;
                best = action;
            }
        }

        return best;
    }

    /**
     * @brief Planner interface adapter.
     *
     * Default: one simulation per call.
     */
    Action decide(
        const Belief& belief,
        const History& history
    ) override {
        // Lazily (re)initialize root for current belief/history
        if (!root_) {
            // root_ = std::make_unique<Node>(belief, history);
            root_ = std::make_unique<Node>();

        }

        run_simulation(belief);
        return best_action();
    }


private:
    Action select_or_expand_root(const Belief& belief) {
        // 1) expansion (PW etc.)
        if (auto a = selection_.propose_expansion(*root_)) {
            root_->ensure_action(*a);
            return *a;
        }

        // 2) selection among existing actions (UCB etc.)
        if (!root_->actions.empty()) {
            return selection_.select_existing(*root_);
        }

        // 3) fallback: sample new action
        Action a = action_sampler_.sample_action(belief);
        root_->ensure_action(a);
        return a;
    }

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

#pragma once

#include <limits>

namespace pomdp::mcst {

template <typename StateT>
MCSTPlanner<StateT>::MCSTPlanner(
    const BeliefSampler<StateT>& belief_sampler,
    const ActionSampler& action_sampler,
    const Simulator<StateT>& simulator,
    const SelectionStrategy& selection,
    const RolloutPolicy<StateT>& rollout_policy,
    std::size_t horizon,
    double discount
)
    : belief_sampler_(belief_sampler)
    , action_sampler_(action_sampler)
    , simulator_(simulator)
    , selection_(selection)
    , rollout_policy_(rollout_policy)
    , horizon_(horizon)
    , discount_(discount)
{}

template <typename StateT>
void MCSTPlanner<StateT>::run_simulation(
    const Belief& belief
)
{
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

template <typename StateT>
Action MCSTPlanner<StateT>::best_action() const
{
    Action best;
    double best_value = -std::numeric_limits<double>::infinity();

    for (const auto& [action, entry] : root_->actions) {
        if (entry.stats.visits == 0) {
            continue;
        }

        double mean = entry.stats.mean();
        if (mean > best_value) {
            best_value = mean;
            best = action;
        }
    }

    return best;
}

template <typename StateT>
Action MCSTPlanner<StateT>::decide(
    const Belief& belief,
    const History& /*history*/
)
{
    // Lazily initialize root
    if (!root_) {
        root_ = std::make_unique<Node>();
    }

    run_simulation(belief);
    return best_action();
}

template <typename StateT>
Action MCSTPlanner<StateT>::select_or_expand_root(
    const Belief& belief
)
{
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

} // namespace pomdp::mcst

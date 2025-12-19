#pragma once

#include <limits>

namespace pomdp::mcst {

template <typename StateT>
PomcpPlanner<StateT>::PomcpPlanner(
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
void PomcpPlanner<StateT>::run_simulation(
    const Belief& belief
)
{
    StateT state = belief_sampler_.sample_state(belief);
    simulate(root_, state, 0);
}

template <typename StateT>
Action PomcpPlanner<StateT>::best_action() const
{
    Action best;
    double best_value = -std::numeric_limits<double>::infinity();

    for (const auto& [action, entry] : root_.action_node.actions) {
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
Action PomcpPlanner<StateT>::decide(
    const Belief& belief,
    const History& /*history*/
)
{
    run_simulation(belief);
    return best_action();
}

template <typename StateT>
double PomcpPlanner<StateT>::simulate(
    PomcpNode& node,
    StateT& state,
    std::size_t depth
)
{
    if (depth >= horizon_) {
        return 0.0;
    }

    // 1) select or expand action
    Action action = select_or_expand(node);

    // 2) environment step
    auto result = simulator_.step(state, action);

    // 3) observation branching
    auto child = node.ensure_child(action, result.observation);

    // 4) rollout from next state
    double future_return = rollout_policy_.rollout(
        result.next_state,
        action,
        simulator_,
        horizon_ - depth - 1,
        discount_
    );

    double total_return =
        result.reward + discount_ * future_return;

    // 5) update statistics
    auto& entry = node.action_node.ensure_action(action);
    entry.stats.update(total_return);
    node.action_node.update(total_return);

    state = result.next_state;
    return total_return;
}

template <typename StateT>
Action PomcpPlanner<StateT>::select_or_expand(
    PomcpNode& node
)
{
    // Expansion (progressive widening etc.)
    if (auto a = selection_.propose_expansion(node.action_node)) {
        node.action_node.ensure_action(*a);
        return *a;
    }

    // Selection among existing actions (UCB etc.)
    if (!node.action_node.actions.empty()) {
        return selection_.select_existing(node.action_node);
    }

    // Fallback: sample new action
    Action a = action_sampler_.sample_action(dummy_belief_);
    node.action_node.ensure_action(a);
    return a;
}

} // namespace pomdp::mcst

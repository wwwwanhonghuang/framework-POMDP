#pragma once

namespace pomdp::mcst {

template <typename StateT>
RandomRollout<StateT>::RandomRollout(
    std::function<Action()> sample_action
)
    : sample_action_(std::move(sample_action))
{}

template <typename StateT>
double RandomRollout<StateT>::rollout(
    StateT state,
    const Action& first_action,
    const Simulator<StateT>& simulator,
    std::size_t horizon,
    double discount
) const
{
    double total_reward = 0.0;
    double gamma = 1.0;

    Action action = first_action;

    for (std::size_t t = 0; t < horizon; ++t) {
        auto result = simulator.step(state, action);

        total_reward += gamma * result.reward;
        gamma *= discount;

        state = result.next_state;

        // belief-free random action
        action = sample_action_();
    }

    return total_reward;
}

} // namespace pomdp::mcst

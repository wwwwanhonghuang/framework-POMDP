#pragma once

#include <limits>

namespace pomdp {

template <typename StateT>
RandomShootingPlanner<StateT>::RandomShootingPlanner(
    const BeliefSampler<StateT>& belief_sampler,
    const ActionSampler& action_sampler,
    const Simulator<StateT>& simulator,
    std::size_t num_action_samples,
    std::size_t horizon,
    double discount
)
    : belief_sampler_(belief_sampler)
    , action_sampler_(action_sampler)
    , simulator_(simulator)
    , num_action_samples_(num_action_samples)
    , horizon_(horizon)
    , discount_(discount)
{}

template <typename StateT>
Action RandomShootingPlanner<StateT>::decide(
    const Belief& belief,
    const History& /*history*/
)
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

template <typename StateT>
double RandomShootingPlanner<StateT>::rollout(
    StateT state,
    const Action& first_action
) const
{
    double total_reward = 0.0;
    double gamma = 1.0;

    Action a = first_action;

    for (std::size_t t = 0; t < horizon_; ++t) {
        auto result = simulator_.step(state, a);

        total_reward += gamma * result.reward;
        gamma *= discount_;

        state = result.next_state;

        // random shooting: resample action each step
        a = action_sampler_.sample_action_from_state(state);
    }

    return total_reward;
}

} // namespace pomdp

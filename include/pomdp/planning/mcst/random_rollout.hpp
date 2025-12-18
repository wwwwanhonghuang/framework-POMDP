#pragma once

#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/mcst/rollout_policy.hpp>

namespace pomdp::mcst {

/**
 * @brief Random rollout policy.
 *
 * This rollout:
 *  - samples actions randomly using ActionSampler
 *  - simulates forward for a fixed horizon
 *  - accumulates discounted rewards
 *
 * This is the canonical baseline rollout for MCST / POMCP.
 */
template <typename StateT>
class RandomRollout : public RolloutPolicy<StateT> {
public:
    explicit RandomRollout(
        const std::function<pomdp::Action()>& sample_action
    )
        : sample_action_(sample_action)
    {}

    double rollout(
        StateT state,
        const Action& first_action,
        const Simulator<StateT>& simulator,
        std::size_t horizon,
        double discount
    ) const override
    {
        double total_reward = 0.0;
        double gamma = 1.0;

        Action action = first_action;

        

        for (std::size_t t = 0; t < horizon; ++t) {
            auto result = simulator.step(state, action);

            total_reward += gamma * result.reward;
            gamma *= discount;

            state = result.next_state;

            action = sample_action_();  // ← belief-free
        }


        return total_reward;
    }

private:    
    std::function<pomdp::Action()> sample_action_;
};

} // namespace pomdp::mcst

#pragma once

#include <functional>
#include <cstddef>

#include <pomdp/action.hpp>
#include <pomdp/planning/mcst/rollout_policy.hpp>

namespace pomdp {
template <typename StateT>
class Simulator;
}

namespace pomdp::mcst {

/**
 * @brief Random rollout policy.
 *
 * This rollout:
 *  - samples actions randomly via an injected sampler
 *  - simulates forward for a fixed horizon
 *  - accumulates discounted rewards
 *
 * Canonical baseline rollout for MCST / POMCP.
 */
template <typename StateT>
class RandomRollout : public RolloutPolicy<StateT> {
public:
    explicit RandomRollout(
        std::function<Action()> sample_action
    );

    double rollout(
        StateT state,
        const Action& first_action,
        const Simulator<StateT>& simulator,
        std::size_t horizon,
        double discount
    ) const override;

private:
    std::function<Action()> sample_action_;
};

} // namespace pomdp::mcst

// ---- template implementation ----
#include "random_rollout.tpp"

#pragma once

#include <optional>

#include <pomdp/action.hpp>
#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/mcst/mcst_selection.hpp>

namespace pomdp::mcst {

/**
 * @brief Progressive Widening (PW) strategy.
 *
 * Expansion rule:
 *
 *   |A(s)| < k * N(s)^alpha
 *
 * where:
 *  - |A(s)| : number of expanded actions
 *  - N(s)   : visit count of the node
 *  - k, α   : hyperparameters
 *
 * This strategy:
 *  - ONLY proposes expansion
 *  - does NOT select among existing actions
 */
class ProgressiveWidening : public SelectionStrategy {
public:
    ProgressiveWidening(
        const ActionSampler& action_sampler,
        double k = 1.0,
        double alpha = 0.5
    );

    /**
     * @brief PW never selects existing actions.
     *
     * Must be composed with a selector (e.g. UCB).
     */
    Action select_existing(
        const Node& node
    ) const override;

    /**
     * @brief Decide whether to expand a new action.
     */
    std::optional<Action> propose_expansion(
        const Node& node
    ) const override;

private:
    const ActionSampler& action_sampler_;
    double k_;
    double alpha_;

    // Placeholder belief for action sampling
    Belief dummy_belief_;
};

} // namespace pomdp::mcst

#pragma once

#include <cmath>
#include <optional>

#include <pomdp/action.hpp>
#include <pomdp/planning/action_sampler.hpp>
#include <pomdp/planning/mcst/mcst_selection.hpp>

namespace pomdp::mcst {

/**
 * @brief Progressive Widening (PW) strategy.
 *
 * Allows action expansion according to:
 *
 *   |A(s)| < k * N(s)^alpha
 *
 * where:
 *  - |A(s)| : number of expanded actions
 *  - N(s)   : visit count of the node
 *  - k, α   : hyperparameters
 *
 * This strategy:
 *  - only proposes expansion
 *  - does NOT select among existing actions
 */
class ProgressiveWidening : public SelectionStrategy {
public:
    ProgressiveWidening(
        const ActionSampler& action_sampler,
        double k = 1.0,
        double alpha = 0.5
    )
        : action_sampler_(action_sampler)
        , k_(k)
        , alpha_(alpha)
    {}

    /**
     * @brief PW never selects existing actions.
     *
     * Selection should be handled by another strategy (e.g. UCB).
     */
    Action select_existing(const Node& /*node*/) const override {
        // This should never be called.
        // Use a composite strategy with UCB.
        throw std::logic_error(
            "ProgressiveWidening does not select existing actions."
        );
    }

    /**
     * @brief Decide whether to expand a new action.
     */
    std::optional<Action> propose_expansion(
        const Node& node
    ) const override
    {
        const std::size_t num_actions = node.actions.size();
        const std::size_t visits = node.visits + 1; // safety

        const double threshold = k_ * std::pow(visits, alpha_);

        if (static_cast<double>(num_actions) < threshold) {
            return action_sampler_.sample_action(dummy_belief_);
        }

        return std::nullopt;
    }

private:
    const ActionSampler& action_sampler_;
    double k_;
    double alpha_;

    // Placeholder belief for action sampling
    Belief dummy_belief_;
};

} // namespace pomdp::mcst

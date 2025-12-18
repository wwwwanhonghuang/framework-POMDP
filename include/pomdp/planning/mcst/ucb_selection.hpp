#pragma once

#include <cmath>
#include <limits>

#include <pomdp/action.hpp>
#include <pomdp/planning/mcst/mcst_selection.hpp>

namespace pomdp::mcst {

/**
 * @brief UCB (UCT) selection strategy for MCST.
 *
 * This strategy:
 *  - selects among existing actions using UCB
 *  - does NOT create new actions
 *  - assumes expansion is handled elsewhere
 *
 * UCB score:
 *   mean_value + c * sqrt( log(parent_visits) / action_visits )
 */
class UCBSelection : public SelectionStrategy {
public:
    explicit UCBSelection(double exploration_constant = 1.4)
        : c_(exploration_constant)
    {}

    Action select_existing(const Node& node) const override {
        Action best_action;
        double best_score = -std::numeric_limits<double>::infinity();

        const double parent_visits =
            static_cast<double>(node.visits + 1); // +1 for numerical safety

        for (const auto& [action, entry] : node.actions) {
            const auto& stats = entry.stats;

            // Skip unvisited actions (should be expanded separately)
            if (stats.visits == 0) {
                continue;
            }

            const double mean = stats.mean();
            const double exploration =
                c_ * std::sqrt(std::log(parent_visits) / stats.visits);

            const double score = mean + exploration;

            if (score > best_score) {
                best_score = score;
                best_action = action;
            }
        }

        return best_action;
    }

private:
    double c_;
};

} // namespace pomdp::mcst

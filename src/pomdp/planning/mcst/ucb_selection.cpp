#include <cmath>
#include <limits>

#include <pomdp/planning/mcst/ucb_selection.hpp>

namespace pomdp::mcst {

UCBSelection::UCBSelection(double exploration_constant)
    : c_(exploration_constant)
{}

Action UCBSelection::select_existing(
    const Node& node
) const
{
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

} // namespace pomdp::mcst

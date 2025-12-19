#pragma once

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
    explicit UCBSelection(double exploration_constant = 1.4);

    Action select_existing(
        const Node& node
    ) const override;

private:
    double c_;
};

} // namespace pomdp::mcst

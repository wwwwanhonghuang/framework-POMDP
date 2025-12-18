#pragma once

#include <optional>

#include <pomdp/action.hpp>
#include <pomdp/planning/mcst/mcst_node.hpp>

namespace pomdp::mcst {

/**
 * @brief Selection / expansion strategy for MCST.
 *
 * This strategy:
 *  - selects an existing action at a node
 *  - may propose expansion of a new action
 *  - is purely internal to search
 *
 * NOT a POMDP policy.
 * NOT an agent decision rule.
 */
class SelectionStrategy {
public:
    virtual ~SelectionStrategy() = default;

    /**
     * @brief Select an action among existing branches.
     */
    virtual Action select_existing(
        const Node& node
    ) const = 0;

    /**
     * @brief Optionally propose a new action to expand.
     *
     * Used for progressive widening.
     */
    virtual std::optional<Action> propose_expansion(
        const Node& node
    ) const {
        return std::nullopt;
    }
};

} // namespace pomdp::mcst

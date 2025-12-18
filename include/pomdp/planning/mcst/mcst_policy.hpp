#pragma once

#include <optional>

#include <pomdp/action.hpp>
#include <pomdp/planning/mcst/mcst_node.hpp>

namespace pomdp::mcst {

/**
 * @brief Selection / expansion policy for MCST.
 *
 * This policy:
 *  - decides which action to take at a node
 *  - may decide to expand a new action
 *  - does NOT execute rollouts
 *  - does NOT own tree memory
 *
 * Typical implementations:
 *  - UCB / UCT
 *  - epsilon-greedy
 *  - progressive widening
 *  - learned (NN-based) policies
 */
class Policy {
public:
    virtual ~Policy() = default;

    /**
     * @brief Select an action from a node.
     *
     * @param node     Current MCST node (read-only)
     * @return         Selected action (must exist in node or be expanded)
     */
    virtual Action select_action(
        const Node& node
    ) const = 0;

    /**
     * @brief Decide whether a new action should be expanded.
     *
     * Returning std::nullopt means "do not expand".
     * Returning an Action means "add this action to the node".
     *
     * This hook enables progressive widening.
     */
    virtual std::optional<Action> propose_expansion(
        const Node& node
    ) const {
        return std::nullopt;
    }
};

} // namespace pomdp::mcst

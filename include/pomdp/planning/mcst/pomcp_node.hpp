#pragma once

#include <memory>
#include <unordered_map>

#include <pomdp/observation.hpp>
#include <pomdp/planning/mcst/mcst_node.hpp>

namespace pomdp::mcst {

/**
 * @brief POMCP node with observation branching.
 *
 * Structure:
 *
 *   Action
 *     └── Observation
 *           └── PomcpNode
 *
 * This node:
 *  - reuses MCST Node for action statistics
 *  - adds observation-indexed children
 *  - does NOT store belief
 *  - does NOT perform belief update
 *  - does NOT own rollout or selection logic
 */
struct PomcpNode {
    using Ptr = std::shared_ptr<PomcpNode>;

    /**
     * @brief Action-level statistics (shared with MCST).
     */
    Node action_node;

    /**
     * @brief Observation branching.
     */
    struct ObservationChildren {
        std::unordered_map<Observation, Ptr> next;
    };

    /**
     * @brief Map:
     *   Action → ObservationChildren
     */
    std::unordered_map<Action, ObservationChildren> obs_children;

    /**
     * @brief Ensure a child node exists for (action, observation).
     */
    Ptr ensure_child(
        const Action& action,
        const Observation& observation
    );

    /**
     * @brief Get child node for (action, observation), or nullptr.
     */
    Ptr get_child(
        const Action& action,
        const Observation& observation
    ) const;
};

} // namespace pomdp::mcst

#pragma once

#include <unordered_map>
#include <memory>

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
     *
     * For each (action, observation) pair, we get a child node.
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
    ) {
        auto& by_obs = obs_children[action].next;
        auto it = by_obs.find(observation);

        if (it == by_obs.end()) {
            auto child = std::make_shared<PomcpNode>();
            by_obs.emplace(observation, child);
            return child;
        }

        return it->second;
    }

    /**
     * @brief Get child node for (action, observation), or nullptr.
     */
    Ptr get_child(
        const Action& action,
        const Observation& observation
    ) const {
        auto act_it = obs_children.find(action);
        if (act_it == obs_children.end()) {
            return nullptr;
        }

        const auto& by_obs = act_it->second.next;
        auto obs_it = by_obs.find(observation);
        if (obs_it == by_obs.end()) {
            return nullptr;
        }

        return obs_it->second;
    }
};

} // namespace pomdp::mcst

#pragma once

#include <unordered_map>
#include <memory>

#include <pomdp/action.hpp>
#include <pomdp/planning/mcst/mcst_statistics.hpp>

namespace pomdp::mcst {

/**
 * @brief A generic Monte-Carlo Search Tree node.
 *
 * This node:
 *  - stores statistics per action
 *  - does NOT know about belief, state, observation, or dynamics
 *  - can be reused for depth-1 or deeper trees
 *
 * Children are optional and may be unused in depth-1 MCST.
 */
struct Node {
    using Ptr = std::shared_ptr<Node>;

    struct ActionEntry {
        Statistics stats;
        Ptr child = nullptr;   // unused in depth-1 MCST
    };

    std::unordered_map<Action, ActionEntry> actions;
    std::size_t visits = 0;

    void update(double value) {
        ++visits;
    }

    ActionEntry& ensure_action(const Action& action) {
        return actions[action]; // default-construct if missing
    }
};

} // namespace pomdp::mcst

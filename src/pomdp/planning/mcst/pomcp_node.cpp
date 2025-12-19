#include <pomdp/planning/mcst/pomcp_node.hpp>

namespace pomdp::mcst {

PomcpNode::Ptr PomcpNode::ensure_child(
    const Action& action,
    const Observation& observation
)
{
    auto& by_obs = obs_children[action].next;
    auto it = by_obs.find(observation);

    if (it == by_obs.end()) {
        auto child = std::make_shared<PomcpNode>();
        by_obs.emplace(observation, child);
        return child;
    }

    return it->second;
}

PomcpNode::Ptr PomcpNode::get_child(
    const Action& action,
    const Observation& observation
) const
{
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

} // namespace pomdp::mcst

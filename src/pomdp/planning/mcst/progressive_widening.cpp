#include <cmath>
#include <stdexcept>

#include <pomdp/planning/mcst/progressive_widening.hpp>

namespace pomdp::mcst {

ProgressiveWidening::ProgressiveWidening(
    const ActionSampler& action_sampler,
    double k,
    double alpha
)
    : action_sampler_(action_sampler)
    , k_(k)
    , alpha_(alpha)
{}

Action ProgressiveWidening::select_existing(
    const Node& /*node*/
) const
{
    throw std::logic_error(
        "ProgressiveWidening does not select existing actions."
    );
}

std::optional<Action>
ProgressiveWidening::propose_expansion(
    const Node& node
) const
{
    const std::size_t num_actions = node.actions.size();
    const std::size_t visits = node.visits + 1; // numerical safety

    const double threshold =
        k_ * std::pow(static_cast<double>(visits), alpha_);

    if (static_cast<double>(num_actions) < threshold) {
        return action_sampler_.sample_action(dummy_belief_);
    }

    return std::nullopt;
}

} // namespace pomdp::mcst

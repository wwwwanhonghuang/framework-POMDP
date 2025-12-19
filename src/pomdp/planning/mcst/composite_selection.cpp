#include <stdexcept>

#include <pomdp/planning/mcst/composite_selection.hpp>

namespace pomdp::mcst {

CompositeSelection::CompositeSelection() = default;

CompositeSelection::CompositeSelection(
    std::vector<std::shared_ptr<SelectionStrategy>> strategies
)
    : strategies_(std::move(strategies))
{}

void CompositeSelection::add_strategy(
    std::shared_ptr<SelectionStrategy> strategy
)
{
    strategies_.push_back(std::move(strategy));
}

std::optional<Action>
CompositeSelection::propose_expansion(
    const Node& node
) const
{
    for (const auto& s : strategies_) {
        if (auto a = s->propose_expansion(node)) {
            return a;
        }
    }
    return std::nullopt;
}

Action CompositeSelection::select_existing(
    const Node& node
) const
{
    for (const auto& s : strategies_) {
        try {
            return s->select_existing(node);
        } catch (...) {
            // ignore and try next strategy
        }
    }

    throw std::logic_error(
        "CompositeSelection: no strategy could select an action."
    );
}

} // namespace pomdp::mcst

#pragma once

#include <vector>
#include <memory>
#include <optional>

#include <pomdp/planning/mcst/mcst_selection.hpp>

namespace pomdp::mcst {

/**
 * @brief Composite selection strategy.
 *
 * Composition rule:
 *  1) Try expansion strategies in order.
 *     - First strategy that proposes an action wins.
 *  2) If no expansion is proposed, fall back to selection strategies in order.
 *
 * Typical usage:
 *  CompositeSelection{
 *      ProgressiveWidening{sampler, k, alpha},
 *      UCBSelection{c}
 *  }
 */
class CompositeSelection : public SelectionStrategy {
public:
    CompositeSelection() = default;

    explicit CompositeSelection(
        std::vector<std::shared_ptr<SelectionStrategy>> strategies
    )
        : strategies_(std::move(strategies))
    {}

    /**
     * @brief Add a strategy (order matters).
     */
    void add_strategy(std::shared_ptr<SelectionStrategy> strategy) {
        strategies_.push_back(std::move(strategy));
    }

    /**
     * @brief Attempt expansion first.
     */
    std::optional<Action> propose_expansion(
        const Node& node
    ) const override
    {
        for (const auto& s : strategies_) {
            if (auto a = s->propose_expansion(node)) {
                return a;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Select among existing actions.
     *
     * The first strategy capable of selection decides.
     */
    Action select_existing(
        const Node& node
    ) const override
    {
        for (const auto& s : strategies_) {
            // Convention:
            // strategies that don't implement selection should throw
            // or be ordered after a selector (e.g., UCB).
            try {
                return s->select_existing(node);
            } catch (...) {
                // ignore and try next
            }
        }

        throw std::logic_error(
            "CompositeSelection: no strategy could select an action."
        );
    }

private:
    std::vector<std::shared_ptr<SelectionStrategy>> strategies_;
};

} // namespace pomdp::mcst

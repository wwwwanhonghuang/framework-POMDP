#pragma once

#include <memory>
#include <optional>
#include <vector>

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
    CompositeSelection();

    explicit CompositeSelection(
        std::vector<std::shared_ptr<SelectionStrategy>> strategies
    );

    /**
     * @brief Add a strategy (order matters).
     */
    void add_strategy(std::shared_ptr<SelectionStrategy> strategy);

    /**
     * @brief Attempt expansion first.
     */
    std::optional<Action> propose_expansion(
        const Node& node
    ) const override;

    /**
     * @brief Select among existing actions.
     *
     * The first strategy capable of selection decides.
     */
    Action select_existing(
        const Node& node
    ) const override;

private:
    std::vector<std::shared_ptr<SelectionStrategy>> strategies_;
};

} // namespace pomdp::mcst

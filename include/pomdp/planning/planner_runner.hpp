#pragma once

#include <chrono>
#include <cstddef>

#include <pomdp/planning/planner.hpp>

namespace pomdp {

/**
 * @brief Online / anytime planner runner.
 *
 * Responsibilities:
 *  - manage execution loop
 *  - enforce time / iteration budgets
 *  - support anytime interruption
 *
 * The planner itself:
 *  - performs one simulation per decide()
 *  - accumulates statistics internally
 */
class PlannerRunner {
public:
    explicit PlannerRunner(Planner& planner);

    Action run_iterations(
        const Belief& belief,
        const History& history,
        std::size_t max_iterations
    );

    Action run_for_duration(
        const Belief& belief,
        const History& history,
        std::chrono::steady_clock::duration duration
    );

    /**
     * @brief Anytime query (one simulation).
     */
    Action step(
        const Belief& belief,
        const History& history
    );

private:
    Planner& planner_;
};

} // namespace pomdp

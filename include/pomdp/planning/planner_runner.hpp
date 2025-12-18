#pragma once

#include <chrono>
#include <cstddef>
#include <limits>

#include <pomdp/planning/planner.hpp>

namespace pomdp {

/**
 * @brief Online / anytime planner runner.
 *
 * The runner:
 *  - owns the execution loop
 *  - controls time or iteration budget
 *  - can interrupt planning at any time
 *
 * The planner:
 *  - performs one simulation per call
 *  - accumulates statistics internally
 */
class PlannerRunner {
public:
    explicit PlannerRunner(Planner& planner)
        : planner_(planner)
    {}

    /**
     * @brief Run planning for a fixed number of iterations.
     */
    Action run_iterations(
        const Belief& belief,
        const History& history,
        std::size_t max_iterations
    ) {
        Action last_action;

        for (std::size_t i = 0; i < max_iterations; ++i) {
            last_action = planner_.decide(belief, history);
        }

        return last_action;
    }

    /**
     * @brief Run planning for a fixed wall-clock time.
     */
    Action run_for_duration(
        const Belief& belief,
        const History& history,
        std::chrono::steady_clock::duration duration
    ) {
        const auto start = std::chrono::steady_clock::now();
        Action last_action;

        while (std::chrono::steady_clock::now() - start < duration) {
            last_action = planner_.decide(belief, history);
        }

        return last_action;
    }

    /**
     * @brief Anytime query.
     *
     * One call = one simulation.
     */
    Action step(
        const Belief& belief,
        const History& history
    ) {
        return planner_.decide(belief, history);
    }

private:
    Planner& planner_;
};

} // namespace pomdp

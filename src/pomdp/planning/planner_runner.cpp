#include <pomdp/planning/planner_runner.hpp>

namespace pomdp {

PlannerRunner::PlannerRunner(Planner& planner)
    : planner_(planner)
{}

Action PlannerRunner::run_iterations(
    const Belief& belief,
    const History& history,
    std::size_t max_iterations
)
{
    Action last_action;

    for (std::size_t i = 0; i < max_iterations; ++i) {
        last_action = planner_.decide(belief, history);
    }

    return last_action;
}

Action PlannerRunner::run_for_duration(
    const Belief& belief,
    const History& history,
    std::chrono::steady_clock::duration duration
)
{
    const auto start = std::chrono::steady_clock::now();
    Action last_action;

    while (std::chrono::steady_clock::now() - start < duration) {
        last_action = planner_.decide(belief, history);
    }

    return last_action;
}

Action PlannerRunner::step(
    const Belief& belief,
    const History& history
)
{
    return planner_.decide(belief, history);
}

} // namespace pomdp

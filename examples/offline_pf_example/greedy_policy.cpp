#include "greedy_policy.hpp"
#include "types.hpp"

namespace example {
GreedyPolicy::GreedyPolicy(const DiscreteModel& model)
    : model_(model) {}
    
std::unique_ptr<pomdp::Action>
GreedyPolicy::decide(
    const pomdp::Belief& belief,
    const pomdp::History&
) const {

    const auto& b = static_cast<const ParticleBelief&>(belief);

    double best_value = -1e9;
    int best_action = 0;

    for (int a = 0; a < model_.num_actions(); ++a) {
        double v = 0.0;
        for (const auto& p : b.particles) {
            v += p.weight * model_.reward(p.x, a);
        }
        if (v > best_value) {
            best_value = v;
            best_action = a;
        }
    }

    return std::make_unique<DiscreteAction>(best_action);
}

}

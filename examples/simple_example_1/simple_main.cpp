#include <iostream>
#include <pomdp/updater.hpp>
#include <pomdp/policy.hpp>
#include "simple_types.hpp"
#include "simple_updater.hpp"
#include "simple_policy.hpp"

int main() {
    example::SimpleUpdater updater;
    example::ThresholdPolicy policy;

    std::unique_ptr<pomdp::Belief> belief =
    std::make_unique<example::IntBelief>(0);
    
    example::EmptyHistory history;

    // offline / online: doesn't matter
    int observations[] = {1, -2, 3};

    for (int o : observations) {
        example::IntObservation obs(o);


        belief = updater.update(*belief, example::IntAction(0), obs, history);

        auto action = policy.decide(*belief, history);

        auto& a = static_cast<const example::IntAction&>(*action);

        
        auto& b = static_cast<const example::IntBelief&>(*belief);

        std::cout << "belief=" << b.value
                  << ", action=" << a.value << "\n";
    }
}

#pragma once

#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>

namespace pomdp {

template <typename StateT>
struct SimulationResult {
    StateT next_state;
    Observation observation;
    double reward;
};

template <typename StateT>
class Simulator {
public:
    virtual ~Simulator() = default;

    virtual SimulationResult<StateT> step(
        const StateT& state,
        const Action& action
    ) const = 0;
};

} // namespace pomdp

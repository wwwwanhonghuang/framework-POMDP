#include "model.hpp"
#include <cmath>

namespace example {

double DiscreteModel::transition_log_prob(
    const int& next,
    const int& prev,
    const pomdp::Action& action
) const {
    int a = action.id;
    int expected = (prev + a) % 10;
    return (next == expected) ? 0.0 : -std::log(10.0);
}

double DiscreteModel::observation_log_prob(
    const pomdp::Observation& obs,
    const int& state
) const {
    return (obs.id == state) ? 0.0 : -std::log(10.0);
}

int DiscreteModel::num_actions() const {
    return 3;
}

} // namespace example

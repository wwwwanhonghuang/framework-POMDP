#pragma once
#include <vector>
#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>

namespace example {

using State = int;

struct Particle {
    State x;
    double weight;
};

// ---- Concrete Action ----
class DiscreteAction : public pomdp::Action {
public:
    int id;
    explicit DiscreteAction(int id_) : id(id_) {}
};

// ---- Concrete Observation ----
class DiscreteObservation : public pomdp::Observation {
public:
    int id;
    explicit DiscreteObservation(int id_) : id(id_) {}
};

}

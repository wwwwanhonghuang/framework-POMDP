#pragma once
#include <memory>
#include "pomdp/belief.hpp"
#include "pomdp/action.hpp"
#include "pomdp/observation.hpp"
#include "pomdp/history.hpp"

namespace example {

// ---- Belief ----
class IntBelief : public pomdp::Belief {
public:
    int value;
    explicit IntBelief(int v = 0) : value(v) {}
};

// ---- Observation ----
class IntObservation : public pomdp::Observation {
public:
    int value;
    explicit IntObservation(int v) : value(v) {}
};

// ---- Action ----
class IntAction : public pomdp::Action {
public:
    int value;
    explicit IntAction(int v) : value(v) {}
};

// ---- History (empty for now) ----
class EmptyHistory : public pomdp::History {};

}

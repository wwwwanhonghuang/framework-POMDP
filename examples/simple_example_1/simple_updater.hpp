#pragma once
#include "pomdp/updater.hpp"
#include "simple_types.hpp"

namespace example {

class SimpleUpdater : public pomdp::BeliefUpdater {
public:
    std::unique_ptr<pomdp::Belief> update(
        const pomdp::Belief& prev,
        const pomdp::Action&,
        const pomdp::Observation& obs,
        const pomdp::History&
    ) const override {

        const auto& b = static_cast<const IntBelief&>(prev);
        const auto& o = static_cast<const IntObservation&>(obs);
        return std::make_unique<IntBelief>(b.value + o.value);
    }
};

}

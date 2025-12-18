#pragma once
#include "pomdp/policy.hpp"
#include "simple_types.hpp"

namespace example {

class ThresholdPolicy : public pomdp::Policy {
public:
    std::unique_ptr<pomdp::Action> decide(
        const pomdp::Belief& belief,
        const pomdp::History&
    ) const override {

        const auto& b = static_cast<const IntBelief&>(belief);
        return std::make_unique<IntAction>(b.value > 0 ? 1 : -1);
    }
};

}

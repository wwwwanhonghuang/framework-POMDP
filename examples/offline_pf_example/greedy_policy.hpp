#pragma once
#include <memory>
#include <pomdp/policy.hpp>
#include <pomdp/belief/pf_belief.hpp>
#include "model.hpp"

namespace example {

class GreedyPolicy : public pomdp::Policy {
public:
    explicit GreedyPolicy(const DiscreteModel& model);

    std::unique_ptr<pomdp::Action> decide(
        const pomdp::Belief& belief,
        const pomdp::History&
    ) const override;

private:
    const DiscreteModel& model_;
};

}

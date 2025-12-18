
#pragma once
#include<pomdp/belief.hpp>
#include<pomdp/history.hpp>
#include<pomdp/action.hpp>

namespace pomdp {

class Policy {
public:
    virtual std::unique_ptr<Action> decide(
        const Belief& belief,
        const History& history
    ) const = 0;
};

}
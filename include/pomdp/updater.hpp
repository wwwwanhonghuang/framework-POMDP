#pragma once
#include <memory>

#include <pomdp/belief.hpp>
#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>
#include <pomdp/history.hpp>

namespace pomdp {


class BeliefUpdater {
public:
    virtual std::unique_ptr<Belief> update(
        const Belief& prev,
        const Action& last_action,
        const Observation& obs,
        const History& history
    ) const = 0;
};

}


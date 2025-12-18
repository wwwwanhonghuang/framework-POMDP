#pragma once

#include <pomdp/dynamics/transition_kernel.hpp>
#include <pomdp/dynamics/observation_kernel.hpp>

namespace pomdp {

template <typename StateT>
class POMDPModel {
public:
    virtual ~POMDPModel() = default;

    virtual const TransitionKernel<StateT>& transition() const = 0;
    virtual const ObservationKernel<StateT>& observation() const = 0;
};

} // namespace pomdp

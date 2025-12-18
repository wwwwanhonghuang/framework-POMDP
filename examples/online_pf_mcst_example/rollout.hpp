#pragma once

#include <pomdp/planning/mcst/random_rollout.hpp>

#include "action_sampler.hpp"
#include "model.hpp"

namespace online_example {

/**
 * @brief Rollout policy used in the online PF + MCST example.
 *
 * This is a simple random rollout over continuous actions.
 *
 * Notes:
 *  - Stateless
 *  - Continuous-action compatible
 *  - Suitable as a baseline for online planning
 */
using Rollout =
    pomdp::mcst::RandomRollout<State>;

} // namespace online_example

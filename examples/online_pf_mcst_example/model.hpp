#pragma once

#include <Eigen/Dense>

#include <pomdp/pomdp_model.hpp>
#include <pomdp/dynamics/transition_kernel.hpp>
#include <pomdp/dynamics/observation_kernel.hpp>
#include <pomdp/planning/simulator.hpp>

namespace online_example {

// Forward declaration (avoid header coupling)
class ContinuousActionSampler;

/**
 * @brief Continuous state type.
 *
 * Example: 2D position + velocity.
 */
using State = Eigen::VectorXd;

/**
 * @brief Continuous POMDP model with generative interface.
 *
 * This model serves THREE purposes:
 *  1) Defines the probabilistic POMDP (for belief update)
 *  2) Provides transition & observation likelihoods
 *  3) Acts as a generative simulator for planning (MCST / POMCP)
 *
 * Action semantics are resolved via an external action sampler.
 */
class ContinuousModel
    : public pomdp::POMDPModel<State>
    , public pomdp::TransitionKernel<State>
    , public pomdp::ObservationKernel<State>
    , public pomdp::Simulator<State>
{
public:
    explicit ContinuousModel(
        const ContinuousActionSampler& action_sampler,
        double dt = 0.1
    );

    // ------------------------------------------------------------------
    // POMDPModel interface
    // ------------------------------------------------------------------
    const pomdp::TransitionKernel<State>& transition() const override {
        return *this;
    }

    const pomdp::ObservationKernel<State>& observation() const override {
        return *this;
    }

    // ------------------------------------------------------------------
    // TransitionKernel interface
    // ------------------------------------------------------------------
    double transition_log_prob(
        const State& next,
        const State& prev,
        const pomdp::Action& action
    ) const override;

    // ------------------------------------------------------------------
    // ObservationKernel interface
    // ------------------------------------------------------------------
    double observation_log_prob(
        const pomdp::Observation& obs,
        const State& state
    ) const override;

    std::int64_t position_to_obs_id(
        const State& state
    ) const;

    // ------------------------------------------------------------------
    // Simulator interface (generative model)
    // ------------------------------------------------------------------
    pomdp::SimulationResult<State> step(
        const State& state,
        const pomdp::Action& action
    ) const override;

private:
    const ContinuousActionSampler& action_sampler_;
    double dt_;
};

} // namespace online_example

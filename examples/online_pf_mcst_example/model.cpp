#include "model.hpp"
#include "action_sampler.hpp"
#include <random>
#include <cmath>

namespace online_example {

namespace {

// ------------------------------------------------------------------
// Noise parameters (example values)
// ------------------------------------------------------------------
constexpr double TRANSITION_STD = 0.05;
constexpr double OBSERVATION_STD = 0.1;

// Random engine (used only for generative simulation)
inline std::mt19937& rng() {
    static thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

inline double normal_sample(double stddev) {
    std::normal_distribution<double> dist(0.0, stddev);
    return dist(rng());
}

inline double normal_log_prob(double x, double stddev) {
    static const double LOG_SQRT_2PI = 0.5 * std::log(2.0 * M_PI);
    return -LOG_SQRT_2PI - std::log(stddev)
           - 0.5 * (x * x) / (stddev * stddev);
}

} // anonymous namespace

// ------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------
ContinuousModel::ContinuousModel(const ContinuousActionSampler& action_sampler, double dt)
    : action_sampler_(action_sampler), dt_(dt)
{}


// ------------------------------------------------------------------
// Transition kernel (log-probability)
// ------------------------------------------------------------------
double ContinuousModel::transition_log_prob(
    const State& next,
    const State& prev,
    const pomdp::Action& action
) const
{
    // Expect state = [px, py, vx, vy]
    // Action encodes acceleration [ax, ay]
    const Eigen::Vector2d& accel =
    action_sampler_.action_value(action);

    Eigen::VectorXd mean(4);
    mean[0] = prev[0] + prev[2] * dt_;
    mean[1] = prev[1] + prev[3] * dt_;
    mean[2] = prev[2] + accel[0] * dt_;
    mean[3] = prev[3] + accel[1] * dt_;

    double logp = 0.0;
    for (int i = 0; i < 4; ++i) {
        logp += normal_log_prob(next[i] - mean[i], TRANSITION_STD);
    }

    return logp;
}

// ------------------------------------------------------------------
// Observation kernel (log-probability)
// ------------------------------------------------------------------
double ContinuousModel::observation_log_prob(
    const pomdp::Observation& obs,
    const State& state
) const
{
    std::int64_t expected_id = position_to_obs_id(state);
    return (obs.id == expected_id) ? 0.0 : -std::numeric_limits<double>::infinity();
}

std::int64_t ContinuousModel::position_to_obs_id(
    const State& s
) const
{
    constexpr double CELL = 0.5;
    int gx = static_cast<int>(std::floor(s[0] / CELL));
    int gy = static_cast<int>(std::floor(s[1] / CELL));
    return (static_cast<std::int64_t>(gx) << 32) | (gy & 0xffffffff);
}


// ------------------------------------------------------------------
// Generative simulator (used by planners)
// ------------------------------------------------------------------
pomdp::SimulationResult<State> ContinuousModel::step(
    const State& state,
    const pomdp::Action& action
) const
{
    const Eigen::Vector2d& accel =
    action_sampler_.action_value(action);

    State next(4);
    next[0] = state[0] + state[2] * dt_ + normal_sample(TRANSITION_STD);
    next[1] = state[1] + state[3] * dt_ + normal_sample(TRANSITION_STD);
    next[2] = state[2] + accel[0] * dt_ + normal_sample(TRANSITION_STD);
    next[3] = state[3] + accel[1] * dt_ + normal_sample(TRANSITION_STD);

    // Observation: noisy position
    Eigen::Vector2d z;
    z[0] = next[0] + normal_sample(OBSERVATION_STD);
    z[1] = next[1] + normal_sample(OBSERVATION_STD);


    std::int64_t obs_id = position_to_obs_id(next);

    pomdp::Observation obs(obs_id);

    double reward = -std::hypot(next[0], next[1]); // TODO: noise

    // pomdp::Observation obs(z);

    // // Simple reward: negative distance from origin
    // const double reward =
    //     -std::sqrt(next[0] * next[0] + next[1] * next[1]);

    return pomdp::SimulationResult<State>{
        next,
        obs,
        reward
    };
}

} // namespace online_example

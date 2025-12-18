#pragma once

#include <random>
#include <unordered_map>

#include <Eigen/Dense>

#include <pomdp/action.hpp>
#include <pomdp/belief.hpp>
#include <pomdp/planning/action_sampler.hpp>

namespace online_example {

/**
 * @brief Continuous action sampler with registry.
 *
 * pomdp::Action is an opaque handle (int64_t).
 * The actual continuous control vector is stored here.
 */
class ContinuousActionSampler : public pomdp::ActionSampler {
public:
    explicit ContinuousActionSampler(double a_max = 1.0)
        : a_max_(a_max)
    {}

    pomdp::Action sample_random_action() const {
        // Reuse the same logic as sample_action,
        // but without consulting belief.
        Eigen::Vector2d a;
        a[0] = uniform_(-a_max_, a_max_);
        a[1] = uniform_(-a_max_, a_max_);

        const std::int64_t id = next_id_++;
        action_table_[id] = a;

        return pomdp::Action{id};
    }


    pomdp::Action sample_action(
        const pomdp::Belief& /*belief*/
    ) const override
    {
        Eigen::Vector2d a;
        a[0] = uniform_(-a_max_, a_max_);
        a[1] = uniform_(-a_max_, a_max_);

        const std::int64_t id = next_id_++;
        action_table_[id] = a;

        return pomdp::Action{id};
    }

    /**
     * @brief Retrieve continuous control vector from Action.
     */
    const Eigen::Vector2d& action_value(
        const pomdp::Action& action
    ) const
    {
        return action_table_.at(action.id);
    }

private:
    double a_max_;

    mutable std::int64_t next_id_ = 1;
    mutable std::unordered_map<std::int64_t, Eigen::Vector2d> action_table_;

    static double uniform_(double lo, double hi) {
        static thread_local std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(gen);
    }
};

} // namespace online_example

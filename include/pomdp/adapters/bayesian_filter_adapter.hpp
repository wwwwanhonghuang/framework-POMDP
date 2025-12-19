#pragma once

#include <memory>

#include <bayesian_filter/core/state_distribution.hpp>
#include <bayesian_filter/core/bayesian_filter.hpp>

#include <pomdp/belief.hpp>
#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>

namespace pomdp {

/**
 * @brief Adapter between POMDP belief and BayesianFilter.
 *
 * This class performs ONLY translation:
 *  - POMDP belief -> inference state
 *  - inference state -> POMDP belief
 *
 * It does NOT define what a belief is.
 * It does NOT implement inference algorithms.
 */
template <typename StateT, typename ActionT, typename ObservationT>
class BayesianFilterAdapter {
public:
    using FilterT =
        bayesian_filter::BayesianFilter<StateT, ActionT, ObservationT>;

    explicit BayesianFilterAdapter(const FilterT& filter)
        : filter_(filter) {}

    std::unique_ptr<Belief> update(
        const Belief& prev,
        const ActionT& action,
        const ObservationT& observation
    ) const
    {
        auto prev_dist = belief_to_distribution(prev);

        auto next_dist =
            filter_.step(*prev_dist, action, observation);

        return distribution_to_belief(*next_dist);
    }

protected:
    virtual std::unique_ptr<bayesian_filter::StateDistribution>
    belief_to_distribution(const Belief& belief) const = 0;

    virtual std::unique_ptr<Belief>
    distribution_to_belief(
        const bayesian_filter::StateDistribution& dist
    ) const = 0;

private:
    const FilterT& filter_;
};


} // namespace pomdp

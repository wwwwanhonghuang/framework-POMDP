#pragma once

#include <cmath>

namespace bayesian_filter {

template <typename StateT>
void ParticleBelief<StateT>::normalize()
{
    double sum = 0.0;
    for (const auto& p : particles) {
        sum += p.weight;
    }
    if (sum <= 0.0) {
        return;
    }

    for (auto& p : particles) {
        p.weight /= sum;
    }
}

template <typename StateT>
double ParticleBelief<StateT>::effective_sample_size() const
{
    double sq_sum = 0.0;
    for (const auto& p : particles) {
        sq_sum += p.weight * p.weight;
    }
    return (sq_sum > 0.0) ? 1.0 / sq_sum : 0.0;
}

} // namespace pomdp

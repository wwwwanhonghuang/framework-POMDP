#pragma once

#include <vector>
#include <memory>

#include <bayesian_filter/core/state_distribution.hpp>

namespace bayesian_filter {

/**
 * @brief Particle-based state distribution.
 */
template <typename StateT>
class ParticleBelief : public StateDistribution {
public:
    struct Particle {
        StateT state;
        double weight;
    };

    std::vector<Particle> particles;

    void normalize();
    double effective_sample_size() const;

    std::unique_ptr<StateDistribution> clone() const override {
        return std::make_unique<ParticleBelief<StateT>>(*this);
    }
};

} // namespace bayesian_filter

#pragma once

#include <Eigen/Dense>
#include <pomdp/belief.hpp>

namespace pomdp {

/**
 * Single Gaussian belief:
 *   p(x) = N(mean, cov)
 *
 * Pure representation only.
 */
template <typename StateT>
class GaussianBelief : public Belief {
public:
    StateT mean;
    Eigen::MatrixXd covariance;

    GaussianBelief() = default;

    GaussianBelief(const StateT& m, const Eigen::MatrixXd& cov)
        : mean(m), covariance(cov) {}
};

} // namespace pomdp

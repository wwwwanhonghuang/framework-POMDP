#pragma once

#include <vector>
#include <Eigen/Dense>
#include <pomdp/belief.hpp>

namespace pomdp {

/**
 * Gaussian Mixture Model belief.
 *
 * p(x) = sum_k w_k N(mean_k, cov_k)
 */
template <typename StateT>
class GaussianMixtureBelief : public Belief {
public:
    struct Component {
        StateT mean;
        Eigen::MatrixXd covariance;
        double weight;
    };

    std::vector<Component> components;
};

} // namespace pomdp

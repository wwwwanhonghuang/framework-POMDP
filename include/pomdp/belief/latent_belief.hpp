#pragma once

#include <Eigen/Dense>
#include <pomdp/belief.hpp>

namespace pomdp {

/**
 * Latent belief representation.
 *
 * z is a learned embedding of belief.
 */
class LatentBelief : public Belief {
public:
    Eigen::VectorXd z;
};

} // namespace pomdp

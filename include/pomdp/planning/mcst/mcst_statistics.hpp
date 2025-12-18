#pragma once

#include <cstddef>

namespace pomdp::mcst {

struct Statistics {
    std::size_t visits = 0;
    double value_sum = 0.0;

    void update(double value) {
        ++visits;
        value_sum += value;
    }

    double mean() const {
        return visits > 0 ? value_sum / visits : 0.0;
    }
};

} // namespace pomdp::mcst

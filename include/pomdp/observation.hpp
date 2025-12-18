#pragma once
#include <cstdint>

namespace pomdp {

struct Observation {
    std::int64_t id = 0;
    constexpr Observation() = default;
    constexpr explicit Observation(std::int64_t id_) : id(id_) {}
};

}

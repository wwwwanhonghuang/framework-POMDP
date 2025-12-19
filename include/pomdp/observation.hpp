#pragma once
#include <cstdint>
#include <functional> 
#include <cstddef>

namespace pomdp {

struct Observation {
    std::int64_t id = 0;
    constexpr Observation() = default;
    constexpr explicit Observation(std::int64_t id_) : id(id_) {}
    bool operator==(const Observation& other) const noexcept {
        return id == other.id;
    }
};

}


namespace std {

template<>
struct hash<pomdp::Observation> {
    std::size_t operator()(const pomdp::Observation& o) const noexcept {
        return std::hash<int>{}(o.id);
    }
};

} // namespace std
#pragma once
#include <cstdint>

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
struct hash<pomdp::Action> {
    std::size_t operator()(const pomdp::Action& a) const noexcept {
        return std::hash<int>{}(a.id);
    }
};

template<>
struct hash<pomdp::Observation> {
    std::size_t operator()(const pomdp::Observation& o) const noexcept {
        return std::hash<int>{}(o.id);
    }
};

} // namespace std

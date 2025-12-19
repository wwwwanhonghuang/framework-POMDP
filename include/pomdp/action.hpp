#pragma once
#include <cstdint>
#include <functional> 
#include <cstddef>

namespace pomdp {

struct Action {
    std::int64_t id = 0;
    constexpr Action() = default;
    constexpr explicit Action(std::int64_t id_) : id(id_) {}

    bool operator==(const Action& other) const noexcept {
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

} // namespace std
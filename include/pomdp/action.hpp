#pragma once
#include <cstdint>

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

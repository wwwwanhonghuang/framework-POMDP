

#pragma once
#include <vector>
#include <pomdp/history.hpp>
#include <pomdp/action.hpp>
#include <pomdp/observation.hpp>
namespace pomdp {

class SequenceHistory : public History {
public:
    struct Entry {
        Action action;
        Observation observation;
    };

    void append(const Action& a, const Observation& o) const;

    const std::vector<Entry>& entries() const { return entries_; }

private:
    mutable std::vector<Entry> entries_;
};

} // namespace pomdp

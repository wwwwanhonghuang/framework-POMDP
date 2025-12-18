#include <pomdp/history/sequence_history.hpp>

namespace pomdp {

void SequenceHistory::append(
    const Action& action,
    const Observation& observation
) const {
    entries_.push_back({action, observation});
}

} // namespace pomdp

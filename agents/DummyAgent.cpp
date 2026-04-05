#include "../env/State.h"
#include "../env/Action.h"

class DummyAgent {
public:
    Action act(const State& state) {
        // Placeholder for initial action logic
        return Action(ActionType::OFFER, 100);
    }
};

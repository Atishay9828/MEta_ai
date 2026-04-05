#pragma once
#include "../env/State.h"
#include "../env/Action.h"

class OpponentStrategy {
public:
    virtual ~OpponentStrategy() = default;
    
    virtual Action getResponse(const State& state, const Action& agent_action) = 0;
};

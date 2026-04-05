#pragma once
#include "State.h"
#include "Action.h"
#include <tuple>
#include <string>

class NegotiationEnv {
private:
    State state;
    
    // Hidden (internal)
    int agent_value;
    int opponent_value;
    std::string opponent_type; // "greedy", "fair", "impatient"

public:
    NegotiationEnv();
    
    void reset();
    std::tuple<State, double, bool> step(Action action);
    State getState() const;
};

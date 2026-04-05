#pragma once
#include "State.h"
#include "Action.h"
#include "../opponent/OpponentStrategy.h"
#include <tuple>
#include <string>
#include <memory>

class NegotiationEnv {
private:
    State state;
    int agent_value;
    int opponent_value;
    std::string opponent_type;
    
    std::unique_ptr<OpponentStrategy> opponent_strategy;

    double compute_reward(int deal_price);

public:
    NegotiationEnv();
    
    void reset();
    std::tuple<State, double, bool> step(Action action);
    State getState() const;
};

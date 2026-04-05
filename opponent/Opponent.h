#pragma once
#include <string>
#include <random>
#include "../env/State.h"
#include "../env/Action.h"
#include "OpponentStrategy.h"

class Opponent : public OpponentStrategy {
private:
    std::string type;
    int opponent_value;
    std::string opponent_role;
    double r;
    double alpha;
    int patience;
    int epsilon;
    
    double concession_rate;
    
    std::mt19937 rng;

public:
    Opponent(const std::string& type, int value, const std::string& role);
    Action getResponse(const State& state, const Action& agent_action) override;
};

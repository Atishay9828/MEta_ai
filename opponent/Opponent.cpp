#include "Opponent.h"
#include <stdexcept>
#include <ctime>

Opponent::Opponent(const std::string& type, int value, const std::string& role) 
    : type(type), opponent_value(value), opponent_role(role) {
    if (type == "greedy") {
        r = 0.05; alpha = 0.7; patience = 10; epsilon = 5;
    } else if (type == "fair") {
        r = 0.15; alpha = 0.4; patience = 7; epsilon = 10;
    } else if (type == "impatient") {
        r = 0.25; alpha = 0.2; patience = 3; epsilon = 15;
    } else {
        throw std::invalid_argument("Unknown opponent type");
    }
    concession_rate = r;
    rng.seed(static_cast<unsigned int>(std::time(nullptr)));
}

Action Opponent::getResponse(const State& state, const Action& agent_action) {
    if (agent_action.getType() != ActionType::OFFER) {
        return Action(ActionType::REJECT);
    }

    int agent_offer = agent_action.getPrice();

    // 1. Acceptance
    bool accept = false;
    if (opponent_role == "seller") {
        if (agent_offer >= opponent_value) accept = true;
    } else { // "buyer"
        if (agent_offer <= opponent_value) accept = true;
    }

    if (accept) {
        return Action(ActionType::ACCEPT);
    }

    // Patience Behavior (STRICT RULE)
    int current_round = state.getRound();
    if (current_round > patience) {
        concession_rate += 0.05;
        if (concession_rate > 0.4) concession_rate = 0.4;
    }

    // 3. Counter Offer
    double target = opponent_value;
    double current_offer = state.getCurrentOffer();
    
    double delta = target - current_offer;
    double next_offer = current_offer + concession_rate * delta;

    // Anchor Effect
    next_offer = (1.0 - alpha) * next_offer + alpha * current_offer;

    // Noise
    std::uniform_int_distribution<int> noise_dist(-epsilon, epsilon);
    next_offer += noise_dist(rng);

    // Clamp between [100, 1000]
    int final_offer = static_cast<int>(next_offer);
    if (final_offer < 100) final_offer = 100;
    if (final_offer > 1000) final_offer = 1000;

    return Action(ActionType::OFFER, final_offer);
}

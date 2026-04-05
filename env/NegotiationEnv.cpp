#include "NegotiationEnv.h"
#include "../opponent/Opponent.h"
#include <random>
#include <ctime>
#include <cmath>

NegotiationEnv::NegotiationEnv() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

void NegotiationEnv::reset() {
    state.setRound(0);
    state.setMaxRounds(20); 
    
    agent_value = 100 + (std::rand() % 901);
    opponent_value = 100 + (std::rand() % 901);
    
    if (std::rand() % 2 == 0) {
        state.setRole("buyer");
    } else {
        state.setRole("seller");
    }
    
    int t = std::rand() % 3;
    if (t == 0) opponent_type = "greedy";
    else if (t == 1) opponent_type = "fair";
    else opponent_type = "impatient";
    
    int initial_offer = (agent_value + opponent_value) / 2;
    state.setCurrentOffer(initial_offer);
    state.setLastOpponentAction("START");
    state.setLastOpponentOffer(0);
    
    std::string opp_role = (state.getRole() == "buyer") ? "seller" : "buyer";
    opponent_strategy = std::make_unique<Opponent>(opponent_type, opponent_value, opp_role);
}

double NegotiationEnv::compute_reward(int deal_price) {
    double profit = 0;
    if (state.getRole() == "seller") {
        profit = deal_price - agent_value;
    } else {
        profit = agent_value - deal_price;
    }
    
    double time_factor = 1.0 - (static_cast<double>(state.getRound()) / state.getMaxRounds());
    double reward = profit * time_factor;
    
    if (profit < 0) {
        reward -= 20;
    }
    return reward;
}

std::tuple<State, double, bool> NegotiationEnv::step(Action action) {
    // 1. Increment round
    int current_round = state.getRound() + 1;
    state.setRound(current_round);
    
    double reward = 0.0;
    bool done = false;

    if (action.getType() == ActionType::ACCEPT) {
        // 2. If ACCEPT
        int deal_price = state.getLastOpponentOffer();
        reward = compute_reward(deal_price);
        done = true;
    } else if (action.getType() == ActionType::REJECT) {
        // 3. If REJECT
        reward = -50;
        done = true;
    } else if (action.getType() == ActionType::OFFER) {
        // 4. If OFFER
        Action opp_response = opponent_strategy->getResponse(state, action);
        
        if (opp_response.getType() == ActionType::ACCEPT) {
            int deal_price = action.getPrice();
            reward = compute_reward(deal_price);
            done = true;
            
            // 5. Update state
            state.setLastOpponentAction("ACCEPT");
        } else {
            // Opponent generates counter-offer
            state.setCurrentOffer(opp_response.getPrice());
            // 5. Update state
            state.setLastOpponentAction("OFFER");
            state.setLastOpponentOffer(opp_response.getPrice());
        }
    }
    
    // 6. If round == max_rounds
    if (!done && current_round >= state.getMaxRounds()) {
        reward = -50;
        done = true;
    }

    // Apply Aggression Penalty if deal resolves this step
    if (done && action.getType() == ActionType::OFFER) {
        if (std::abs(action.getPrice() - opponent_value) > 150) {
            reward -= 2;
        }
    }
    
    // 7. If NOT done (IMPORTANT strict rule)
    if (!done) {
        reward = 0;
    }
    
    return std::make_tuple(state, reward, done);
}

State NegotiationEnv::getState() const {
    return state;
}

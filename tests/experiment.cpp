#define private public
#include "../env/NegotiationEnv.h"
#undef private
#include "../opponent/Opponent.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <memory>

void run_simulation(std::string test_name, std::string opp_type, std::string role, int a_val, int o_val, int behavior_type) {
    NegotiationEnv env;
    env.reset();
    
    // Force specific scenario
    env.opponent_type = opp_type;
    env.agent_value = a_val;
    env.opponent_value = o_val;
    env.state.setRole(role);
    env.state.setCurrentOffer((a_val + o_val) / 2);
    env.state.setLastOpponentOffer(0);
    
    std::string opp_role = (role == "buyer") ? "seller" : "buyer";
    env.opponent_strategy = std::make_unique<Opponent>(opp_type, o_val, opp_role);
    
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Opp Type: " << opp_type << " | Role: " << role << " | Ag_Val: " << a_val << " | Opp_Val: " << o_val << std::endl;
    
    bool done = false;
    double final_reward = 0;
    int round = 0;
    
    int action_price = 0;
    while (!done) {
        round = env.state.getRound() + 1;
        
        if (behavior_type == 1) { // Baseline: fixed offer

            action_price = (role == "buyer") ? 100 : 900;
        } else if (behavior_type == 2) { // Extreme: super bad
            action_price = (role == "buyer") ? 10 : 1500;
        } else if (behavior_type == 3) { // Gradual improvement
            if (role == "buyer") {
                if (round == 1) action_price = 100;
                else if (round == 2) action_price = a_val - 200; // e.g. 600
                else action_price = a_val - 50; // e.g. 750 (near acceptable for opp who wants 500)
            } else { // seller
                if (round == 1) action_price = 1000;
                else if (round == 2) action_price = a_val + 200;
                else action_price = a_val + 50;
            }
        }
        
        Action agent_action(ActionType::OFFER, action_price);
        std::cout << "[Round " << round << "] Agent OFFER " << action_price << " -> ";
        
        auto [next_state, reward, is_done] = env.step(agent_action);
        done = is_done;
        
        std::cout << "Opponent " << env.state.getLastOpponentAction();
        if (env.state.getLastOpponentAction() == "OFFER") {
            std::cout << " " << env.state.getLastOpponentOffer();
        }
        std::cout << " | Reward: " << reward << " | Done: " << (done ? "True" : "False") << std::endl;
        
        if (done) final_reward = reward;
        if (round >= env.state.getMaxRounds()) break; 
    }
    
    if (env.state.getLastOpponentAction() == "ACCEPT") {
        // Last op action is accept, means opponent accepted the agent's OFFER.
        // Wait, deal price is agent's offer.
        std::cout << "Final Deal Price: " << action_price << " | Final Reward: " << final_reward << std::endl;
    } else {
        std::cout << "Final Deal Price: NONE (" << env.state.getLastOpponentAction() << ") | Final Reward: " << final_reward << std::endl;
    }
}

int main() {
    std::cout << "--- EXPERIMENT LOGS ---" << std::endl;
    // Buyer wants to buy for as low as possible (Profit = target - price). So target = 800.
    // Opponent is Seller, wants price >= 500.
    
    // TEST SET 1: Baseline
    run_simulation("TEST SET 1A (Baseline vs Greedy)", "greedy", "buyer", 800, 500, 1);
    run_simulation("TEST SET 1B (Baseline vs Fair)", "fair", "buyer", 800, 500, 1);
    run_simulation("TEST SET 1C (Baseline vs Impatient)", "impatient", "buyer", 800, 500, 1);
    
    // TEST SET 2: Extreme Strategy
    run_simulation("TEST SET 2A (Extreme vs Fair)", "fair", "buyer", 800, 500, 2);
    run_simulation("TEST SET 2B (Extreme vs Impatient)", "impatient", "buyer", 800, 500, 2);
    
    // TEST SET 3: Gradual Improvement
    run_simulation("TEST SET 3A (Gradual vs Fair)", "fair", "buyer", 800, 500, 3);
    run_simulation("TEST SET 3B (Gradual vs Impatient)", "impatient", "buyer", 800, 500, 3);
    
    // TEST SET 4: Edge Cases
    // Approx Equal (very tight margins)
    run_simulation("TEST SET 4A (Edge - Approx Equal)", "fair", "buyer", 510, 500, 3);
    // Large gap (very wide margins, easy deal)
    run_simulation("TEST SET 4B (Edge - Large Gap)", "fair", "buyer", 900, 200, 3);

    return 0;
}

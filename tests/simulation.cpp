#include <iostream>
#include "../env/NegotiationEnv.h"
#include "../agents/DummyAgent.cpp"

int main() {
    std::cout << "Starting Phase 2 Simulation Trace..." << std::endl;
    NegotiationEnv env;
    DummyAgent agent;
    
    env.reset();
    State state = env.getState();
    std::cout << "Agent Role: " << state.getRole() << std::endl;
    std::cout << "Initial Offer state: " << state.getCurrentOffer() << std::endl;
    
    bool done = false;
    double total_reward = 0;
    
    while (!done) {
        // The dummy agent just offers 100 statically in this demo
        Action a = agent.act(state);
        std::cout << "\n[Round " << state.getRound() + 1 << "]" << std::endl;
        std::cout << "  Agent Action: OFFER " << a.getPrice() << std::endl;
        
        auto [next_state, reward, is_done] = env.step(a);
        state = next_state;
        done = is_done;
        total_reward += reward;
        
        std::cout << "  Opponent Action: " << state.getLastOpponentAction();
        if (state.getLastOpponentAction() == "OFFER") {
            std::cout << " " << state.getLastOpponentOffer();
        }
        std::cout << "\n  Step Reward: " << reward << ", Done: " << (done ? "true" : "false") << std::endl;
    }
    
    std::cout << "\nSimulation Ended. Final Aggregate Reward Processed: " << total_reward << std::endl;
    return 0;
}

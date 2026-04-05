#pragma once
#include <string>

class State {
private:
    int current_offer;
    int round;
    int max_rounds;
    std::string role; // "buyer" or "seller"
    std::string last_opponent_action;
    int last_opponent_offer;

public:
    State() : current_offer(0), round(0), max_rounds(0), last_opponent_offer(0) {}

    // Getters and Setters can be added as needed
    int getCurrentOffer() const { return current_offer; }
    void setCurrentOffer(int offer) { current_offer = offer; }

    int getRound() const { return round; }
    void setRound(int r) { round = r; }

    int getMaxRounds() const { return max_rounds; }
    void setMaxRounds(int max) { max_rounds = max; }

    std::string getRole() const { return role; }
    void setRole(const std::string& r) { role = r; }

    std::string getLastOpponentAction() const { return last_opponent_action; }
    void setLastOpponentAction(const std::string& a) { last_opponent_action = a; }

    int getLastOpponentOffer() const { return last_opponent_offer; }
    void setLastOpponentOffer(int o) { last_opponent_offer = o; }
};

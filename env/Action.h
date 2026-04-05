#pragma once

enum class ActionType {
    OFFER,
    ACCEPT,
    REJECT
};

class Action {
private:
    ActionType type;
    int price; // Only relevant if type == ActionType::OFFER

public:
    Action() : type(ActionType::REJECT), price(0) {}
    Action(ActionType t, int p = 0) : type(t), price(p) {}

    ActionType getType() const { return type; }
    int getPrice() const { return price; }
};

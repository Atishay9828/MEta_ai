import random

class Opponent:
    def __init__(self, type_str, value, role):
        self.type = type_str
        self.opponent_value = value
        self.opponent_role = role
        if type_str == "greedy":
            self.r, self.alpha, self.patience, self.epsilon = 0.05, 0.7, 10, 5
        elif type_str == "fair":
            self.r, self.alpha, self.patience, self.epsilon = 0.15, 0.4, 7, 10
        elif type_str == "impatient":
            self.r, self.alpha, self.patience, self.epsilon = 0.25, 0.2, 3, 15
        else:
            self.r, self.alpha, self.patience, self.epsilon = 0.15, 0.4, 7, 10
        self.concession_rate = self.r

    def get_response(self, round_num, current_offer, agent_offer, agent_action_type):
        if agent_action_type != "OFFER":
            return "REJECT", 0

        if self.opponent_role == "seller" and agent_offer >= self.opponent_value: 
            return "ACCEPT", agent_offer
        if self.opponent_role == "buyer" and agent_offer <= self.opponent_value: 
            return "ACCEPT", agent_offer

        if round_num > self.patience:
            self.concession_rate = min(0.4, self.concession_rate + 0.05)

        target = self.opponent_value
        delta = target - current_offer
        next_offer = current_offer + self.concession_rate * delta
        next_offer = (1.0 - self.alpha) * next_offer + self.alpha * current_offer
        next_offer += random.randint(-self.epsilon, self.epsilon)
        next_offer = max(100, min(1000, int(next_offer)))
        return "OFFER", next_offer

class EnvWrapper:
    def __init__(self, opp_type="fair", a_val=800, o_val=500, agent_role="buyer"):
        self.agent_value = a_val
        self.opponent_value = o_val
        self.role = agent_role
        self.opp_role = "seller" if agent_role == "buyer" else "buyer"
        self.opp = Opponent(opp_type, o_val, self.opp_role)
        self.max_rounds = 20
        self.reset()
        
    def reset(self):
        self.round = 0
        if self.role == "buyer":
            self.current_offer = self.agent_value + 200
        else:
            self.current_offer = max(100, self.agent_value - 200)
        self.last_opp_action = "START"
        self.last_opp_offer = self.current_offer

    def step(self, action_str, action_price=0):
        self.round += 1
        aggressive = False
        done = False
        reward = 0.0
        
        if action_str == "ACCEPT":
            deal_price = self.last_opp_offer
            done = True
            profit = deal_price - self.agent_value if self.role == "seller" else self.agent_value - deal_price
            t_factor = 1.0 - (self.round / self.max_rounds)
            reward = profit * t_factor
            if profit < 0: reward -= 20
            
        elif action_str == "REJECT":
            reward = -50.0
            done = True
            
        elif action_str.startswith("OFFER"):
            aggressive = abs(action_price - self.opponent_value) > 150
            opp_action, opp_price = self.opp.get_response(self.round, self.current_offer, action_price, "OFFER")
            if opp_action == "ACCEPT":
                deal_price = action_price
                done = True
                self.last_opp_action = "ACCEPT"
                self.last_opp_offer = deal_price
                
                profit = deal_price - self.agent_value if self.role == "seller" else self.agent_value - deal_price
                t_factor = 1.0 - (self.round / self.max_rounds)
                reward = profit * t_factor
                if profit < 0: reward -= 20
                if aggressive: reward -= 2
            else:
                self.current_offer = opp_price
                self.last_opp_action = "OFFER"
                self.last_opp_offer = opp_price
                if self.round >= self.max_rounds:
                    reward = -50.0
                    done = True
        
        if not done:
            reward = 0.0
            
        return reward, done

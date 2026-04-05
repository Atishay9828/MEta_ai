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
        self.concession_rate = self.r

    def get_response(self, round_num, current_offer, agent_offer):
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

class Env:
    def __init__(self, opp_type, a_val, o_val, role):
        self.agent_value = a_val
        self.opponent_value = o_val
        self.role = role
        self.opp_type = opp_type
        self.opp_role = "seller" if role == "buyer" else "buyer"
        self.opp = Opponent(opp_type, o_val, self.opp_role)
        self.current_offer = (a_val + o_val) // 2
        self.max_rounds = 20
        self.round = 0
        self.last_opp_action = "START"
        self.last_opp_offer = 0

    def step(self, action_price):
        self.round += 1
        aggressive = abs(action_price - self.opponent_value) > 150
        
        opp_action, opp_price = self.opp.get_response(self.round, self.current_offer, action_price)
        done = False
        reward = 0
        
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
                reward = -50
                done = True
                
        return reward, done

def run_sim(name, opp_type, role, a_val, o_val, b_type):
    print(f"\n=== {name} ===")
    print(f"Opponent Type: {opp_type} | Agent Role: {role} | Agent Value: {a_val} | Opp Value: {o_val}")
    env = Env(opp_type, a_val, o_val, role)
    done = False
    
    while not done and env.round <= 25:
        act_price = 0
        rnd = env.round + 1
        if b_type == 1:
            act_price = 100 if role == "buyer" else 900
        elif b_type == 2:
            act_price = 10 if role == "buyer" else 1500
        elif b_type == 3:
            if role == "buyer":
                act_price = 100 if rnd == 1 else (o_val - 100 if rnd == 2 else o_val)
            else:
                act_price = 1000 if rnd == 1 else (o_val + 100 if rnd == 2 else o_val)
                
        r, d = env.step(act_price)
        done = d
        opp_val_print = env.last_opp_offer if env.last_opp_action == "OFFER" else ""
        print(f"[Round {env.round}] Agent OFFER {act_price} -> Opponent {env.last_opp_action} {opp_val_print} | Step Reward: {r:.2f} | Done: {done}")
    
    if env.last_opp_action == "ACCEPT":
        print(f"Final Deal Price: {env.last_opp_offer} | Final Reward: {r:.2f}")
    else:
        print(f"Final Deal Price: NONE | Final Reward: {r:.2f}")

random.seed(42)
print("--- TEST LOGS ---")
run_sim("Test 1A Baseline Greedy", "greedy", "buyer", 800, 500, 1)
run_sim("Test 1B Baseline Fair", "fair", "buyer", 800, 500, 1)
run_sim("Test 1C Baseline Impatient", "impatient", "buyer", 800, 500, 1)
run_sim("Test 2A Extreme vs Fair", "fair", "buyer", 800, 500, 2)
run_sim("Test 3A Gradual vs Fair", "fair", "buyer", 800, 500, 3)
run_sim("Test 4A Edge Approx Equal", "fair", "buyer", 510, 500, 3)
run_sim("Test 4B Edge Large Gap", "fair", "buyer", 900, 200, 3)

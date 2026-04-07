"""
Negotiation Environment Wrapper — OpenEnv Compliant
Implements: reset(), step(), state()
Typed models via Pydantic for Observation, Action, Reward
"""

import random
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# OpenEnv Typed Models
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """Observable state visible to the agent."""
    agent_value: int = Field(description="The agent's private valuation/target value for the deal")
    current_offer: int = Field(description="Current price on the table")
    round: int = Field(description="Current round number (0-indexed before first step)")
    max_rounds: int = Field(description="Maximum allowed rounds")
    role: str = Field(description="Agent role: 'buyer' or 'seller'")
    last_opponent_action: str = Field(description="Opponent's last action: 'START', 'OFFER', 'ACCEPT'")
    last_opponent_offer: int = Field(description="Opponent's last offered price")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="History of all actions this episode")


class ActionModel(BaseModel):
    """Action the agent can take."""
    action_type: str = Field(description="One of: 'OFFER', 'ACCEPT', 'REJECT'")
    price: int = Field(default=0, description="Price for OFFER actions, ignored for ACCEPT/REJECT")


class RewardInfo(BaseModel):
    """Reward information returned by step()."""
    reward: float = Field(description="Numeric reward for this step")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Reward component breakdown")


# ─────────────────────────────────────────────
# Opponent Strategy
# ─────────────────────────────────────────────

class Opponent:
    """
    Simulates opponent negotiation behavior.
    Three personalities: greedy, fair, impatient.
    Each has different concession rates, anchor effects, patience, and noise.
    """

    PROFILES = {
        "greedy":    {"r": 0.05, "alpha": 0.7, "patience": 10, "epsilon": 5},
        "fair":      {"r": 0.15, "alpha": 0.4, "patience": 7,  "epsilon": 10},
        "impatient": {"r": 0.25, "alpha": 0.2, "patience": 3,  "epsilon": 15},
    }

    def __init__(self, type_str: str, value: int, role: str):
        self.type = type_str
        self.opponent_value = value
        self.opponent_role = role
        self.history: List[Dict[str, Any]] = []

        profile = self.PROFILES.get(type_str, self.PROFILES["fair"])
        self.r = profile["r"]
        self.alpha = profile["alpha"]
        self.patience = profile["patience"]
        self.epsilon = profile["epsilon"]
        self.concession_rate = self.r

    def reset_state(self):
        """Reset concession rate and history for new episode."""
        self.concession_rate = self.r
        self.history = []

    def get_response(self, round_num: int, current_offer: int, agent_offer: int, agent_action_type: str):
        """
        Generate opponent response to agent's action.
        Returns: (action_type: str, price: int)
        """
        if agent_action_type != "OFFER":
            return "REJECT", 0

        # ── Acceptance Check ──
        # Opponent negotiates for a minimum number of rounds before accepting.
        # Greedy opponents hold out longer; impatient ones settle sooner.
        min_round_to_accept = max(2, self.patience // 3)

        offer_acceptable = (
            (self.opponent_role == "seller" and agent_offer >= self.opponent_value) or
            (self.opponent_role == "buyer" and agent_offer <= self.opponent_value)
        )
        if offer_acceptable and round_num >= min_round_to_accept:
            self.history.append({"round": round_num, "action": "ACCEPT", "price": agent_offer})
            return "ACCEPT", agent_offer

        # ── Patience-based concession acceleration ──
        if round_num > self.patience:
            self.concession_rate = min(0.4, self.concession_rate + 0.05)

        # ── Counter-offer calculation ──
        target = self.opponent_value
        delta = target - current_offer
        next_offer = current_offer + self.concession_rate * delta

        # Anchor effect — blend toward current offer
        next_offer = (1.0 - self.alpha) * next_offer + self.alpha * current_offer

        # Add noise
        next_offer += random.randint(-self.epsilon, self.epsilon)

        # ── VALUE-BASED CLAMPING (Tolerance Bug Fix) ──
        # Seller must not offer below their own value
        # Buyer must not offer above their own value
        next_offer_int = int(next_offer)
        if self.opponent_role == "seller":
            next_offer_int = max(next_offer_int, self.opponent_value)
        elif self.opponent_role == "buyer":
            next_offer_int = min(next_offer_int, self.opponent_value)

        # Absolute bounds
        next_offer_int = max(100, min(1000, next_offer_int))

        self.history.append({"round": round_num, "action": "OFFER", "price": next_offer_int})
        return "OFFER", next_offer_int


# ─────────────────────────────────────────────
# Main Environment Wrapper
# ─────────────────────────────────────────────

class EnvWrapper:
    """
    OpenEnv-compliant negotiation environment.
    Exposes: reset(), step(), state()
    """

    def __init__(self, opp_type: str = "fair", a_val: int = 800, o_val: int = 500,
                 agent_role: str = "buyer", max_rounds: int = 20):
        self.agent_value = a_val
        self.opponent_value = o_val
        self.role = agent_role
        self.opp_type = opp_type
        self.opp_role = "seller" if agent_role == "buyer" else "buyer"
        self.max_rounds = max_rounds
        self.opp = Opponent(opp_type, o_val, self.opp_role)

        # Episode tracking
        self.round = 0
        self.current_offer = 0
        self.last_opp_action = "START"
        self.last_opp_offer = 0
        self.history: List[Dict[str, Any]] = []
        self.cumulative_aggression_penalty = 0.0
        self.done = False

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self.round = 0
        self.done = False
        self.history = []
        self.cumulative_aggression_penalty = 0.0
        self.opp.reset_state()

        # Initial offer is shifted away from agent's value to force negotiation
        if self.role == "buyer":
            # Start high — agent (buyer) must negotiate DOWN
            self.current_offer = min(1000, self.agent_value + 200)
        else:
            # Start low — agent (seller) must negotiate UP
            self.current_offer = max(100, self.agent_value - 200)

        self.last_opp_action = "START"
        self.last_opp_offer = self.current_offer

        return self.state()

    def state(self) -> Observation:
        """Return current observable state."""
        return Observation(
            agent_value=self.agent_value,
            current_offer=self.current_offer,
            round=self.round,
            max_rounds=self.max_rounds,
            role=self.role,
            last_opponent_action=self.last_opp_action,
            last_opponent_offer=self.last_opp_offer,
            history=list(self.history),
        )

    def _compute_reward(self, deal_price: int) -> tuple:
        """
        Compute reward for a completed deal.
        Returns: (total_reward, breakdown_dict)
        """
        if self.role == "seller":
            profit = deal_price - self.agent_value
        else:
            profit = self.agent_value - deal_price

        # Gentle time decay: linear, max 50% loss even if all rounds used.
        # This rewards fast deals but doesn't destroy multi-round negotiation.
        time_factor = 1.0 - 0.5 * (self.round / self.max_rounds)
        base_reward = profit * time_factor

        # Penalty for bad deals (agent accepts a losing deal)
        bad_deal_penalty = -20.0 if profit < 0 else 0.0

        # Cumulative aggression penalty
        aggression = -self.cumulative_aggression_penalty

        total = base_reward + bad_deal_penalty + aggression

        breakdown = {
            "profit": float(profit),
            "time_factor": round(time_factor, 4),
            "base_reward": round(base_reward, 4),
            "bad_deal_penalty": bad_deal_penalty,
            "aggression_penalty": aggression,
            "total": round(total, 4),
        }
        return total, breakdown

    def _partial_progress_reward(self, action_str: str, action_price: int) -> tuple:
        """
        Provide a small shaping reward for intermediate steps.
        Rewards the agent for moving toward a deal (improving offers).
        """
        reward = 0.0
        breakdown = {}

        if action_str.startswith("OFFER") and len(self.history) >= 2:
            # Check if agent is making progress toward opponent
            prev_agent_offers = [h["agent_price"] for h in self.history[:-1]
                                 if h.get("agent_action", "").startswith("OFFER")]
            if prev_agent_offers:
                last_agent_offer = prev_agent_offers[-1]
                # Positive signal if agent moves toward a reasonable range
                if self.role == "buyer":
                    # Buyer should increase offers (toward seller's value)
                    improvement = action_price - last_agent_offer
                    reward = min(2.0, max(-1.0, improvement / 50.0))
                else:
                    # Seller should decrease offers (toward buyer's value)
                    improvement = last_agent_offer - action_price
                    reward = min(2.0, max(-1.0, improvement / 50.0))

                breakdown = {"progress_signal": round(reward, 4)}

        return reward, breakdown

    def step(self, action_str: str, action_price: int = 0):
        """
        Take one step in the environment.

        Args:
            action_str: "OFFER", "ACCEPT", or "REJECT"
            action_price: price for OFFER actions

        Returns:
            (observation: Observation, reward: float, done: bool, info: dict)
        """
        if self.done:
            return self.state(), 0.0, True, {"error": "Episode already ended"}

        self.round += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"error": None}
        breakdown: Dict[str, float] = {}

        # ── AGENT OFFER CLAMPING ──
        if action_str.startswith("OFFER"):
            action_price = max(100, min(1000, action_price))
            action_str = f"OFFER {action_price}"

            # ── CUMULATIVE AGGRESSION PENALTY ──
            # Scale threshold to ZOPA width so narrow-ZOPA tasks aren't unfairly punished
            zopa = abs(self.agent_value - self.opponent_value)
            aggression_threshold = max(100, int(zopa * 1.25))
            if abs(action_price - self.opponent_value) > aggression_threshold:
                self.cumulative_aggression_penalty += 2.0

        # Record this step in history
        step_record = {
            "round": self.round,
            "agent_action": action_str,
            "agent_price": action_price,
        }

        if action_str == "ACCEPT":
            deal_price = self.last_opp_offer
            reward, breakdown = self._compute_reward(deal_price)
            done = True
            info["deal_price"] = deal_price
            info["deal_type"] = "agent_accepted"

        elif action_str == "REJECT":
            reward = -50.0
            breakdown = {"rejection_penalty": -50.0}
            done = True
            info["deal_type"] = "agent_rejected"

        elif action_str.startswith("OFFER"):
            opp_action, opp_price = self.opp.get_response(
                self.round, self.current_offer, action_price, "OFFER"
            )

            if opp_action == "ACCEPT":
                deal_price = action_price
                reward, breakdown = self._compute_reward(deal_price)
                done = True
                self.last_opp_action = "ACCEPT"
                self.last_opp_offer = deal_price
                info["deal_price"] = deal_price
                info["deal_type"] = "opponent_accepted"
            else:
                # Opponent counters
                self.current_offer = opp_price
                self.last_opp_action = "OFFER"
                self.last_opp_offer = opp_price

                # Check max rounds
                if self.round >= self.max_rounds:
                    reward = -50.0
                    breakdown = {"timeout_penalty": -50.0}
                    done = True
                    info["deal_type"] = "timeout"
                else:
                    # Partial progress reward for intermediate steps
                    step_record["agent_price"] = action_price
                    self.history.append(step_record)
                    reward, breakdown = self._partial_progress_reward(action_str, action_price)
                    info["opponent_counter"] = opp_price

            step_record["opp_action"] = opp_action
            step_record["opp_price"] = opp_price

        # Record history for terminal steps too
        if done or action_str == "ACCEPT" or action_str == "REJECT":
            # Avoid double-append for non-OFFER terminal steps
            if step_record not in self.history:
                self.history.append(step_record)

        self.done = done
        info["reward_breakdown"] = breakdown

        return self.state(), reward, done, info


# ─────────────────────────────────────────────
# Convenience — max possible reward for scoring
# ─────────────────────────────────────────────

def get_max_possible_reward(agent_value: int, opponent_value: int) -> float:
    """
    Maximum reward possible if agent gets the best possible deal on round 1.
    """
    return float(abs(agent_value - opponent_value))

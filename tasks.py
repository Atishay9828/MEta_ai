"""
Task Definitions & Graders for the Negotiation Environment.
3 tasks: Easy → Medium → Hard, each with a programmatic grader (0.0–1.0).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TaskConfig:
    """Configuration for a single evaluation task."""
    name: str
    description: str
    difficulty: str
    opp_type: str
    agent_value: int
    opponent_value: int
    agent_role: str
    max_rounds: int
    success_threshold: float  # score >= this means success


# ─────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────

TASK_A = TaskConfig(
    name="task_a_easy",
    description="Easy negotiation: fair opponent, wide ZOPA, plenty of rounds",
    difficulty="easy",
    opp_type="fair",
    agent_value=800,
    opponent_value=400,
    agent_role="buyer",
    max_rounds=20,
    success_threshold=0.2,
)

TASK_B = TaskConfig(
    name="task_b_medium",
    description="Medium negotiation: greedy opponent, narrow ZOPA, fewer rounds",
    difficulty="medium",
    opp_type="greedy",
    agent_value=700,
    opponent_value=500,
    agent_role="buyer",
    max_rounds=15,
    success_threshold=0.3,
)

TASK_C = TaskConfig(
    name="task_c_hard",
    description="Hard negotiation: impatient opponent, tight margins, very few rounds",
    difficulty="hard",
    opp_type="impatient",
    agent_value=600,
    opponent_value=480,
    agent_role="buyer",
    max_rounds=6,
    success_threshold=0.4,
)

ALL_TASKS: List[TaskConfig] = [TASK_A, TASK_B, TASK_C]


# ─────────────────────────────────────────────
# Grader
# ─────────────────────────────────────────────

class Grader:
    """
    Programmatic grader for a negotiation task.
    Scores agent performance on a 0.0–1.0 scale.
    """

    def __init__(self, task: TaskConfig):
        self.task = task
        self.max_possible = float(abs(task.agent_value - task.opponent_value))

    def grade(self, rewards: List[float], steps: int, deal_made: bool) -> dict:
        """
        Grade an episode.

        Args:
            rewards: list of per-step rewards
            steps: number of steps taken
            deal_made: whether a deal was successfully completed

        Returns:
            dict with score, success, and breakdown
        """
        total_reward = sum(rewards)

        # Score normalization: total_reward / max_possible, clamped to [0, 1]
        if self.max_possible > 0:
            raw_score = total_reward / self.max_possible
        else:
            raw_score = 0.0

        score = max(0.0, min(1.0, raw_score))
        success = score >= self.task.success_threshold

        # ── Detailed breakdown ──
        efficiency = 0.0
        if deal_made and steps > 0:
            # Bonus for fewer steps — max 1.0 if done in 1 step
            efficiency = max(0.0, 1.0 - (steps / self.task.max_rounds))

        return {
            "task": self.task.name,
            "difficulty": self.task.difficulty,
            "score": round(score, 4),
            "success": success,
            "threshold": self.task.success_threshold,
            "total_reward": round(total_reward, 4),
            "max_possible": self.max_possible,
            "steps": steps,
            "deal_made": deal_made,
            "efficiency": round(efficiency, 4),
        }


def get_grader(task: TaskConfig) -> Grader:
    """Create a grader for the given task."""
    return Grader(task)

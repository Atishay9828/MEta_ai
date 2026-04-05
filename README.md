# Strategic Negotiation Simulation — Meta OpenEnv

A reinforcement learning simulation environment developed for the Meta PyTorch OpenEnv Hackathon. This project tests an AI agent's ability to negotiate prices under conditions of partial observability and uncertainty. The environment adheres strictly to the Meta OpenEnv Gymnasium specifications.

## Project Overview

This environment evaluates an LLM agent on realistic B2B and marketplace price negotiation dynamics. Instead of traditional grid-based tasks, the agent is placed in multi-turn conversational exchanges where it must infer opponent constraints and optimize its final profit margins.

The core objectives for the agent are to:
1. Optimize profit by closing favorable agreements.
2. Deduce and adapt to hidden opponent parameters (such as minimum viable limits and behavioral models).
3. Execute multi-step reasoning while adhering to strict environmental boundaries.

## Architecture & Compliance

The codebase strictly utilizes the required `OpenEnv` structural patterns.

*   **Pydantic Enforcement:** `Observation` and `ActionModel` definitions strictly type-check all LLM outputs before they influence the environment state.
*   **State Transparency:** `step()`, `reset()`, and `state()` map directly to expected OpenEnv outputs returning `(observation, reward, done, info)`.
*   **Decoupled Logic Components:** 
    *   `env_wrapper.py`: Manages mathematical boundaries, execution logic, and reward distribution.
    *   `tasks.py`: Defines task bracket parameters (ZOPA margins, total rounds) and holds the programmatic `Grader`.
    *   `inference.py`: Executes the LLM integration loop securely with automated parsing and fallback contingencies.

## Interactive Space Definitions

### Action Space

The agent has three declarative actions available during any given turn:

| Action | Execution Logic |
|---|---|
| `OFFER <price>` | Issues a counter-offer. The `<price>` parameter is constrained to integer values between 100 and 1000. |
| `ACCEPT` | Terminates the episode by agreeing to the `last_opponent_offer`. Calculates profit based on private valuation. |
| `REJECT` | Terminates the episode immediately with no deal, yielding a heavy penalty. |

### Observation Space

The `state()` context exposes exactly what a real-world negotiator would know, while explicitly hiding the opponent's true target threshold.

| Field | Data Type | Implementation Detail |
|---|---|---|
| `agent_value` | integer | The agent's private valuation target (its bottom-line). |
| `current_offer` | integer | The active bid currently on the table. |
| `round` | integer | Current iteration out of maximum rounds allowed. |
| `max_rounds` | integer | Hard limit before a timeout termination. |
| `role` | string | Either "buyer" or "seller". Determines profit calculation algorithms. |
| `last_opponent_action` | string | Indicates "START", "OFFER", or "ACCEPT". |
| `last_opponent_offer` | integer | The direct integer value of the last proposal. |
| `history` | array | A comprehensive step-by-step memory of all previous bids across the episode limit. |

## Reward Formulation

The environment employs both dense shaping signals and sparse terminal rewards to effectively direct the agent toward optimal strategies.

1.  **Terminal Base Reward:** Calculated as `profit × (1 - (round / max_rounds))`. This actively encourages closing positive deals as fast as possible.
2.  **Negative Outcome Penalties:** 
    *   Failing to reach an agreement or forcing a `REJECT` results in a direct `-50.0` score loss.
    *   Accepting a deal that results in negative profit yields an added `-20.0` penalty.
3.  **Aggression Stacking:** Submitting offers that wildly diverge from reasonable limits assigns a cumulative `-2.0` penalty per occurrence.
4.  **Dense Shaping:** Intermediate fractional rewards (`±2.0`) encourage the agent when making minor constructive movements toward the opponent's ZOPA limits.

## Evaluated Tasks

The configuration executes three escalating difficulties managed by different opponent behavior modules.

| Task Profile | Classification | Opponent Bias | Margin of ZOPA | Turn Limit | Baseline Success Threshold |
|---|---|---|---|---|---|
| `task_a_easy` | Easy | Fair | Broad (400 units) | 20 Rounds | 0.2 |
| `task_b_medium` | Medium | Greedy | Constrained (200 units) | 15 Rounds | 0.3 |
| `task_c_hard` | Hard | Impatient | Narrow (120 units) | 6 Rounds | 0.4 |

## Installation and Execution

### System Requirements
*   Python 3.11+
*   HuggingFace Inference Token (`HF_TOKEN`)

### Local Environment Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Assign environment variables:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="insert_token_here"
```

3. Trigger the inference framework:
```bash
python inference.py
```

### Docker Deployment

```bash
docker build -t meta-openenv-negotiation .
docker run -e HF_TOKEN=token -e API_BASE_URL=https://router.huggingface.co/v1 -e MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct meta-openenv-negotiation
```

## Baseline Evaluation Scores

Model Tested: `meta-llama/Meta-Llama-3-8B-Instruct`
API Protocol: `router.huggingface.co/v1`

| Evaluated Task | Score Computed | Rounds Used | Agreement Reached |
|---|---|---|---|
| `task_a_easy` | 0.1138 | 1 | True |
| `task_b_medium` | 0.2333 | 1 | True |
| `task_c_hard` | 0.3472 | 1 | True |

*Note: Baseline results evaluate the model natively against bounded thresholds without few-shot prompting modifications. The model successfully recognized margins and consistently closed tasks on the first available step.*

## License

This project operates under the Apache 2.0 software license.
# 🤝 Strategic Negotiation Environment — OpenEnv

A simulation environment where an AI agent learns to negotiate under uncertainty, compliant with the [Meta OpenEnv specification](https://github.com/meta-llama/open-env).

## 🧠 Overview

This environment simulates **real-world price negotiation** — a task humans do daily in marketplaces, business deals, and automated pricing systems. The agent must:

- **Maximize profit** by negotiating favorable deals
- **Adapt to opponent behavior** (greedy, fair, or impatient personalities)
- **Make multi-step strategic decisions** under partial observability

The agent cannot see the opponent's true valuation or strategy — it must infer patterns and adjust.

---

## 🎮 Action Space

| Action | Description |
|---|---|
| `OFFER <price>` | Make a counter-offer at the given price (100–1000) |
| `ACCEPT` | Accept the current offer on the table |
| `REJECT` | Walk away from the negotiation (terminal, -50 penalty) |

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `current_offer` | int | Current price on the table |
| `round` | int | Current round number |
| `max_rounds` | int | Maximum allowed rounds |
| `role` | string | Agent's role: "buyer" or "seller" |
| `last_opponent_action` | string | "START", "OFFER", or "ACCEPT" |
| `last_opponent_offer` | int | Opponent's last offered price |
| `history` | list | History of all actions this episode |

## 💰 Reward Function

| Event | Reward |
|---|---|
| Successful deal | `profit × (1 - round/max_rounds)` |
| Bad deal (profit < 0) | Additional -20 penalty |
| Rejection / Timeout | -50 |
| Aggressive offers | Cumulative -2 per aggressive step |
| Progress toward deal | Small shaping signal (±2 max) |

---

## 📋 Tasks

| Task | Difficulty | Opponent | ZOPA | Rounds | Threshold |
|---|---|---|---|---|---|
| `task_a_easy` | Easy | Fair | Wide (400) | 20 | 0.2 |
| `task_b_medium` | Medium | Greedy | Narrow (200) | 15 | 0.3 |
| `task_c_hard` | Hard | Impatient | Tight (120) | 6 | 0.4 |

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.11+
- HuggingFace API token

### Install
```bash
pip install -r requirements.txt
```

### Configure Environment Variables
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="your_token_here"
```

### Run Inference
```bash
python inference.py
```

### Docker
```bash
docker build -t negotiation-env .
docker run -e HF_TOKEN=your_token -e API_BASE_URL=https://router.huggingface.co/v1 -e MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct negotiation-env
```

---

## 📊 Baseline Scores

<!-- Person 3: Fill in baseline scores after running inference -->
| Task | Score | Steps | Deal Made |
|---|---|---|---|
| task_a_easy | _TBD_ | _TBD_ | _TBD_ |
| task_b_medium | _TBD_ | _TBD_ | _TBD_ |
| task_c_hard | _TBD_ | _TBD_ | _TBD_ |

---

## 🏗️ Architecture

```
LLM (HuggingFace via OpenAI Client)
        ↓
inference.py (control loop + logging)
        ↓
env_wrapper.py (OpenEnv-compatible environment)
        ↓
tasks.py (task configs + graders)
```

## 📄 License

Apache 2.0
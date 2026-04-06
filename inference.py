"""
Inference Script — OpenEnv Negotiation Environment
Runs LLM agent against all 3 tasks, produces structured logs.
Uses OpenAI-compatible client with HuggingFace router.

STDOUT format (strict — parsed by automated judges):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

All other output goes to stderr.
"""

import os
import re
import sys
from openai import OpenAI
from env_wrapper import EnvWrapper
from tasks import ALL_TASKS, get_grader


def parse_action(llm_text: str):
    """Parse LLM output into (action_str, action_price)."""
    match = re.search(r'(OFFER\s+\d+|ACCEPT|REJECT)', llm_text, re.IGNORECASE)
    if match:
        action = match.group(1).upper()
        if action.startswith("OFFER"):
            parts = action.split()
            try:
                price = int(parts[1])
                return f"OFFER {price}", price, None
            except (IndexError, ValueError):
                return "REJECT", 0, "invalid price in OFFER"
        return action, 0, None
    return None, 0, "no action match"


def run_task(client, model_name: str, task_config):
    """
    Run a single task: LLM negotiates against the environment.
    Returns: (rewards, steps, deal_made, score_info)
    """
    env = EnvWrapper(
        opp_type=task_config.opp_type,
        a_val=task_config.agent_value,
        o_val=task_config.opponent_value,
        agent_role=task_config.agent_role,
        max_rounds=task_config.max_rounds,
    )
    obs = env.reset()

    print(f"[START] task={task_config.name} env=negotiation model={model_name}", flush=True)

    done = False
    step_n = 0
    rewards = []
    deal_made = False
    history_for_prompt = []

    try:
        while not done and step_n < env.max_rounds:
            step_n += 1

            # ── Build prompt with history ──
            history_text = ""
            if history_for_prompt:
                history_lines = []
                for h in history_for_prompt[-5:]:  # Last 5 rounds for context
                    history_lines.append(f"  Round {h['round']}: You → {h['agent']}, Opponent → {h['opp']}")
                history_text = "Negotiation history:\n" + "\n".join(history_lines) + "\n\n"

            target_goal = "buy for as low as possible (below your maximum value)" if obs.role == "buyer" else "sell for as high as possible (above your minimum value)"

            prompt = f"""You are an expert negotiator acting as a {obs.role}. Your objective is to {target_goal} and maximize your profit through strategic multi-round bargaining.

CURRENT STATE:
* Your PRIVATE Valuation: {obs.agent_value} (your absolute limit — NEVER go past this)
* Current offer on the table: {obs.current_offer}
* Round: {step_n} of {obs.max_rounds}
* Opponent's last action: {obs.last_opponent_action}
* Opponent's last offer: {obs.last_opponent_offer}

{history_text}YOUR NEGOTIATION PLAYBOOK:

ROUND-BY-ROUND STRATEGY (you are a {obs.role}):
{"- Round 1: Start AGGRESSIVE. Offer around 30-35% of the opponent's opening price. (e.g., if they open at 1000, offer 300-350). This anchors the negotiation in your favor." if obs.role == "buyer" else "- Round 1: Start AGGRESSIVE. Offer around 2-3x your minimum value. This anchors the negotiation in your favor."}
- Round 2-3: Concede SLOWLY. Increase your offer by only 50-80 per round. Watch how the opponent responds.
- Round 3-4: If the opponent's counter-offer is profitable for you ({"below" if obs.role == "buyer" else "above"} your valuation), ACCEPT it. Otherwise make one final offer near the midpoint.
- Round 5+: You are running out of time. ACCEPT any profitable deal immediately.

SCORING RULES:
1. PROFIT MATTERS MOST: Your score = (your profit) × (time bonus). A great deal on round 3 beats a mediocre deal on round 1.
2. TIME BONUS: Decreases each round. Don't drag past round 5.
3. AGGRESSION PENALTY: Offers extremely far from reasonable (e.g., offering 100 when market is 500+) are penalized. Stay within a plausible range.
4. NEVER REJECT unless it's the absolute last round and you cannot profit.

Choose exactly ONE action:
* OFFER <price> — counter-offer (follow the round strategy above)
* ACCEPT — accept if the opponent's offer gives you good profit
* REJECT — walk away (almost never do this)

Respond with ONLY your action. Example: OFFER 350"""

            action_str = "REJECT"
            action_price = 0
            error_msg = "null"

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.3,
                )
                llm_text = response.choices[0].message.content.strip()

                parsed_action, parsed_price, parse_err = parse_action(llm_text)

                if parsed_action:
                    action_str = parsed_action
                    action_price = parsed_price
                else:
                    # Retry with stricter prompt
                    error_msg = f"parse failed: {parse_err}, retrying"
                    retry_response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": llm_text},
                            {"role": "user", "content": "Output strictly ONLY ONE of: 'OFFER <price>', 'ACCEPT', or 'REJECT'. Nothing else."},
                        ],
                        max_tokens=15,
                        temperature=0.1,
                    )
                    llm_text2 = retry_response.choices[0].message.content.strip()
                    parsed2, price2, err2 = parse_action(llm_text2)
                    if parsed2:
                        action_str = parsed2
                        action_price = price2
                        error_msg = "null"
                    else:
                        action_str = "REJECT"
                        action_price = 0
                        error_msg = "parse error on retry, defaulting to REJECT"

            except Exception as e:
                error_msg = f"API_Error: {str(e)[:50]}"
                action_str = "REJECT"
                action_price = 0

            # ── Step the environment ──
            obs, reward, done, info = env.step(action_str, action_price)
            rewards.append(reward)

            # Track deal
            if done and info.get("deal_type") in ("agent_accepted", "opponent_accepted"):
                deal_made = True

            # Track history for prompting
            history_for_prompt.append({
                "round": step_n,
                "agent": action_str,
                "opp": f"{obs.last_opponent_action} {obs.last_opponent_offer}" if obs.last_opponent_action == "OFFER" else obs.last_opponent_action,
            })

            # ── Log step (stdout — parsed by judges) ──
            log_action = action_str if not action_str.startswith("OFFER") else f"OFFER {action_price}"
            print(f"[STEP] step={step_n} action={log_action} reward={reward:.2f} done={str(done).lower()} error={error_msg}", flush=True)

    finally:
        # [END] MUST always be printed, even on exceptions
        grader = get_grader(task_config)
        result = grader.grade(rewards, step_n, deal_made)
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        score = result['score']
        print(f"[END] success={str(result['success']).lower()} steps={step_n} score={score:.4f} rewards={rewards_str}", flush=True)

    return result


def main():
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Set it with: export HF_TOKEN='your_token_here'", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    # Debug info goes to stderr only
    print("=" * 60, file=sys.stderr)
    print("NEGOTIATION ENVIRONMENT — OpenEnv Inference", file=sys.stderr)
    print(f"Model: {model_name}", file=sys.stderr)
    print(f"API:   {api_base_url}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_results = []

    for task in ALL_TASKS:
        result = run_task(client, model_name, task)
        all_results.append(result)

    # ── Summary to stderr (not parsed) ──
    print("\n" + "=" * 60, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['task']} ({r['difficulty']}): score={r['score']:.4f} "
              f"steps={r['steps']} deal={r['deal_made']} threshold={r['threshold']}",
              file=sys.stderr)

    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  Average Score: {avg_score:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()

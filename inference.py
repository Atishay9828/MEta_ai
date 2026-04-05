import os
import re
import sys
from openai import OpenAI
from env_wrapper import EnvWrapper

def main():
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    env = EnvWrapper(opp_type="fair", a_val=300, o_val=700, agent_role="buyer")
    env.max_rounds = 4
    env.reset()
    
    print(f"[START] task=negotiation env=custom model={model_name}")
    
    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    
    done = False
    step_n = 0
    rewards = []
    
    while not done and step_n < env.max_rounds:
        step_n += 1
        
        prompt = f"""You are negotiating as a {env.role}.
State:
* Current offer: {env.current_offer}
* Round: {env.round}
* Max rounds: {env.max_rounds}

Choose ONE:
* OFFER <price> (Preferred: counter-offer if you do not like the price!)
* ACCEPT
* REJECT"""
        
        action_str = "REJECT"
        action_price = 0
        error_msg = "null"
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3
            )
            llm_text = response.choices[0].message.content.strip()
            
            match = re.search(r'(OFFER\s+\d+|ACCEPT|REJECT)', llm_text, re.IGNORECASE)
            if match:
                action_str = match.group(1).upper()
            else:
                error_msg = "parsing failed, retrying"
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": llm_text}, {"role": "user", "content": "Output strictly ONLY ONE of: 'OFFER <price>', 'ACCEPT', or 'REJECT'."}],
                    max_tokens=15,
                    temperature=0.1
                )
                llm_text2 = response.choices[0].message.content.strip()
                match2 = re.search(r'(OFFER\s+\d+|ACCEPT|REJECT)', llm_text2, re.IGNORECASE)
                if match2:
                    action_str = match2.group(1).upper()
                    error_msg = "null"
                else:
                    action_str = "REJECT"
                    error_msg = "parse error on retry, defaulting to REJECT"
        except Exception as e:
            error_msg = "API_Error"
            action_str = "REJECT"
            
        if action_str.startswith("OFFER"):
            try:
                action_price = int(action_str.split()[1])
            except ValueError:
                action_str = "REJECT"
                action_price = 0
                error_msg = "invalid price format"
        elif action_str == "ACCEPT":
            action_str = "ACCEPT"
        elif action_str == "REJECT":
            action_str = "REJECT"
            
        # Strip potential garbage
        if "OFFER" in action_str:
            action_str = f"OFFER {action_price}"
            
        reward, d = env.step(action_str, action_price)
        done = d
        rewards.append(reward)
        
        print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")

    
    # SCORING
    max_possible_reward = float(abs(env.agent_value - env.opponent_value))
    score = sum(rewards) / max_possible_reward if max_possible_reward > 0 else 0.0
    score = max(0.0, min(1.0, score))
    success = score > 0.3
    
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={step_n} score={score:.4f} rewards={rewards_str}")

if __name__ == "__main__":
    main()

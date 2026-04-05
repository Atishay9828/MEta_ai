"""Quick validation test for the environment — no API keys needed."""
import random
random.seed(42)

from env_wrapper import EnvWrapper, Observation
from tasks import ALL_TASKS, get_grader

print("=" * 50)
print("TEST 1: Multi-round negotiation")
print("=" * 50)
env = EnvWrapper(opp_type="fair", a_val=800, o_val=400, agent_role="buyer", max_rounds=10)
obs = env.reset()
print(f"Initial offer: {obs.current_offer}")

offers = [650, 600, 550, 500, 480, 450, 420, 400, 400, 400]
for i, price in enumerate(offers):
    obs, r, d, info = env.step("OFFER", price)
    opp_info = f"{obs.last_opponent_action} {obs.last_opponent_offer}"
    print(f"  R{i+1}: OFFER {price} -> Opp {opp_info} | reward={r:.2f} done={d}")
    if d:
        deal_type = info.get("deal_type", "none")
        deal_price = info.get("deal_price", "N/A")
        print(f"  >>> Deal: {deal_type}, price={deal_price}")
        break

print(f"  History entries: {len(obs.history)}")
print()

print("=" * 50)
print("TEST 2: Value-based clamping")
print("=" * 50)
from env_wrapper import Opponent
bugs = 0
for trial in range(100):
    opp = Opponent("fair", 500, "seller")
    for rnd in range(20):
        action, price = opp.get_response(rnd, 300, 250, "OFFER")
        if action == "OFFER" and price < 500:
            bugs += 1
            print(f"  BUG: trial={trial} round={rnd} seller offered {price} < 500")
            break
if bugs == 0:
    print("  PASS: Seller never offered below own value (100 trials x 20 rounds)")
else:
    print(f"  FAIL: {bugs} violations found")
print()

print("=" * 50)
print("TEST 3: Cumulative aggression penalty")
print("=" * 50)
env2 = EnvWrapper(opp_type="greedy", a_val=800, o_val=500, agent_role="buyer", max_rounds=20)
env2.reset()
# Make multiple aggressive offers (>150 away from opp_val=500, so <350 or >650)
for i in range(5):
    obs, r, d, info = env2.step("OFFER", 200)  # 300 away from 500 → aggressive
    print(f"  R{i+1}: penalty_so_far={env2.cumulative_aggression_penalty}")
    if d:
        break

expected_penalty = 10.0  # 5 rounds x 2.0 per round
actual_penalty = env2.cumulative_aggression_penalty
print(f"  Expected cumulative penalty: {expected_penalty}, Actual: {actual_penalty}")
print(f"  {'PASS' if actual_penalty == expected_penalty else 'FAIL'}")
print()

print("=" * 50)
print("TEST 4: Task configs and graders")
print("=" * 50)
for task in ALL_TASKS:
    grader = get_grader(task)
    # Test with sample rewards
    result = grader.grade([0.0, 0.0, 50.0], 3, True)
    print(f"  {task.name} ({task.difficulty}): score={result['score']}, success={result['success']}")
print()

print("=" * 50)
print("TEST 5: state() method")
print("=" * 50)
env3 = EnvWrapper(opp_type="fair", a_val=800, o_val=400, agent_role="buyer")
env3.reset()
s = env3.state()
assert isinstance(s, Observation), "state() must return Observation"
assert s.role == "buyer"
assert s.round == 0
print(f"  PASS: state() returns Observation with role={s.role}, round={s.round}")
print()

print("=" * 50)
print("TEST 6: ACCEPT and REJECT")
print("=" * 50)
# ACCEPT test
env4 = EnvWrapper(opp_type="fair", a_val=800, o_val=400, agent_role="buyer")
env4.reset()
env4.step("OFFER", 500)  # Get an opponent counter
obs, r, d, info = env4.step("ACCEPT", 0)
print(f"  ACCEPT: reward={r:.2f} done={d} deal_type={info.get('deal_type')}")

# REJECT test
env5 = EnvWrapper(opp_type="fair", a_val=800, o_val=400, agent_role="buyer")
env5.reset()
obs, r, d, info = env5.step("REJECT", 0)
print(f"  REJECT: reward={r:.2f} done={d} deal_type={info.get('deal_type')}")
print()

print("=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)

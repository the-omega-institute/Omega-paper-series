# gridworld_shield.py
import random
import math
from dataclasses import dataclass
import numpy as np

# --------------------------
# GridWorld environment
# --------------------------
class GridWorld:
    def __init__(self, size=8, hazards=None, start=(0, 0), goal=None):
        self.size = size
        self.start = start
        self.state = start
        if goal is None:
            goal = (size - 1, size - 1)
        self.goal = goal
        if hazards is None:
            # random hazards, avoid start and goal
            hazards = set()
            while len(hazards) < 8:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                if (x, y) != start and (x, y) != goal:
                    hazards.add((x, y))
        self.hazards = set(hazards)
        # actions: 0 up, 1 down, 2 left, 3 right
        self.actions = [0, 1, 2, 3]
        self.max_steps = 50
        self.steps = 0

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self._state_id(self.state)

    def _state_id(self, pos):
        x, y = pos
        return x * self.size + y

    def _pos_from_id(self, sid):
        x = sid // self.size
        y = sid % self.size
        return (x, y)

    def step(self, action):
        self.steps += 1
        x, y = self.state
        if action == 0:      # up
            x -= 1
        elif action == 1:    # down
            x += 1
        elif action == 2:    # left
            y -= 1
        elif action == 3:    # right
            y += 1
        # keep inside grid
        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))
        new_state = (x, y)
        self.state = new_state
        reward = 0.0
        done = False
        catastrophic = False
        if new_state in self.hazards:
            reward = -1.0
            done = True
            catastrophic = True
        elif new_state == self.goal:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
        return self._state_id(new_state), reward, done, {
            "catastrophic": catastrophic
        }

# --------------------------
# Shield based on hazard set
# --------------------------
class Shield:
    def __init__(self, env: GridWorld):
        self.env = env
        self.size = env.size

    def is_hazard(self, pos):
        return pos in self.env.hazards

    def next_pos(self, state_id, action):
        x, y = self.env._pos_from_id(state_id)
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1
        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))
        return (x, y)

    def safe_action(self, state_id, proposed_action):
        """If proposed action leads to hazard, override with a random safe action."""
        proposed_next = self.next_pos(state_id, proposed_action)
        if self.is_hazard(proposed_next):
            safe_actions = []
            for a in self.env.actions:
                pos = self.next_pos(state_id, a)
                if not self.is_hazard(pos):
                    safe_actions.append(a)
            if not safe_actions:
                return proposed_action, False
            return random.choice(safe_actions), True
        else:
            return proposed_action, False

# --------------------------
# Tabular Q-learning
# --------------------------
@dataclass
class EvalStats:
    catastrophic_rate: float
    avg_return: float
    intervention_rate: float

def q_learning(env: GridWorld,
               shield: Shield = None,
               episodes=50000,
               gamma=0.99,
               alpha=0.1,
               epsilon_start=1.0,
               epsilon_end=0.05,
               epsilon_decay=30000):
    n_states = env.size * env.size
    n_actions = len(env.actions)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    total_steps = 0
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                math.exp(-total_steps / max(1, epsilon_decay))
            total_steps += 1
            if random.random() < epsilon:
                action = random.choice(env.actions)
            else:
                action = int(np.argmax(Q[state]))
            intervened = False
            if shield is not None:
                final_action, intervened = shield.safe_action(state, action)
            else:
                final_action = action
            next_state, reward, done, info = env.step(final_action)
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next * (0.0 if done else 1.0)
            td_error = td_target - Q[state, final_action]
            Q[state, final_action] += alpha * td_error
            state = next_state
        if (ep + 1) % 5000 == 0:
            print(f"Episode {ep+1}/{episodes}")
    return Q

def evaluate(env: GridWorld,
             Q,
             shield: Shield = None,
             episodes=500):
    n_states = env.size * env.size
    catastrophic_count = 0
    total_return = 0.0
    total_steps = 0
    total_interventions = 0
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = int(np.argmax(Q[state]))
            if shield is not None:
                final_action, intervened = shield.safe_action(state, action)
                if intervened:
                    total_interventions += 1
            else:
                final_action = action
                intervened = False
            next_state, reward, done, info = env.step(final_action)
            ep_return += reward
            if info.get("catastrophic", False):
                catastrophic_count += 1
            total_steps += 1
            state = next_state
        total_return += ep_return
    catastrophic_rate = catastrophic_count / episodes
    avg_return = total_return / episodes
    intervention_rate = (total_interventions / total_steps
                         if total_steps > 0 else 0.0)
    return EvalStats(catastrophic_rate=catastrophic_rate,
                     avg_return=avg_return,
                     intervention_rate=intervention_rate)

def main():
    random.seed(0)
    np.random.seed(0)
    env = GridWorld(size=8)
    shield = Shield(env)
    print("Training unshielded agent...")
    Q_unshielded = q_learning(env, shield=None, episodes=50000)
    print("Training shielded agent...")
    Q_shielded = q_learning(env, shield=shield, episodes=50000)
    print("Evaluating unshielded agent...")
    stats_unshielded = evaluate(env, Q_unshielded, shield=None, episodes=500)
    print("Evaluating shielded agent...")
    stats_shielded = evaluate(env, Q_shielded, shield=shield, episodes=500)
    print(f"Unshielded: cat_rate={stats_unshielded.catastrophic_rate:.3f}, "
          f"avg_return={stats_unshielded.avg_return:.3f}, "
          f"interv_rate={stats_unshielded.intervention_rate:.3f}")
    print(f"Shielded: cat_rate={stats_shielded.catastrophic_rate:.3f}, "
          f"avg_return={stats_shielded.avg_return:.3f}, "
          f"interv_rate={stats_shielded.intervention_rate:.3f}")

if __name__ == "__main__":
    main()


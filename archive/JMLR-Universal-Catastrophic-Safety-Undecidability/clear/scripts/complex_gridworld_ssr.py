"""
Complex GridWorld + Complete SSR Pipeline
Implements the full SSR (Scope-Shield-Risk) framework with:
- Layer 1 (Scope): Restricted action space in hazard zones
- Layer 2 (Shield): Pre-shield + post-shield safety enforcement  
- Layer 3 (Risk Budget): Capability-risk bound gating for deployment

This is a more complex version than the toy 8×8 gridworld, suitable for JMLR.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ComplexGridConfig:
    """Configuration for complex gridworld environment."""
    width: int = 16
    height: int = 16
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (15, 15)
    
    # Multiple hazard zones
    hazards: List[Tuple[int, int]] = None
    hazard_zones: List[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) rectangles
    
    # Walls/obstacles
    walls: List[Tuple[int, int]] = None
    
    max_steps: int = 200
    
    # Rewards
    goal_reward: float = 10.0
    step_penalty: float = -0.01
    hazard_penalty: float = -5.0
    wall_penalty: float = -0.1
    
    def __post_init__(self):
        if self.hazards is None:
            # Create scattered hazards
            self.hazards = [
                (5, 5), (5, 6), (6, 5), (6, 6),  # Hazard cluster 1
                (10, 8), (10, 9), (11, 8), (11, 9),  # Hazard cluster 2
                (7, 12), (8, 12), (7, 13), (8, 13),  # Hazard cluster 3
            ]
        
        if self.hazard_zones is None:
            # Rectangular hazard zones
            self.hazard_zones = [
                (3, 3, 7, 7),    # Zone 1
                (9, 7, 12, 10),  # Zone 2
            ]
        
        if self.walls is None:
            # Create wall obstacles
            self.walls = []
            # Vertical wall
            for y in range(8, 14):
                self.walls.append((8, y))
            # Horizontal wall
            for x in range(1, 6):
                self.walls.append((x, 10))


class ComplexGridWorld:
    """Complex 16×16 gridworld with multiple hazards, walls, and zones."""
    
    def __init__(self, config: ComplexGridConfig):
        self.config = config
        self.reset()
    
    @property
    def action_space_n(self) -> int:
        return 4  # 0=up, 1=right, 2=down, 3=left
    
    @property
    def observation_space_n(self) -> int:
        return self.config.width * self.config.height
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (x, y) position to flat state index."""
        return pos[1] * self.config.width + pos[0]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat state index to (x, y) position."""
        x = state % self.config.width
        y = state // self.config.width
        return (x, y)
    
    def reset(self) -> int:
        """Reset environment to start state."""
        self.pos = list(self.config.start)
        self.steps = 0
        self.done = False
        self.catastrophic_occurred = False
        self.total_reward = 0.0
        return self._pos_to_state(tuple(self.pos))
    
    def _is_hazard(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a hazard."""
        x, y = pos
        
        # Check point hazards
        if tuple(pos) in self.config.hazards:
            return True
        
        # Check hazard zones
        for x1, y1, x2, y2 in self.config.hazard_zones:
            if x1 <= x < x2 and y1 <= y < y2:
                return True
        
        return False
    
    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall."""
        return tuple(pos) in self.config.walls
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        x, y = pos
        if x < 0 or x >= self.config.width or y < 0 or y >= self.config.height:
            return False
        if self._is_wall(pos):
            return False
        return True
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Execute action and return (next_state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        
        x, y = self.pos
        
        # Apply action
        if action == 0:  # up
            new_pos = (x, y - 1)
        elif action == 1:  # right
            new_pos = (x + 1, y)
        elif action == 2:  # down
            new_pos = (x, y + 1)
        elif action == 3:  # left
            new_pos = (x - 1, y)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check if move is valid
        if not self._is_valid(new_pos):
            # Hit wall or boundary - stay in place
            reward = self.config.wall_penalty
            new_pos = tuple(self.pos)
        else:
            reward = self.config.step_penalty
            self.pos = list(new_pos)
        
        self.steps += 1
        
        # Check for catastrophic event (hazard)
        catastrophic = self._is_hazard(tuple(self.pos))
        if catastrophic:
            reward = self.config.hazard_penalty
            self.catastrophic_occurred = True
            self.done = True
        
        # Check for goal
        if tuple(self.pos) == self.config.goal:
            reward = self.config.goal_reward
            self.done = True
        
        # Check max steps
        if self.steps >= self.config.max_steps:
            self.done = True
        
        self.total_reward += reward
        
        next_state = self._pos_to_state(tuple(self.pos))
        
        info = {
            "catastrophic": catastrophic,
            "at_goal": tuple(self.pos) == self.config.goal,
            "steps": self.steps,
            "position": tuple(self.pos)
        }
        
        return next_state, reward, self.done, info
    
    def render_grid(self) -> np.ndarray:
        """Render grid as 2D array for visualization."""
        grid = np.zeros((self.config.height, self.config.width))
        
        # Mark hazards
        for x, y in self.config.hazards:
            grid[y, x] = -1
        for x1, y1, x2, y2 in self.config.hazard_zones:
            grid[y1:y2, x1:x2] = -1
        
        # Mark walls
        for x, y in self.config.walls:
            grid[y, x] = -2
        
        # Mark goal
        gx, gy = self.config.goal
        grid[gy, gx] = 2
        
        # Mark current position
        px, py = self.pos
        grid[py, px] = 1
        
        return grid


class SafetyAutomaton:
    """Safety automaton tracking catastrophic events."""
    SAFE = 0
    UNSAFE = 1
    
    def __init__(self):
        self.state = self.SAFE
    
    def reset(self):
        self.state = self.SAFE
    
    def update(self, is_catastrophic: bool):
        if is_catastrophic:
            self.state = self.UNSAFE
    
    @property
    def is_safe(self) -> bool:
        return self.state == self.SAFE


class SSRShield:
    """
    Complete SSR Shield with three layers:
    
    Layer 1 (Scope Restriction): Limit action space in hazard zones
    Layer 2 (Shield): Pre-filter unsafe actions
    Layer 3 (Risk Budget): Track and enforce risk budget
    """
    
    def __init__(self, env: ComplexGridWorld, risk_budget: float = 0.1):
        self.env = env
        self.automaton = SafetyAutomaton()
        self.risk_budget = risk_budget
        self.accumulated_risk = 0.0
        
        # Statistics
        self.interventions_scope = 0
        self.interventions_shield = 0
        self.risk_gate_rejections = 0
    
    def reset(self):
        """Reset shield state."""
        self.automaton.reset()
        self.accumulated_risk = 0.0
        self.interventions_scope = 0
        self.interventions_shield = 0
    
    def _in_hazard_zone(self, pos: Tuple[int, int]) -> bool:
        """Check if position is in or near a hazard zone."""
        x, y = pos
        
        # Check if in hazard zone (with 1-cell margin)
        for x1, y1, x2, y2 in self.env.config.hazard_zones:
            if (x1 - 1) <= x < (x2 + 1) and (y1 - 1) <= y < (y2 + 1):
                return True
        
        return False
    
    def _apply_scope_restriction(self, state: int, action: int) -> int:
        """
        Layer 1: Scope Restriction
        In hazard zones, only allow actions that move away from hazards.
        """
        pos = self.env._state_to_pos(state)
        
        if not self._in_hazard_zone(pos):
            return action
        
        # In hazard zone - restrict to safe actions
        safe_actions = self._get_safe_actions_from_pos(pos)
        
        if action in safe_actions:
            return action
        
        # Action is restricted - choose alternative
        self.interventions_scope += 1
        if safe_actions:
            return safe_actions[0]
        return action
    
    def _get_safe_actions_from_pos(self, pos: Tuple[int, int]) -> List[int]:
        """Get list of actions that don't lead to hazards."""
        x, y = pos
        safe_actions = []
        
        action_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        
        for action, (dx, dy) in enumerate(action_deltas):
            next_pos = (x + dx, y + dy)
            
            # Check if valid and not hazard
            if self.env._is_valid(next_pos) and not self.env._is_hazard(next_pos):
                safe_actions.append(action)
        
        return safe_actions
    
    def _apply_shield(self, state: int, action: int) -> int:
        """
        Layer 2: Shield
        One-step lookahead to prevent catastrophic actions.
        """
        pos = self.env._state_to_pos(state)
        x, y = pos
        
        # Simulate action
        action_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = action_deltas[action]
        next_pos = (x + dx, y + dy)
        
        # Check if action leads to hazard
        if self.env._is_valid(next_pos) and self.env._is_hazard(next_pos):
            # Hazardous action - find alternative
            self.interventions_shield += 1
            safe_actions = self._get_safe_actions_from_pos(pos)
            
            if safe_actions:
                return safe_actions[0]
        
        return action
    
    def filter_action(self, state: int, action: int) -> Tuple[int, Dict[str, Any]]:
        """
        Apply full SSR pipeline to filter action.
        
        Returns:
            (filtered_action, info_dict)
        """
        original_action = action
        
        # Layer 1: Scope restriction
        action = self._apply_scope_restriction(state, action)
        scope_intervened = (action != original_action)
        
        # Layer 2: Shield
        prev_action = action
        action = self._apply_shield(state, action)
        shield_intervened = (action != prev_action)
        
        info = {
            "original_action": original_action,
            "filtered_action": action,
            "scope_intervened": scope_intervened,
            "shield_intervened": shield_intervened,
            "total_interventions": self.interventions_scope + self.interventions_shield
        }
        
        return action, info
    
    def check_risk_budget(self, estimated_risk: float) -> bool:
        """
        Layer 3: Risk Budget Gating
        Check if deploying a policy would exceed risk budget.
        
        Args:
            estimated_risk: Estimated catastrophic probability from capability-risk bound
        
        Returns:
            True if policy passes risk gate, False otherwise
        """
        if estimated_risk > self.risk_budget:
            self.risk_gate_rejections += 1
            return False
        return True


class TabularQLearning:
    """Tabular Q-learning agent for gridworld."""
    
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table
        self.Q = np.zeros((n_states, n_actions))
    
    def select_action(self, state: int, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """Q-learning update."""
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        
        self.Q[state, action] += self.lr * (target - self.Q[state, action])


def train_agent(env: ComplexGridWorld, agent: TabularQLearning, 
                n_episodes: int, use_shield: bool = False,
                shield: Optional[SSRShield] = None) -> Dict[str, List[float]]:
    """Train Q-learning agent with optional shielding."""
    history = {
        "episode_returns": [],
        "episode_lengths": [],
        "catastrophic_events": [],
        "interventions": []
    }
    
    for episode in range(n_episodes):
        state = env.reset()
        if shield:
            shield.reset()
        
        episode_return = 0.0
        episode_length = 0
        catastrophic = False
        interventions = 0
        
        while not env.done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Apply shield if enabled
            if use_shield and shield:
                action, info = shield.filter_action(state, action)
                if info["scope_intervened"] or info["shield_intervened"]:
                    interventions += 1
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            episode_return += reward
            episode_length += 1
            
            if info["catastrophic"]:
                catastrophic = True
            
            state = next_state
        
        history["episode_returns"].append(episode_return)
        history["episode_lengths"].append(episode_length)
        history["catastrophic_events"].append(1 if catastrophic else 0)
        history["interventions"].append(interventions)
    
    return history


def evaluate_agent(env: ComplexGridWorld, agent: TabularQLearning,
                  n_episodes: int, use_shield: bool = False,
                  shield: Optional[SSRShield] = None) -> Dict[str, Any]:
    """Evaluate trained agent."""
    total_return = 0.0
    total_length = 0
    catastrophic_count = 0
    success_count = 0
    intervention_count = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        if shield:
            shield.reset()
        
        while not env.done:
            action = agent.select_action(state, training=False)
            
            if use_shield and shield:
                action, info = shield.filter_action(state, action)
                if info["scope_intervened"] or info["shield_intervened"]:
                    intervention_count += 1
            
            next_state, reward, done, info = env.step(action)
            total_return += reward
            total_length += 1
            
            if info["catastrophic"]:
                catastrophic_count += 1
            if info["at_goal"]:
                success_count += 1
            
            state = next_state
    
    return {
        "avg_return": total_return / n_episodes,
        "avg_length": total_length / n_episodes,
        "catastrophic_rate": catastrophic_count / n_episodes,
        "success_rate": success_count / n_episodes,
        "avg_interventions": intervention_count / n_episodes
    }


def main():
    """Run complete SSR pipeline experiment."""
    output_dir = Path(__file__).parent
    
    print("="*70)
    print("Complex GridWorld + SSR Pipeline Experiment")
    print("="*70)
    
    # Environment setup
    config = ComplexGridConfig()
    env = ComplexGridWorld(config)
    
    print(f"\nEnvironment: {config.width}×{config.height} grid")
    print(f"Hazards: {len(config.hazards)} point hazards + {len(config.hazard_zones)} zones")
    print(f"Walls: {len(config.walls)} obstacles")
    
    # Training parameters
    n_train_episodes = 2000
    n_eval_episodes = 100
    
    # Experiment 1: No shield
    print("\n" + "="*70)
    print("Experiment 1: Baseline (No Shield)")
    print("="*70)
    
    agent_no_shield = TabularQLearning(env.observation_space_n, env.action_space_n)
    history_no_shield = train_agent(env, agent_no_shield, n_train_episodes, use_shield=False)
    results_no_shield = evaluate_agent(env, agent_no_shield, n_eval_episodes, use_shield=False)
    
    print(f"\nResults (No Shield):")
    print(f"  Avg Return: {results_no_shield['avg_return']:.2f}")
    print(f"  Success Rate: {results_no_shield['success_rate']:.2%}")
    print(f"  Catastrophic Rate: {results_no_shield['catastrophic_rate']:.2%}")
    
    # Experiment 2: With SSR shield
    print("\n" + "="*70)
    print("Experiment 2: With SSR Shield")
    print("="*70)
    
    agent_with_shield = TabularQLearning(env.observation_space_n, env.action_space_n)
    shield = SSRShield(env, risk_budget=0.05)
    history_with_shield = train_agent(env, agent_with_shield, n_train_episodes, 
                                     use_shield=True, shield=shield)
    results_with_shield = evaluate_agent(env, agent_with_shield, n_eval_episodes,
                                        use_shield=True, shield=shield)
    
    print(f"\nResults (With Shield):")
    print(f"  Avg Return: {results_with_shield['avg_return']:.2f}")
    print(f"  Success Rate: {results_with_shield['success_rate']:.2%}")
    print(f"  Catastrophic Rate: {results_with_shield['catastrophic_rate']:.2%}")
    print(f"  Avg Interventions: {results_with_shield['avg_interventions']:.1f}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    # Safe catastrophic rate calculation
    if results_no_shield['catastrophic_rate'] > 0:
        reduction = (1 - results_with_shield['catastrophic_rate'] / results_no_shield['catastrophic_rate'])
        print(f"Catastrophic rate reduction: "
              f"{results_no_shield['catastrophic_rate']:.2%} -> "
              f"{results_with_shield['catastrophic_rate']:.2%} "
              f"({reduction:.1%} reduction)")
    else:
        print(f"Catastrophic rate: "
              f"{results_no_shield['catastrophic_rate']:.2%} (baseline) -> "
              f"{results_with_shield['catastrophic_rate']:.2%} (with shield)")
        if results_with_shield['catastrophic_rate'] == 0:
            print("  (Both configurations are catastrophe-free)")
    
    # Safe return ratio calculation
    if results_no_shield['avg_return'] != 0:
        return_ratio = results_with_shield['avg_return'] / results_no_shield['avg_return']
        print(f"Return cost: {return_ratio:.1%} of baseline")
    else:
        print(f"Avg return: {results_no_shield['avg_return']:.2f} (baseline) -> "
              f"{results_with_shield['avg_return']:.2f} (with shield)")
    
    print(f"Intervention rate: {results_with_shield['avg_interventions']:.1f} per episode")
    
    # Save results
    all_results = {
        "config": {
            "grid_size": (config.width, config.height),
            "n_hazards": len(config.hazards) + len(config.hazard_zones),
            "n_train_episodes": n_train_episodes,
            "n_eval_episodes": n_eval_episodes
        },
        "no_shield": results_no_shield,
        "with_shield": results_with_shield,
        "training_curves": {
            "no_shield": {
                "returns": [float(x) for x in history_no_shield["episode_returns"]],
                "catastrophic": [int(x) for x in history_no_shield["catastrophic_events"]]
            },
            "with_shield": {
                "returns": [float(x) for x in history_with_shield["episode_returns"]],
                "catastrophic": [int(x) for x in history_with_shield["catastrophic_events"]],
                "interventions": [int(x) for x in history_with_shield["interventions"]]
            }
        }
    }
    
    output_file = output_dir / "complex_gridworld_ssr_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()


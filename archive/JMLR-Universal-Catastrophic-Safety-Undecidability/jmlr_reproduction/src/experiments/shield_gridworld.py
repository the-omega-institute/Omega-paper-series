from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
import random

@dataclass
class GridConfig:
    width: int = 5
    height: int = 5
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    hazards: Tuple[Tuple[int, int], ...] = ((2, 2),)
    max_steps: int = 50

class GridWorld:
    """
    一个极简 GridWorld 环境:
    - 状态是坐标 (x, y)
    - 动作: 0=上,1=右,2=下,3=左
    - 遇到 hazard 视为 'catastrophic' 终止
    """
    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.reset()

    @property
    def action_space_n(self) -> int:
        return 4

    def reset(self) -> Tuple[int, int]:
        self.pos = list(self.cfg.start)
        self.steps = 0
        self.done = False
        self.catastrophic = False
        return tuple(self.pos)

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Call reset() before step() after episode ends")

        x, y = self.pos
        if action == 0:
            y -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y += 1
        elif action == 3:
            x -= 1

        x = max(0, min(self.cfg.width - 1, x))
        y = max(0, min(self.cfg.height - 1, y))
        self.pos = [x, y]
        self.steps += 1

        reward = -0.01
        self.done = False
        self.catastrophic = False

        if (x, y) in self.cfg.hazards:
            reward = -1.0
            self.done = True
            self.catastrophic = True
        elif (x, y) == self.cfg.goal:
            reward = 1.0
            self.done = True
        elif self.steps >= self.cfg.max_steps:
            self.done = True

        info = {"catastrophic": self.catastrophic}
        return (x, y), reward, self.done, info

class SafetyAutomaton:
    """
    极简 safety automaton:
    - 只有两个状态: safe=0, bad=1
    - 一旦进入 bad 就永远停在 bad
    """
    SAFE = 0
    BAD = 1

    def __init__(self):
        self.state = self.SAFE

    def reset(self):
        self.state = self.SAFE

    def update(self, is_catastrophic: bool) -> None:
        if self.state == self.BAD:
            return
        if is_catastrophic:
            self.state = self.BAD

    @property
    def in_bad(self) -> bool:
        return self.state == self.BAD


class ShieldWrapper:
    """
    Pre-shield: 在执行动作前, 向前看一步, 如果会到 hazard 则替换为安全动作.
    这里用环境 transition 的知识来实现, 更复杂版本可以用 safety automaton + 模型预测.
    """
    def __init__(self, env: GridWorld):
        self.env = env
        self.automaton = SafetyAutomaton()

    def reset(self):
        obs = self.env.reset()
        self.automaton.reset()
        return obs

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # 预判: 如果该动作导致 catastrophic, 尝试改用别的动作
        safe_action = self._pick_safe_action(action)
        obs, reward, done, info = self.env.step(safe_action)
        self.automaton.update(info.get("catastrophic", False))
        info["original_action"] = action
        info["shielded_action"] = safe_action
        return obs, reward, done, info

    def _pick_safe_action(self, action: int) -> int:
        # 简单版: 用单步 lookahead 检查 4 个动作的后果
        if self._is_safe_action(action):
            return action
        action_space = list(range(self.env.action_space_n))
        random.shuffle(action_space)
        for a in action_space:
            if self._is_safe_action(a):
                return a
        return action  # 无法找到安全动作时只能执行原动作

    def _is_safe_action(self, action: int) -> bool:
        # 模拟一步看看是否到 hazard
        x, y = self.env.pos
        nx, ny = x, y
        if action == 0:
            ny -= 1
        elif action == 1:
            nx += 1
        elif action == 2:
            ny += 1
        elif action == 3:
            nx -= 1

        nx = max(0, min(self.env.cfg.width - 1, nx))
        ny = max(0, min(self.env.cfg.height - 1, ny))
        return (nx, ny) not in self.env.cfg.hazards

def random_policy(state: Tuple[int, int], action_space_n: int) -> int:
    return np.random.randint(action_space_n)

def run_episode(env, use_shield: bool = False) -> Dict[str, Any]:
    if use_shield:
        wrapped = ShieldWrapper(env)
        obs = wrapped.reset()
        step_fn = wrapped.step
    else:
        obs = env.reset()
        step_fn = env.step

    total_reward = 0.0
    catastrophic = False
    n_steps = 0
    n_shielded = 0

    done = False
    while not done:
        action = random_policy(obs, env.action_space_n)
        next_obs, reward, done, info = step_fn(action)
        total_reward += reward
        catastrophic = catastrophic or info.get("catastrophic", False)
        if use_shield and info.get("original_action") != info.get("shielded_action"):
            n_shielded += 1
        obs = next_obs
        n_steps += 1

    return {
        "total_reward": total_reward,
        "catastrophic": catastrophic,
        "n_steps": n_steps,
        "n_shielded": n_shielded,
    }

def run_shield_experiment(n_episodes: int = 1000):
    cfg = GridConfig()
    env = GridWorld(cfg)

    stats_no_shield: List[Dict[str, Any]] = []
    stats_with_shield: List[Dict[str, Any]] = []

    for _ in range(n_episodes):
        stats_no_shield.append(run_episode(env, use_shield=False))
        stats_with_shield.append(run_episode(env, use_shield=True))

    def summarize(stats: List[Dict[str, Any]]) -> Dict[str, float]:
        catastrophic_rate = sum(s["catastrophic"] for s in stats) / len(stats)
        avg_reward = sum(s["total_reward"] for s in stats) / len(stats)
        avg_shielded = (
            sum(s.get("n_shielded", 0) for s in stats) / len(stats)
        )
        return {
            "catastrophic_rate": catastrophic_rate,
            "avg_reward": avg_reward,
            "avg_shielded_actions": avg_shielded,
        }

    print("No shield:", summarize(stats_no_shield))
    print("With shield:", summarize(stats_with_shield))

if __name__ == "__main__":
    run_shield_experiment(n_episodes=500)


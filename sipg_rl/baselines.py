"""Phase-1 comparison policies. Each exposes act(env) -> action index."""
from __future__ import annotations
from collections import deque
import numpy as np
import torch
from .env import INVOKE, STOP, MOVE0
from .model import collate


class NoOpPolicy:
    name = "no-op (source)"
    def reset(self): pass
    def act(self, env): return STOP


class RandomPolicy:
    name = "random traversal"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self): pass
    def act(self, env):
        return int(self.rng.choice(np.where(env.action_mask())[0]))


class InvokeAllPolicy:
    """Visit every node (DFS over the graph) and fire every prompt - the non-selective upper bound on edits."""
    name = "invoke-all"
    def reset(self): self.stack = None
    def act(self, env):
        if env.invoked[env.cur] == 0:
            return INVOKE
        n = env.g.n
        todo = [j for j in range(n) if env.invoked[j] == 0]
        if not todo:
            return STOP
        return MOVE0 + _next_hop(env, todo)


def _next_hop(env, targets):
    """First step of a shortest path from env.cur to the nearest target (BFS)."""
    adj, start, tgt = env.g.adj, env.cur, set(targets)
    q, prev = deque([start]), {start: None}
    while q:
        u = q.popleft()
        if u in tgt and u != start:
            while prev[u] != start:
                u = prev[u]
            return u
        for v in np.where(adj[u] >= 0)[0]:
            if v not in prev:
                prev[v] = u
                q.append(int(v))
    return None


class GreedyReadabilityPolicy:
    """No learning: repeatedly go to the not-yet-invoked node whose prompt most increases S (readability
    only - ignores fidelity and edit cost, as in the proposal's 'greedy-readability traversal') and fire it."""
    name = "greedy-readability"
    def reset(self): pass
    def act(self, env):
        gains = {j: env.peek(j)[0] for j in range(env.g.n) if env.invoked[j] == 0}
        gains = {j: g for j, g in gains.items() if g > 1e-6}
        if not gains:
            return STOP
        best = max(gains, key=gains.get)
        if best == env.cur:
            return INVOKE
        return MOVE0 + _next_hop(env, [best])


class LearnedPolicy:
    def __init__(self, model, greedy=True, name="SIPG-RL (PPO)"):
        self.model, self.greedy, self.name = model, greedy, name
    def reset(self): pass
    @torch.no_grad()
    def act(self, env):
        logits, _ = self.model(collate([env.obs()]))
        if self.greedy:
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())


TERM_KINDS = {"LEX", "NOM", "AMOD", "CPLX"}


class LearnedThenSweep:
    """Learned policy first; when it wants to STOP, also try term nodes it never fired. Needed with an LLM
    rewriter because the policy was trained where CPLX/unknown-term prompts were no-ops and learned to skip them.
    Every step records who chose it (`last_driver`) so the audit trail stays honest."""
    name = "policy + sweep"

    def __init__(self, base, kinds=TERM_KINDS):
        """kinds=None sweeps *every* node (apply-all mode)."""
        self.base, self.kinds, self.last_driver = base, kinds, "policy"

    def reset(self):
        self.base.reset()

    def act(self, env):
        a = self.base.act(env)
        self.last_driver = "policy"
        todo = [n.id for n in env.g.nodes if (self.kinds is None or n.kind in self.kinds) and env.invoked[n.id] == 0]
        # let the policy wander only while enough step budget remains to finish the sweep
        tight = env.max_steps - env.steps <= 2 * len(todo) + 1
        if a != STOP and not (tight and todo):
            return a
        if not todo:
            return STOP
        if env.cur in todo:
            self.last_driver = "sweep"
            return INVOKE
        hop = _next_hop(env, todo)
        if hop is None:
            return STOP
        self.last_driver = "sweep"
        return MOVE0 + hop


def run_episode(env, graph, policy):
    env.reset(graph)
    policy.reset()
    while not env.done:
        env.step(policy.act(env))
    return env

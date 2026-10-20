"""RL environment for traversal-and-invocation over a SentenceGraph.

Actions (index): 0 = INVOKE prompt at current node, 1 = STOP, 2+j = MOVE to node j (j must be a
neighbour of the current node). Reward implements Eq.(1), R = a*S + b*Sim - g*Cedit, as a telescoping
per-step difference, so the return equals R(final) - R(source). Every edit is verifier-gated: a rejected
edit is rolled back and penalised (this is what keeps the policy from reward-hacking S)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .editstate import EditState
from .graph import N_MAX, STATIC_DIM

INVOKE, STOP, MOVE0 = 0, 1, 2
N_ACTIONS = MOVE0 + N_MAX
GLOBAL_DIM = 5
DYN_DIM = 3


@dataclass
class EnvConfig:
    alpha: float = 1.0          # weight of suitability S(y)
    beta: float = 0.5           # weight of semantic fidelity Sim(y,x)
    gamma: float = 0.01         # cost per prompt invocation (edit-cost penalty)
    move_cost: float = 0.0025   # cost per traversal move
    reject_penalty: float = 0.10
    coverage_bonus: float = 0.0  # optional shaping: bonus for an accepted edit at a node with a non-trivial prompt
    fail_final_penalty: float = 1.0
    max_steps_factor: int = 3


class SimplifyEnv:
    def __init__(self, scorer, encoder, verifier, realizer, cfg: EnvConfig | None = None):
        self.scorer, self.enc, self.ver, self.rz = scorer, encoder, verifier, realizer
        self.cfg = cfg or EnvConfig()

    # -------------------------------------------------------------------------------------
    def reset(self, graph):
        self.g = graph
        self.st = EditState(graph.toks)
        self.src = self.st.render()
        if not isinstance(graph.src_vec, dict):      # cache is per-encoder: graphs may be shared across encoders
            graph.src_vec = {}
        if id(self.enc) not in graph.src_vec:
            graph.src_vec[id(self.enc)] = self.enc.encode(self.src)
        self.src_vec = graph.src_vec[id(self.enc)]
        self.cur = 0
        self.visited = np.zeros(N_MAX, np.float32)
        self.invoked = np.zeros(N_MAX, np.float32)
        self.visited[0] = 1
        self.text = self.src
        self.S = self.scorer(self.text)
        self.sim = 1.0
        self.phi = self.cfg.alpha * self.S + self.cfg.beta * self.sim
        self.S0 = self.S
        self.n_inv = self.n_move = self.n_changed = self.n_reject = self.steps = 0
        self.done, self.trail = False, []
        self.max_steps = self.cfg.max_steps_factor * graph.n + 2
        return self.obs()

    def _score(self, text):
        return self.scorer(text), self.enc.cos(self.enc.encode(text), self.src_vec)

    @property
    def edit_cost(self):
        return self.cfg.gamma * self.n_inv + self.cfg.move_cost * self.n_move

    @property
    def final_reward(self):
        """R(y,x) of Eq.(1) for the current text."""
        c = self.cfg
        return c.alpha * self.S + c.beta * self.sim - self.edit_cost

    # -------------------------------------------------------------------------------------
    def action_mask(self):
        m = np.zeros(N_ACTIONS, bool)
        m[STOP] = True
        m[INVOKE] = self.invoked[self.cur] == 0
        nb = np.where(self.g.adj[self.cur] >= 0)[0]
        for j in nb:
            if j != self.cur:
                m[MOVE0 + j] = True
        return m

    def obs(self):
        n = self.g.n
        dyn = np.zeros((N_MAX, DYN_DIM), np.float32)
        dyn[:, 0], dyn[:, 1] = self.visited, self.invoked
        dyn[self.cur, 2] = 1.0
        adj = -np.ones((N_MAX, N_MAX), np.int64)
        adj[:n, :n] = self.g.adj
        nmask = np.zeros(N_MAX, bool)
        nmask[:n] = True
        glob = np.array([self.S, self.sim, self.steps / self.max_steps, self.n_inv / max(n, 1),
                         self.n_changed / max(n, 1)], np.float32)
        return dict(x=np.concatenate([self.g.static, dyn], 1), adj=adj, nmask=nmask, glob=glob,
                    amask=self.action_mask(), cur=np.int64(self.cur))

    # -------------------------------------------------------------------------------------
    def peek(self, node_id):
        """Gain in S if the prompt at `node_id` were invoked now (no side effects). Used by baselines."""
        snap = self.st.snapshot()
        changed = self.rz.apply(self.g, self.g.nodes[node_id], self.st)
        S = self.scorer(self.st.render()) if changed else self.S
        self.st.restore(snap)
        return S - self.S, changed

    def step(self, a):
        assert not self.done
        a, r, info = int(a), 0.0, {}
        self.steps += 1
        if not self.action_mask()[a]:          # illegal -> treated as STOP with penalty
            a, r = STOP, r - 0.05
        node = self.g.nodes[self.cur]
        if a == INVOKE:
            snap = self.st.snapshot()
            self.invoked[self.cur] = 1
            self.n_inv += 1
            r -= self.cfg.gamma
            changed = self.rz.apply(self.g, node, self.st)
            if changed:
                new_text = self.st.render()
                vr = self.ver.check(self.src, new_text, use_nli=False)
                if not vr.ok:
                    self.st.restore(snap)
                    self.n_reject += 1
                    r -= self.cfg.reject_penalty
                    info = dict(event="rejected", violations=vr.violations)
                else:
                    S, sim = self._score(new_text)
                    phi = self.cfg.alpha * S + self.cfg.beta * sim
                    r += phi - self.phi
                    self.phi, self.S, self.sim, self.text = phi, S, sim, new_text
                    self.n_changed += 1
                    r += self.cfg.coverage_bonus
                    info = dict(event="edited")
            else:
                info = dict(event="noop")
        elif a == STOP:
            self.done = True
            final = self.ver.check(self.src, self.text, use_nli=True)
            if not final.ok:
                r -= self.cfg.fail_final_penalty
                self.st = EditState(self.g.toks)
                self.text, self.S, self.sim = self.src, self.S0, 1.0
                info = dict(event="final_reject", violations=final.violations)
        else:
            j = a - MOVE0
            self.cur = j
            self.visited[j] = 1
            self.n_move += 1
            r -= self.cfg.move_cost
        if self.steps >= self.max_steps and not self.done:
            self.done, r = True, r - 0.05
        self.ret = getattr(self, 'ret', 0.0) * (self.steps > 1) + r
        self.trail.append(dict(step=self.steps, ret=round(self.ret, 4), action=("INVOKE" if a == INVOKE else "STOP" if a == STOP
                                                         else f"MOVE->{a - MOVE0}"),
                               node=node.id, kind=node.kind, prompt=node.prompt if a == INVOKE else "",
                               reward=round(r, 4), text=self.text, **info))
        return r, self.done, info


def format_trail(env, with_prompts=False):
    """Human-readable traversal log: step, node visited, action, reward delta, running return."""
    nodes = env.g.nodes
    out = [f"{'step':>4} {'at':>3} {'kind':8} {'action':12} {'event':10} {'d_reward':>9} {'return':>8}"]
    for t in env.trail:
        out.append(f"{t['step']:4d} {t['node']:3d} {t['kind']:8} {t['action']:12} {str(t.get('event','')):10} "
                   f"{t['reward']:+9.4f} {t['ret']:+8.4f}")
        if with_prompts and t["prompt"]:
            out.append(f"       prompt: {t['prompt'][:110]}")
    seen = {t['node'] for t in env.trail} | {int(a.split('->')[1]) for a in (t['action'] for t in env.trail) if a.startswith('MOVE')}
    miss = [f"{n.id}:{n.kind}" for n in nodes if n.id not in seen]
    out.append(f"never visited: {miss or 'none'} | invoked: {[i for i in range(env.g.n) if env.invoked[i]]}")
    return "\n".join(out)

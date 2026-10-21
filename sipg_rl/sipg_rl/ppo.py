"""Component 3b: PPO training of the traversal-and-invocation policy (GAE, clipped objective)."""
from __future__ import annotations
import random, time
import numpy as np
import torch
from torch.distributions import Categorical
from .env import SimplifyEnv
from .model import SIPGPolicy, collate


def make_env(parts):
    return SimplifyEnv(parts["scorer"], parts["encoder"], parts["verifier"], parts["realizer"], parts.get("cfg"))


class PPOTrainer:
    def __init__(self, policy: SIPGPolicy, parts: dict, lr=3e-4, clip=0.2, vf_coef=0.5, ent_coef=0.02,
                 gamma=0.99, lam=0.95, epochs=4, minibatch=128, seed=0):
        self.pi, self.parts = policy, parts
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip, self.vf, self.ent, self.gamma, self.lam = clip, vf_coef, ent_coef, gamma, lam
        self.epochs, self.mb = epochs, minibatch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    @torch.no_grad()
    def rollout(self, graphs, n_episodes):
        envs = [make_env(self.parts) for _ in range(n_episodes)]
        gs = [random.choice(graphs) for _ in range(n_episodes)]
        obs = [e.reset(g) for e, g in zip(envs, gs)]
        traj = [[] for _ in envs]
        active = list(range(n_episodes))
        while active:
            logits, value = self.pi(collate([obs[i] for i in active]))
            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
            for k, i in enumerate(active):
                r, done, _ = envs[i].step(int(act[k]))
                traj[i].append((obs[i], int(act[k]), float(logp[k]), float(value[k]), r, done))
                if not done:
                    obs[i] = envs[i].obs()
            active = [i for i in active if not envs[i].done]
        return envs, traj

    def _gae(self, tr):
        adv, last = np.zeros(len(tr), np.float32), 0.0
        for t in reversed(range(len(tr))):
            _, _, _, v, r, done = tr[t]
            nv = 0.0 if t == len(tr) - 1 else tr[t + 1][3]
            delta = r + self.gamma * nv - v
            last = delta + self.gamma * self.lam * last
            adv[t] = last
        ret = adv + np.array([x[3] for x in tr], np.float32)
        return adv, ret

    def update(self, traj):
        O, A, LP, ADV, RET = [], [], [], [], []
        for tr in traj:
            adv, ret = self._gae(tr)
            O += [x[0] for x in tr]; A += [x[1] for x in tr]; LP += [x[2] for x in tr]
            ADV += list(adv); RET += list(ret)
        A, LP = torch.tensor(A), torch.tensor(LP, dtype=torch.float32)
        ADV, RET = torch.tensor(ADV), torch.tensor(RET)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n, stats = len(O), []
        for _ in range(self.epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, self.mb):
                idx = perm[s:s + self.mb]
                logits, v = self.pi(collate([O[i] for i in idx]))
                dist = Categorical(logits=logits)
                lp = dist.log_prob(A[idx])
                ratio = torch.exp(lp - LP[idx])
                l1 = ratio * ADV[idx]
                l2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[idx]
                pl = -torch.min(l1, l2).mean()
                vl = (v - RET[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pl + self.vf * vl - self.ent * ent
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
                self.opt.step()
                stats.append((pl.item(), vl.item(), ent.item()))
        return np.mean(stats, 0)

    def train(self, graphs, iters=60, episodes=32, log=print):
        t0 = time.time()
        for it in range(1, iters + 1):
            envs, traj = self.rollout(graphs, episodes)
            pl, vl, ent = self.update(traj)
            if it == 1 or it % 5 == 0 or it == iters:
                R = np.mean([e.final_reward - (e.cfg.alpha * e.S0 + e.cfg.beta) for e in envs])
                log(f"it {it:3d} | dR {R:+.4f} | S {np.mean([e.S for e in envs]):.3f} | "
                    f"sim {np.mean([e.sim for e in envs]):.3f} | inv {np.mean([e.n_inv for e in envs]):.1f} "
                    f"mv {np.mean([e.n_move for e in envs]):.1f} rej {np.mean([e.n_reject for e in envs]):.2f} "
                    f"| ent {ent:.2f} | {time.time() - t0:.0f}s")

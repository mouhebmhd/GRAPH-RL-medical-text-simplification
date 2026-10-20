"""Item 4 - reward sanity checks. Each block prints evidence for/against a specific failure mode."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sipg_rl.pipeline import default_parts
from sipg_rl.env import SimplifyEnv, INVOKE, STOP, MOVE0, format_trail
from sipg_rl.baselines import RandomPolicy, InvokeAllPolicy, NoOpPolicy, run_episode
from sipg_rl.data import synth

P = default_parts(); sc, enc, ver, cfg = P["scorer"], P["encoder"], P["verifier"], P["cfg"]
mk = lambda: SimplifyEnv(sc, enc, ver, P["realizer"], cfg)
graphs = [P["builder"].build_text(s) for s in synth(150, 1)]
R = lambda S, sim: cfg.alpha * S + cfg.beta * sim

print("A. Telescoping identity: sum of step rewards == R(final) - R(source) (clean episodes)")
rng, errs = RandomPolicy(0), []
for g in graphs:
    e = mk(); e.reset(g); tot = 0.0
    while not e.done:
        tot += e.step(rng.act(e))[0]
    if e.n_reject == 0 and e.steps < e.max_steps and e.trail[-1].get("event") != "final_reject":
        errs.append(abs(tot - (e.final_reward - R(e.S0, 1.0))))
print(f"   {len(errs)} episodes, max |error| = {max(errs):.2e}  -> {'OK' if max(errs) < 1e-6 else 'BROKEN'}\n")

print("B. Saturation of S (flat reward = no gradient)")
for pol in (NoOpPolicy(), InvokeAllPolicy()):
    S = [run_episode(mk(), g, pol).S for g in graphs]
    print(f"   {pol.name:16s} mean S {np.mean(S):.3f} | share S>=0.95: {np.mean(np.array(S) >= .95)*100:.0f}% | share S>=0.999: {np.mean(np.array(S) >= .999)*100:.0f}%")

print("\nC. Reward hacking: can degenerate text out-score a faithful rewrite?")
src = "The patient, who takes lisinopril 10 mg BID, reports no chest pain."
good = "The patient takes lisinopril 10 mg twice a day. The patient reports no chest pain."
cands = {"source": src, "good rewrite": good, "empty": "", "'No.'": "No.", "generic": "Take the medicine.",
         "dose dropped": "The patient takes lisinopril twice a day. The patient reports no chest pain.",
         "negation flipped": "The patient takes lisinopril 10 mg twice a day. The patient reports chest pain.",
         "dose x10": "The patient takes lisinopril 100 mg twice a day. The patient reports no chest pain.",
         "drug swapped": "The patient takes metformin 10 mg twice a day. The patient reports no chest pain."}
v0 = enc.encode(src)
print(f"   {'candidate':18s} {'S':>6s} {'Sim':>6s} {'R(a=1,b=.5)':>11s} verifier")
for k, t in cands.items():
    S, sim = sc(t), enc.cos(enc.encode(t), v0)
    r = ver.check(src, t, use_nli=False)
    print(f"   {k:18s} {S:6.3f} {sim:6.3f} {R(S, sim):11.3f} {'ok' if r.ok else 'REJECT: ' + r.violations[0]}")

print("\nD. Order dependence: relative-clause split before vs after rewriting the jargon adjective")
g = P["builder"].build_text("The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain.")
byk = {n.kind: n.id for n in g.nodes}
for label, order in (("RELCL only", ["RELCL"]), ("AMOD then RELCL", ["AMOD", "RELCL"]), ("RELCL then AMOD", ["RELCL", "AMOD"])):
    e = mk(); e.reset(g); tot = 0
    for k in order:
        e.cur = byk[k]; tot += e.step(INVOKE)[0]
    print(f"   {label:18s} total reward {tot:+.3f} | S {e.S:.3f} sim {e.sim:.3f} | {e.text}")

print("\nE. Cost vs gain scale (is gamma doing any work?)")
gains, costs = [], []
for g in graphs:
    e = mk(); e.reset(g)
    gains += [e.peek(j)[0] * cfg.alpha for j in range(g.n) if e.peek(j)[1]]
print(f"   median positive S-gain per edit {np.median([x for x in gains if x>0]):.3f} vs gamma {cfg.gamma} "
      f"(ratio {np.median([x for x in gains if x>0])/cfg.gamma:.0f}x) -> cost only breaks ties unless gain < gamma")
print(f"   share of changing edits that LOWER S immediately: {np.mean(np.array(gains) < 0)*100:.0f}%")

print("\nF. Does sentence splitting alone inflate S? (same content, different layout)")
a = "The patient takes lisinopril and reports no chest pain."
b = "The patient takes lisinopril. The patient reports no chest pain."
print(f"   joined S={sc(a):.3f} | split S={sc(b):.3f} | Sim(joined,split)={enc.cos(enc.encode(a), enc.encode(b)):.3f}")

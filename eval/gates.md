# Instrument validation (G1-G4)

prereg sha256: `2341d6fef7659211`  n=150  scorer: bands fitted on 150 reference texts from train

| system | Sim(out, source) | S (fitted band) | S + 0.5*Sim |
|---|---|---|---|
| copy | 1.000 | 0.663 | 1.163 |
| reference | 0.526 | 0.763 | 1.026 |
| rules | 0.991 | 0.734 | 1.230 |
| llm | 0.872 | 0.786 | 1.222 |
| sipg | 0.891 | 0.787 | 1.232 |

## G1 Sim / reward ordering  -> **FAIL**
- literal criterion (ordering reference > rules > llm > copy on Sim, and Sim(ref) - Sim(copy) > 0.15): **fail**. NOTE: for any source-anchored fidelity score Sim(copy, source) = 1 is the maximum, so `reference > copy` on Sim alone cannot hold by construction; it is only meaningful for the composite reward below.
- (i) gold is not penalised for being a rewrite: Sim(copy) - Sim(ref) = 0.474 <= 0.15: **fail**
- (ii) Sim discriminates unsafe edits (mean drop on the reference when corrupted): drop_number 0.423 (n=26), drop_negation 0.149 (n=17), drop_hedge 0.254 (n=32), swap_drug n<10 (need >= 0.15): **fail**
- (iii) composite reward S + 0.5*Sim: reference - copy = -0.137 [-0.173, -0.099] (need > 0.15, CI > 0): **fail**; ordering ref > rules > llm > copy on composite: **fail** (informational)

## G2 S calibrated to gold  -> **PASS**
- S(llm) - S(reference) = +0.023; S(sipg) - S(reference) = +0.023; limit +0.03. |S(ref) - S(llm)| = 0.023. bands fitted on 150 reference texts from train

## G3 verifier vs gold  -> **PENDING (label audit_gold.csv -> audit_gold_labeled.csv with label a/b/c, then re-run)**
- the verifier rejects 29/150 references = 19.3% (95% CI 13.8-26.4); by violation: {'hedge dropped': 10, 'negation cue dropped': 8, 'hallucinated number': 9, 'conditional dropped': 6, 'number/dose lost or altered': 28}
- auto-triage: 7 rejections are *likely* false positives (format/paraphrase) - a heuristic, not a label. Classify eval\audit_gold.csv manually as (a) false positive / (b) acceptable change / (c) annotator error.

## G4 fallback executes  -> **PASS**
- real data (sipg): final fallback 3.3% of examples; rejected edits per example 0.00
- adversarial injection (NLI off): drop_number: injected 9, edit-gate 9, final-gate 0, leaked 0; drop_negation: injected 9, edit-gate 9, final-gate 0, leaked 0; swap_drug: injected 0, edit-gate 0, final-gate 0, leaked 0; hallucinate_number: injected 10, edit-gate 10, final-gate 0, leaked 0; antonym_drift: injected 10, edit-gate 0, final-gate 0, leaked 9; drop_hedge: injected 16, edit-gate 16, final-gate 0, leaked 0
- adversarial injection (NLI on): drop_number: injected 9, edit-gate 9, final-gate 0, leaked 0; drop_negation: injected 9, edit-gate 9, final-gate 0, leaked 0; swap_drug: injected 0, edit-gate 0, final-gate 0, leaked 0; hallucinate_number: injected 10, edit-gate 10, final-gate 0, leaked 0; antonym_drift: injected 10, edit-gate 0, final-gate 9, leaked 1; drop_hedge: injected 16, edit-gate 16, final-gate 0, leaked 0
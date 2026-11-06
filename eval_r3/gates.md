# Instrument validation (G1-G4)

prereg sha256: `2341d6fef7659211`  n=150  scorer: bands fitted on 150 reference texts from train | Sim = content

| system | Sim(out, source) | S (fitted band) | S + 0.5*Sim |
|---|---|---|---|
| copy | 1.000 | 0.638 | 1.138 |
| reference | 0.526 | 0.741 | 1.005 |
| rules | 0.992 | 0.726 | 1.221 |
| llm | 0.872 | 0.786 | 1.222 |
| sipg | 0.913 | 0.775 | 1.231 |

## G1 Sim / reward ordering  -> **FAIL**
- literal criterion (ordering reference > rules > llm > copy on Sim, and Sim(ref) - Sim(copy) > 0.15): **fail**. NOTE: for any source-anchored fidelity score Sim(copy, source) = 1 is the maximum, so `reference > copy` on Sim alone cannot hold by construction; it is only meaningful for the composite reward below.
- (i) gold is not penalised for being a rewrite: Sim(copy) - Sim(ref) = 0.474 <= 0.15: **fail**
- (ii) Sim discriminates unsafe edits (mean drop on the reference when corrupted): drop_number 0.423 (n=26), drop_negation 0.148 (n=17), drop_hedge 0.254 (n=32), swap_drug n<10 (need >= 0.15): **fail**
- (iii) composite reward S + 0.5*Sim: reference - copy = -0.133 [-0.170, -0.095] (need > 0.15, CI > 0): **fail**; ordering ref > rules > llm > copy on composite: **fail** (informational)
- (iv) S ALONE (the training objective): S(reference) - S(copy) = +0.104 [+0.071, +0.141] (need > 0 with the CI excluding 0): **pass**

## G2 S calibrated to gold  -> **FAIL**
- S(llm) - S(reference) = +0.045; S(sipg) - S(reference) = +0.034; limit +0.03. |S(ref) - S(llm)| = 0.045. bands fitted on 150 reference texts from train | Sim = content

## G3 verifier vs gold  -> **PENDING (label audit_gold.csv -> audit_gold_labeled.csv with label a/b/c, then re-run)**
- the verifier rejects 29/150 references = 19.3% (95% CI 13.8-26.4); by violation: {'hedge dropped': 10, 'negation cue dropped': 8, 'hallucinated number': 9, 'conditional dropped': 6, 'number/dose lost or altered': 27}
- number normalisation: legacy check rejected 29/150 = 19.3%; after normalisation 29 = 19.3%. 0 rows were formatting artefacts (causes: {}); see audit_gold_format_artefacts.csv
- auto-triage: 7 rejections are *likely* false positives (format/paraphrase) - a heuristic, not a label. Classify eval_r3\audit_gold.csv manually as (a) false positive / (b) acceptable change / (c) annotator error.

## G4 fallback executes  -> **PASS**
- real data (sipg): final fallback 4.0% of examples; rejected edits per example 0.01
- adversarial injection (NLI off): drop_number: injected 9, edit-gate 8, final-gate 0, leaked 2; drop_negation: injected 9, edit-gate 9, final-gate 0, leaked 0; swap_drug: injected 9, edit-gate 9, final-gate 0, leaked 0; hallucinate_number: injected 10, edit-gate 10, final-gate 0, leaked 0; antonym_drift: injected 10, edit-gate 0, final-gate 0, leaked 9; drop_hedge: injected 16, edit-gate 16, final-gate 0, leaked 0
- adversarial injection (NLI on): drop_number: injected 9, edit-gate 8, final-gate 0, leaked 2; drop_negation: injected 9, edit-gate 9, final-gate 0, leaked 0; swap_drug: injected 9, edit-gate 9, final-gate 0, leaked 0; hallucinate_number: injected 10, edit-gate 10, final-gate 0, leaked 0; antonym_drift: injected 10, edit-gate 0, final-gate 9, leaked 1; drop_hedge: injected 16, edit-gate 16, final-gate 0, leaked 0
- swap_drug requirement (injected >= 9, caught >= 9, leaked = 0): injected 9 (9 synthetic top-up), caught 9, leaked 0: **PASS**
- leaked cases are written with their NLI probabilities to eval_r3\adversarial_leaks.md (14 total)
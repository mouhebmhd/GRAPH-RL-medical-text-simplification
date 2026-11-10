# Round-3 audit changes (15 items)

| # | Change | Status |
|---|---|---|
| 1 | `EnvConfig.beta = 0` (S alone); `--beta` in train/pilot; checkpoints record beta, legacy ones warn; post-hoc `SIPGRL(sim_floor=...)`; G1 (iv) S(ref) - S(copy) with CI | done, tested |
| 2 | `fidelity.py`: `NLISim` (bidirectional entailment), `BERTScoreSim`; `--sim content|nli|bertscore`. `reference > rules > llm > copy` on a source-anchored Sim is impossible by construction (copy == source is the maximum) | NLISim tested with stub; BERTScore untested |
| 3 | `numnorm.py` + verifier. The old regex already tolerated `50mg`/`50 mg`; real causes were spelled-out units, hyphenated durations, thousands separators, spelled numbers, `.5`, fractions, "twice a day". `FidelityVerifier(legacy_numbers=True)` reproduces the old check | done, tested; gold rate must be re-measured on Med-EASi |
| 4 | `audit_gold` writes audit_gold.csv (still rejected, with `suggested`), audit_gold_format_artefacts.csv, gold_audit.json; `audit` stage reports (a)/(b)/(c) counts and corrected rate | tooling done; labelling is manual |
| 5 | `default_nli()` ON by default (`SIPG_NLI=off` to disable), eval_suite `--nli` default hf; edit gate lexical, final gate NLI (env STOP + pipeline) | done; residual antonym leak must be inspected with the real model (adversarial_leaks.md) |
| 6 | swap_drug covers 53 drugs, synthetic top-up so injected >= 9, leaks logged with NLI probabilities, requirement line in gates.md | done, tested |
| 7 | `mining.py`, `scripts/mine_lexicon.py` (train split only), loader in `lexicon.py` (hand entries win, safety-gated) | mechanics tested; real gain unmeasured |
| 8/9 | LLM: prompt + sanitiser + guard reject compression; LLM_KINDS lexical only; hybrid no longer rewrites clauses unless `--llm-structure`. Rules own clause splits (item 8 reading) | done; 72% deletion must be re-measured with Ollama |
| 10 | `scripts/lexicon_recall.py` (top-1/3/5, per node type, 30% verdict) | mechanics tested |
| 11 | gamma is 0.03 here (0.0025 is `move_cost`) - unchanged | n/a |
| 12 | Band scorer is the default everywhere; old monotone scorer saturates at 66% >= 0.95, band 31% (synthetic) | measured, no new rescale |
| 13 | `runs/v3/policy_seed0-4.pt` trained with beta=0 on SYNTHETIC data (smoke checkpoints) - retrain on Med-EASi; `--ckpts` defaults to the 5 seeds | retrain needed |
| 14 | `SIPG_NMAX` env var; `scripts/graph_stats.py` (clip % per cap) | mechanics tested; run on Med-EASi |
| 15 | `scripts/verify_cache.py`: mechanism works; key = (prompt, span, full sentence) so 0 hits on one pass is expected | run `--file runs/llm_cache.jsonl` |

Retrain on real data:  python scripts/train_policy.py --corpus <file> --seeds 0,1,2,3,4 --iters 200 --out runs/v3

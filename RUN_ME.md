# RUN_ME - the 30 human-free tasks: where each one is and how to run it

Setup (once):  `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
**One command for everything offline:**  `python scripts/run_all_offline.py --out eval`   (add `--jsonl your.jsonl` to use your own sentences)
**Reproducibility check:** `./reproduce.sh` (runs it twice, compares hashes, writes `eval/manifest.json`).
**Tests (CI gate):** `python -m pytest tests -q`  -> 107 pass.

Default data is synthetic template sentences, so results are MACHINERY checks. For real claims pass `--jsonl` with your Med-EASi-style file
(keys `Expert`/`Simple`, optional `expert_terms`/`layman_terms`). Nothing below needs a human; steps marked (model) need a model you download.

| # | task | file | run |
|---|---|---|---|
| 1 | Unguarded-vs-gated ablation (Table 2) | `scripts/ablate_verifier.py` | `python scripts/ablate_verifier.py` (add `--strong llm --llm-model ...` for a real LLM) -> `eval/table2.{json,md}` |
| 2 | Per-category checks | `sipg_rl/checks.py` | `from sipg_rl.checks import check_all; check_all(src, out)` |
| 3 | Verifier off-switch | `FidelityVerifier(enabled=False)`, `default_parts(verifier_enabled=False)` | used by 1, 5 |
| 4 | Weak realizer | `CopyLexiconRealizer` in `sipg_rl/realize.py` | used by 5 |
| 5 | 2x2 runner | `scripts/run_2x2.py` | `python scripts/run_2x2.py` -> `eval/2x2.json` |
| 6 | NLI on/off, antonym drift, edit vs final gate | `scripts/ablate_nli.py` | `python scripts/ablate_nli.py` (offline row); add `--nli microsoft/deberta-large-mnli` (model) |
| 7 | Drop `Sim` from reward | `sipg_rl/reward.py`, `EnvConfig` | default `["S"]`; `Sim` still logged |
| 8 | Number/unit normalisation | `sipg_rl/normalize.py` | wired into verifier + metrics |
| 9 | `swap_drug` injector fix | `sipg_rl/inject.py` | `python scripts/run_injection.py --base synth` |
| 10 | BERTScore / NLI columns | `sipg_rl/metrics.py`, `fidelity.py` | `bertscore` (offline soft fallback; `--fidelity bertscore` for the real one), `nli_entailment` (model) |
| 11 | S-ranked candidate selection | `sipg_rl/select.py` | used by `RuleRealizer` when a key has several candidates |
| 12-15 | Diagnostics (node-type layman hit, per-category preservation, edit types, target band) | `sipg_rl/metrics.py` | printed by `scripts/run_eval.py` |
| 16 | Mined-lexicon recall by node type | `scripts/lexicon_recall.py`, `mine_lexicon.py` | `python scripts/lexicon_recall.py --train train.jsonl --test test.jsonl` (or `--cv 5`) |
| 17 | Natural-drift rate | `scripts/natural_drift.py` | `python scripts/natural_drift.py --nli microsoft/deberta-large-mnli` (model; skipped without) |
| 18 | Holm-Bonferroni | `sipg_rl/stats.py::holm_bonferroni` | applied in `run_eval.py` |
| 19 | Paired bootstrap by example | `stats.py::paired_bootstrap` | pinned by `tests/test_modules_round4.py` |
| 20 | Multi-seed runner | `scripts/run_multiseed.py` | `python scripts/run_multiseed.py --seeds 0,1,2,3,4 --train --jsonl test.jsonl` |
| 21 | Per-category CIs | `scripts/run_eval.py` | sub-tables now print `rate [lo,hi] (n)` |
| 22 | Manifest + reproduce.sh | `scripts/reproduce.py`, `reproduce.sh` | `./reproduce.sh` |
| 23 | Cache verification | `sipg_rl/llm.py` (persistent + per-span lexical cache), `scripts/check_cache.py` | `python scripts/check_cache.py`; real runs: `--llm-cache eval/llm_cache.jsonl` |
| 24 | Audit-trail export | `sipg_rl/trail.py` | `eval_out/trail.jsonl` from `run_eval.py` |
| 25 | Graph truncation frequency | `scripts/measure_truncation.py` | `python scripts/measure_truncation.py --jsonl your.jsonl` |
| 26 | Six CI tests | `tests/` | `python -m pytest tests -q` |
| 27 | Rule coverage from mined lexicon | `sipg_rl/mined.py` | `python scripts/mine_lexicon.py --jsonl train.jsonl` then `SIPG_MINED_LEXICON=lexicon_mined.json python ...` |
| 28 | LLM span-only constraint + sanitizer | `sipg_rl/sanitize.py`, `HybridRealizer`, `llm.py` modes | automatic in hybrid mode |
| 29 | Hybrid router | `HybridRealizer` | `--systems hybrid,sipg` in `run_eval.py` |
| 30 | Greedy / invoke_all / random baselines | `sipg_rl/baselines.py` | `run_eval.py --systems copy,random,invoke_all,greedy,rules,hybrid,sipg` |

Full evaluation on your data (all systems, CIs, Holm, seeds, audit trail):
`python scripts/run_multiseed.py --seeds 0,1,2,3,4 --train --train-jsonl train.jsonl --jsonl test.jsonl --limit 0 --compare-to greedy`

## Caveats (read before quoting numbers)
* The offline "strong realizer" in Table 2 / the 2x2 is a STAND-IN that injects hazards; it measures the gate, not an LLM. Use `--strong llm` for the real arm.
* In the stand-in, an injected hazard and the legitimate edit share one step, so a rejection discards both: strong+verifier has a lower S gain (0.224) than weak (0.396). That is an artefact of the stand-in.
* Offline fidelity (`SoftTokenFidelity`) is weak on open paraphrase; use BERTScore/NLI for real fidelity claims.
* Truncation, lexicon recall and natural drift need YOUR text/model; on synthetic data they only prove the scripts run.

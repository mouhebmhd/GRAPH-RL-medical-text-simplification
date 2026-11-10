# SIPG-RL — reference implementation

Syntax-Induced Prompt Graph + PPO traversal policy + post-hoc fidelity verifier, following the proposal.

```
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python -m pytest tests -q                                  # 7 tests
python scripts/pilot.py --n 450 --iters 120                # Phase-1 pilot + kill-criterion check (~1 min CPU)
python scripts/demo.py "The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain."
```

## Round 3 (P0-P2 task list) - what was done, what was verified, what you must re-run on Med-EASi

**Environment note:** the build sandbox was reset mid-session; this round was rebuilt on top of your `sipg_rl_eval.zip`. Med-EASi is not
reachable from the sandbox, so every number below comes from the 24-pair hand-written toy fixture (`tests/fixtures`, NOT Med-EASi) or
synthetic template sentences, and tests that guard Med-EASi behaviour run on the toy fixture unless you set `SIPG_MEDEASI_JSONL=/path/test.jsonl`.

| task | status | where |
|---|---|---|
| P0.1 drop `Sim` from reward | done: default potential = `["S"]`; `Sim` still computed+logged; `band`, `fid_hinge`, legacy `v1` selectable | `sipg_rl/reward.py`, `env.py` (`EnvConfig.reward_version`) |
| P0.2 number/unit normalisation | done: `50mg = 50 mg = 50 milligrams`, `4,000 = 4000`; real changes (`5 mg -> 50 mg`, `mg -> mcg`) still caught | `sipg_rl/normalize.py`, `verifier.py`, `metrics.numbers_kept` |
| P0.3 injectors fire | done: raise on empty vocabulary / zero injections. Bug found in MY injector (decimal x10 appended a zero) and fixed | `sipg_rl/inject.py`, `scripts/run_injection.py` |
| P1.1 baselines + hybrid | done: copy/random/invoke_all/greedy/rules/llm/hybrid/sipg(/reference); `HybridRealizer` router | `scripts/run_eval.py`, `realize.py` |
| P1.2 diagnostics | done (8 metrics, pure functions) | `metrics.py` |
| P1.3 CIs + Holm | done | `stats.py`, `run_eval.py` |
| P1.4 multi-seed | done: 5 checkpoints trained (synthetic) + across-seed table | `train_seeds.py`, `run_eval.py --seeds` |
| P2.1 risk-coverage + ECE | code done; the demo uses a 75-pair hand-written suite and shows the OFFLINE signal is unusable (AUROC 0.45) | `selective.py`, `scripts/risk_coverage.py` |
| P2.2 audit trail | done: per (system, example) JSONL with candidate_used/rank/count, realizer, verifier, S_before/after | `trail.py`, `run_eval.py` -> `trail.jsonl` |
| P2.3 ablations | done: verifier, nli (plumbing only offline), graph, reward | `scripts/run_ablations.py` |

### Run the acceptance checks on Med-EASi
```
SIPG_MEDEASI_JSONL=medeasi_test.jsonl pytest tests/test_reward_ordering.py -s     # P0.1 GATE: S(ref) > S(copy), CI excludes 0. STOP if it fails.
python scripts/run_injection.py --jsonl medeasi_test.jsonl --base source --n 500  # P0.3 / G4: swap_drug injected >= 9, caught == injected
python scripts/train_seeds.py --jsonl medeasi_train.jsonl --seeds 0,1,2,3,4 --out-pattern sipg_policy_v3_seed{seed}.pt
python scripts/run_eval.py --jsonl medeasi_test.jsonl --limit 0 --seeds 0,1,2,3,4 --ckpt-pattern "sipg_policy_v3_seed{seed}.pt" \
       --fidelity bertscore --nli microsoft/deberta-large-mnli --llm-model qwen2.5:7b-instruct --compare-to greedy --out eval_out
python scripts/run_ablations.py --ablate all --nli microsoft/deberta-large-mnli --seeds 0,1,2
```
Re-run your G3 (gold rejections) after P0.2: if the 28 number/dose rejections do not drop, they are real and the verifier is stricter than the annotators.

### Measured here (toy / synthetic only: machinery evidence, not scientific claims)
* Injection harness, 300 synthetic sentences, lexical gate: swap_drug 273/273 caught, dose x10 228/228, drop_negation 211/211, drop_hedge 11/11, drop_conditional 15/15, **antonym 0/90 (known blind spot, leaks 100%)**.
* Verifier ablation (injected unsafe edits, rate 0.5): full verifier 204 injected / 0 leaked / 0.0% unsafe outputs; ablated 202 leaked / 77.2% unsafe; negation preserved 58.4%, conditionals 23.1%.
* Five seeds, S-only reward, synthetic: PPO dR 0.368/0.392/0.386/0.390/0.376 (mean 0.382, std 0.010) vs greedy 0.388 -> **tie with greedy is inside seed noise; edits 3.0 vs 3.0, so "same S with fewer edits" is NOT shown.**
* Graph ablation (3 seeds): flat BiGRU R 0.375 +- 0.004 vs GAT 0.382 +- 0.012 -> **the graph adds nothing on template data** (expected: independent edits). Must be re-run on Med-EASi before any graph claim.
* Reward ablation (3 seeds): legacy `S + 0.5 Sim` -> S 0.838 +- 0.078 vs 0.918 +- 0.011, edits 2.3 vs 2.9: confirms the failure mode.
* Offline soft fidelity: gold-vs-source F1 0.515 vs copy 1.000 on the toy fixture, and AUROC 0.45 as a confidence signal. Use BERTScore/NLI on real data.

### Findings worth your attention
1. **NLI never saw drift in sentences without a number or drug name**: the old gate ran NLI only when the source had critical spans. `FidelityVerifier(nli_scope="all")` is now the default (`"critical"` = old behaviour; `nli_mode="bidirectional"` available). Any earlier NLI-on recall numbers measured through that gate understate drift misses.
2. **C4 (`hypertensive urgency`)**: the rule lexicon HAS the phrase and applies it. A trace that shows "condition node reached, no-op" is the enclosing `if ...` clause (deliberately never split), not the term node. If the phrase were missing, rules would no-op on the AMOD node (reproduced in `tests/test_regression_S1.py`) and the hybrid's constrained LLM fallback rescues it.
3. **`cache_hits = 0`**: the old LLM cache key included the whole sentence + prompt. Lexical requests are now cached per span text (`OpenAICompatLLM.lexical_context_free`).
4. Fallback% = 0 is structural while NLI is off: the edit gate and the final gate are the same lexical check. Report `rejected` (edit rollbacks) alongside `fallback%`.
5. Under the S-only reward the policy no longer splits clauses on its own (the coverage bonus was doing that); `apply_all` still does.

### Not done / lost in the reset (say if you want them rebuilt)
A3 gold-audit triage script, lexicon miner + blocklist audit + coverage (D1/D3), clinician annotation sheets + kappa (E2), graph truncation handling (F1), glossary tier restriction/audit sheet (F2),
`train_seeds.py --realizer hybrid` (LLM in the loop, B4a) is implemented but untested against a real model. Until then, LLM/hybrid mode is an **inference-time augmentation**: the policy was trained with the rule realizer.
Paper sentence: *safety is measured by deterministic gates and injected hazards, not clinician adjudication; verifier recall is bounded by the injection taxonomy and is an upper bound on true recall.*


<<<<<<< HEAD
## Health Literacy and Patient Profile module (new)

**Invariant.** The patient profile conditions the **Generation layer** (LLM prompts) and the **Evaluation scorer** (target FKGL band,
RVS/RTS targets). It is **never** passed to `FidelityVerifier`, which stays profile-blind, so personalisation cannot bypass a safety
check. A text that misses profile targets is *flagged for revision* (`needs_revision`), never safety-rejected. `tests/test_patient_profile.py`
enforces this (verifier signature + source contain no "profile"; verdict depends only on `(source, output)`).

| piece | file | notes |
|---|---|---|
| Profile schema (core proficiency, coherence need, spiritual-first, digital vs functional, adaptability, topic variance) | `sipg_rl/patient_profile.py` | continuous [0,1] dimensions; archetype derived: Balanced / Specialized / Transitional |
| RVS / RTS / FKGL | `sipg_rl/coherence.py` | RVS = std of per-sentence FKGL; RTS = mean abs change between adjacent sentences. **My operational definitions** - edit `rvs`/`rts` if your paper defines them differently |
| Profile -> prompts | `build_system_prompt` (document level), `span_style_suffix` (span level) | 4th-6th grade, steady flow, transitional scaffold, spiritual-first order, digital/functional formatting, safety footer |
| Profile -> scorer | `ProfileSuitabilityScorer` | `S = (1-w)*base + w*band(FKGL)`; works with any base scorer |
| Report | `profile_report(text, profile)` | FKGL in band, RVS/RTS vs caps, `needs_revision` + reasons |
| Pipeline | `SIPGRL.simplify(text, profile=..., revise=False)` | `revise=True`: one document-level LLM rewrite, kept ONLY if the profile-blind verifier accepts it against the original source |
| LLM | `OpenAICompatLLM.style_suffix`, `.rewrite_document()` | cache keys include the suffix (old cache files stay valid) |
| Dashboard | `dashboard/server.py`, `index.html` | POST `/api/simplify {"profile": {...}, "revise": true}`; response adds `profile_report`, `profile_revision` |

```
python scripts/demo.py "The patient takes lisinopril 10 mg BID ..." --profile low_literacy
python scripts/demo.py "..." --profile spiritual_first --revise --llm-model qwen2.5:7b-instruct
python scripts/demo.py "..." --profile '{"core_proficiency":0.2,"coherence_need":"steady"}'
python scripts/profile_eval.py --jsonl medeasi_test.jsonl --src-field Expert --limit 100 --out eval_profile
python -m pytest tests/test_patient_profile.py -q
```
Presets: `balanced, low_literacy, specialized_digital, specialized_functional, transitional, spiritual_first`.

**Defaults to calibrate, not validated constants:** grade bands (low 4-6 per the module spec; mid 6-8 and high 8-10 are my assumption),
steady-flow caps (RVS <= 3.0, RTS <= 3.5), archetype thresholds (`TRANSITIONAL_VARIANCE=0.6`, `SPECIALIZED_GAP=0.3`). Fit the caps from
reference simplifications with `coherence.calibrate_coherence(refs, 0.75)`, the same way the S bands are fitted.
**Not done:** learning the profile from patient interaction (you supply the profile); retraining the PPO policy with the profile in the
state (the policy is unchanged - the profile only shifts S at inference and the LLM prompts); RVS/RTS are document-level post-hoc checks,
not part of the per-sentence RL reward.

**Merged mode (no switches).** The dashboard and `scripts/demo.py` / `scripts/profile_eval.py` now always apply ALL operations (clause splits,
jargon, adjective+noun phrases, long words) with rules first and the LLM as constrained fallback when reachable (rules only otherwise), and try the
profile revision when an LLM is in use. Every edit and the revision still pass the profile-blind verifier. `demo.py --selective` restores the
policy-decides behaviour; `SIPGRL.simplify()` keeps `apply_all=False` as its default so evaluation scripts are unchanged.
Limit: still a single pass (output is not re-parsed), so clauses created by a split are not split again.

=======
>>>>>>> 1bfb6c5672b8225d0d769783c21f87e9190e7f5b
## Paper component -> code
| Proposal | Module | Notes |
|---|---|---|
| 3.1 Syntax-induced graph | `graph.py` | spaCy dependencies -> clause nodes (ROOT/RELCL/CONJ_CL/ADVCL/CCOMP/ACL) + term nodes (LEX/NOM/AMOD/CPLX). Edges: syn_down/up, cue-based discourse (contrast, elaboration, condition, cause, sequence), next/prev, self. No KB/AMR. |
| 3.2 Local prompt instantiation | `prompts.py` | One prompt per node from role + local context (head, relation, conjunction). `LLMPromptInstantiator` adapter. |
| (executing a prompt) | `realize.py`, `editstate.py` | `RuleRealizer` = deterministic offline stand-in for the instruction model; `CallableRealizer(llm)` plugs in T5/LLM. Edits are token-id based, so prompts compose in any order and every step is reversible. |
| 3.3 GAT encoder | `model.py` | Dense multi-head GAT with learned per-edge-type attention bias (no PyG needed). Pointer head for MOVE, INVOKE/STOP heads, value head. |
| 3.3 PPO policy | `ppo.py`, `env.py` | Actions: INVOKE / MOVE(neighbour) / STOP with action masking. Reward = Eq.(1) as telescoping per-step difference: `a*S + b*Sim - g*Cedit`. |
| 3.4 Fidelity verifier | `verifier.py` | Number+unit gate, negation gate, critical-term gate with approved substitutions (BID -> "twice a day"), no hallucinated numbers/drugs, optional NLI (`HFNLI`). Applied per edit (rollback + penalty) and on final output (fallback to source). |
| S(y) Suitability Scorer | `scoring.py` | `FeatureSuitabilityScorer` = the "operationalise as feature-based scorer" fallback from Table 2. Replace with the real scorer: any `callable(text)->[0,1]`. |
| Sim(y,x) | `scoring.py` | `ConceptEncoder` offline stand-in for a clinical sentence encoder. |
| 5 Baselines / kill criterion | `baselines.py`, `scripts/pilot.py` | no-op, random, invoke-all, greedy-readability, learned; paired bootstrap. |
| Audit trail | `pipeline.py` | `SIPGRL.simplify()` returns the per-sentence traversal (which node, which prompt, reward). |

## Dashboard (local)
```
python scripts/pilot.py --iters 120 --coverage_bonus 0.05 --ckpt sipg_policy_bonus.pt   # once, if no checkpoint is present
python dashboard/server.py --port 8000          # then open http://localhost:8000
```
Paste text, press **Simplify text**. You get: tracked-change redline (and clean text), a card per applied operation (what it did, the node
span, its own before/after diff, the instantiated prompt, reward delta), prompts that fired with no effect, edits blocked by the verifier,
the induced sentence graph with the policy's moves (numbered dotted arrows) and fired prompts (numbered badges), and the full step log.
Stdlib server (`dashboard/server.py`) + one HTML file (`dashboard/index.html`); API: `POST /api/simplify {"text": ...}`, `GET /api/health`.
Note: when a clause is split off, the word-level redline shows it as delete+insert (moved text); the per-operation cards are the precise record.

## Apply all operations (simplest output)
Dashboard: **Coverage -> Apply all operations**, or the **Apply all remaining operations** button under the operations list. After the
learned policy stops, every node it left unused is visited and its prompt fired (`LearnedThenSweep(kinds=None)`, step budget raised so
the sweep can finish). Each edit still goes through the verifier, so unsafe edits are rolled back. Steps added this way are tagged
"applied at the end" / "end sweep". API: `simplify(text, apply_all=True)` or `{"apply_all": true}`.
Measured on 60 synthetic sentences (rule rewriter): output differs from the policy's in 16, all 60 verified, readability +0.008 on
average, similarity to the source -0.053 (more rewording). It is a single pass: it does not re-parse its own output, so a clause it
creates (e.g. "who has high blood pressure") is not split further.

## Free LLM rewriter (Ollama, local)
```
# 1. install Ollama (https://ollama.com), then:
ollama pull qwen2.5:3b            # any small chat model works: llama3.2:3b, phi3.5, gemma2:2b ...
# 2. start the dashboard; it auto-detects Ollama on localhost:11434
python dashboard/server.py --port 8000            # add --llm-model llama3.2:3b to use another model
```
The page then shows a **Rewriter** switch (Rules / LLM). In LLM mode the model proposes a plain-language replacement for one span at a time
(terms, adjective+noun phrases, long words); clause splitting stays rule-based because it is structure-safe. The graph policy still
decides *where* to fire, and the verifier can still block any edit (numbers, negations, hedges, conditionals, drug names): a blocked edit
is rolled back and shown in the dashboard with the reason. LLM edits are tagged **LLM**; spans the model leaves unchanged count as no-ops.
Other backends (any OpenAI-compatible API): `--llm-url http://localhost:1234/v1` (LM Studio / llama.cpp), or a hosted free tier such as
`--llm-url https://api.groq.com/openai/v1 --llm-key $KEY --llm-model <model>`. **Hosted APIs send the text to a third party - do not use
them with real patient data**; Ollama keeps everything on your machine. Code: `sipg_rl/llm.py`, `CallableRealizer` in `realize.py`.

*Sweep:* the policy was trained where the rule rewriter did nothing on "long word" nodes, so it learned to skip them. In LLM mode,
`LearnedThenSweep` also tries term nodes the policy skipped, and the log marks each step "policy" or "sweep". Retraining the policy with
the LLM in the loop would remove the need for this and has not been done.

**Verification status:** the client, sanitising, caching, error handling, verifier gating and sweep are tested against a *stand-in*
OpenAI-compatible server (`tests/fake_llm.py`, canned answers) - not against a real model, which was unreachable in the build sandbox.
Expect real small models to sometimes ignore instructions or add words; the sanitiser and verifier are there for that, but
their hit rate on real model output is unmeasured.

## Medical-term info boxes
In the dashboard, body parts, medicines, conditions, drug classes and dosing abbreviations in the result are underlined; hover, focus
or click one to open an info box (what it is, one useful general fact, a safety tip for medicines). A **Medical terms in this text** list
under the result shows the same terms, and the **Explain medical terms** checkbox turns it all off. Code: `sipg_rl/glossary.py`,
`SimplifyResult.terms`, `POST /api/explain`.
* **Built-in glossary** (~190 surface forms): hand-written, general, no doses. Extend it without code: `SIPG_GLOSSARY=/path/extra.json`,
  a list of `{"keys": [...], "type": "body|medicine|class|condition|abbr|test|other", "definition": "...", "extra": "..."}`.
* **Known drug names** from `CRITICAL_TERMS` with no glossary text are flagged as medicines (dashed underline).
* **Word-shape guesses** (-itis, -emia, -pril, -statin ...) get a dashed underline and the label "not in the built-in glossary". The info box
  offers **Explain with AI**, which asks the local Ollama model for a short entry (cached) and labels it *AI-generated, not verified*.
* Terms are matched in the *output* text, so a term the rewriter replaced (e.g. "dyspnea" -> "trouble breathing") is not shown; the plain
  wording that remains is (e.g. "high blood pressure", "kidney"). In the tracked-changes view a multi-word term that straddles an
  edit boundary is not underlined; the clean view always is.
* Default dashboard model is now `qwen2.5:7b-instruct`.
* Tests: `tests/test_glossary.py`.

## What is NOT here (needs network/data that the build sandbox lacked)
* No pretrained sentence encoder or NLI model, and the LLM backend has only been tested against a stand-in server (see above). Defaults are rule-based; adapters: `CallableRealizer`/`OpenAICompatLLM`, `HFNLI`, any `encoder` with `.encode/.cos`.
* `lexicon.py` holds a ~60-entry seed lexicon and ~60 drug names as the realizer's stand-in and critical-term list. Set `SIPG_CRITICAL_TERMS=/path/list.txt` to load the ~200-term clinician list. This lexicon is *not* read by the policy.
* No Med-EASi / ClinicalEase / Cochrane-auto: `data.load_parallel_jsonl` + `pilot.py --corpus file.jsonl` consume them once downloaded. The pilot below uses a synthetic template corpus.
* Not implemented: T5 SFT of the actor (Phase 2), BLEU/ROUGE, clinician protocol, ablation harness.

## Review round 2 (graph construction, prompts, traversal log, reward)
**1. Graph / NER.** NER never merged anything here (it was disabled), but `GraphBuilder` now *refuses* a spaCy pipeline containing
`merge_entities/merge_noun_chunks`. NER is enabled and entity spans are stored as node attributes (`meta["entities"]`); the internal
syntax of multi-token terms is kept (`meta["structure"]` = token/dep/head, e.g. `hypertensive amod urgency`). A parser quirk surfaced
and was fixed: imperative "Discontinue" is tagged `amod`; a modifier now counts only if its head lies *inside* the term.
**2. Adjective + noun prompt.** New `LEX_NP`/`AMOD` templates use the requested wording ("replace the whole noun phrase with a
patient-friendly equivalent that preserves the clinical meaning"). `CallableRealizer` rewrites the whole NP for AMOD nodes. Lexicon gained
NP entries (hypertensive urgency/crisis/emergency, diabetic ketoacidosis) and plural matching.
**3. Traversal log.** `env.format_trail(env)` prints step, node, action, d_reward, running return, and unvisited nodes.
`scripts/diagnose.py` runs the H3 ablation and prints the trail for sentence S1 (I did not have your S1; the script uses
"Discontinue lisinopril 10 mg if the patient develops hypertensive urgency, because ...": edit `S1` at the top).
**4. Reward sanity** (`scripts/reward_sanity.py`; your message was cut off at item 4, so these are the standard checks):
* Telescoping identity holds (max error 1.7e-16).
* **Sim cannot catch critical errors**: dose dropped / negation flipped / dose x10 / drug swapped score Sim 0.64-0.84, versus 0.82-0.90 for a
  faithful rewrite, and raw R of an unsafe edit exceeds the unedited source. Safety rests entirely on the verifier gate, so keep
  `reject_penalty` >= the typical edit gain (median 0.115; default 0.10 is borderline, 0.25 recommended with an LLM realizer).
* S saturates: 67% of invoke-all outputs have S >= 0.95 (19% >= 0.999), so late-stage gradient is flat.
* Gamma (0.01) is ~12x smaller than a typical gain: it only breaks ties. 17% of changing edits lower S immediately (e.g. a relative-clause
  split before the jargon adjective is rewritten: -0.145 alone, +0.029 after).
* Bug found and fixed while doing this: the source-vector cache was keyed on the graph, not the encoder, which silently broke any run that
  reused graphs with a different encoder (it made one ablation collapse to "always STOP").

### H3 ablation (single seed, 301 train / 149 test, 120 PPO iters, all evaluated with the baseline reward)
| config | dR | S | Sim | prompts fired (non-trivial) | condition node visited | RELCL fired |
|---|---|---|---|---|---|---|
| greedy-readability | +0.402 | 0.925 | 0.884 | 98% | n/a | 92% |
| baseline gamma=.01 | +0.398 | 0.900 | 0.920 | 89% | 100% | 41% |
| gamma=.002 | +0.406 | 0.907 | 0.934 | 91% | 100% | 57% |
| coverage bonus .05 | +0.415 | 0.925 | 0.910 | 98% | 100% | 85% |
| binary Sim + reject .25 | +0.384 | 0.889 | 0.918 | 87% | 100% | 62% |

The condition node was reached in 100% of the 10 test sentences that have one, so H3 as "policy never gets there" is **not supported** here.
What differs is *which prompts fire*: the baseline skips relative-clause splits (Sim cost > S gain). A coverage bonus restores coverage.
Differences of ~0.01 dR are within seed noise.

## Pilot result with coverage bonus .05 (synthetic, 120 iters, seed 0)
| policy | dR | S | Sim |
|---|---|---|---|
| no-op | 0.000 | 0.423 | 1.000 |
| random | +0.023 | 0.465 | 0.987 |
| invoke-all | +0.386 | 0.923 | 0.881 |
| greedy-readability | +0.408 | 0.925 | 0.895 |
| **SIPG-RL (PPO)** | +0.409 | 0.925 | 0.899 |

Paired bootstrap: PPO - random = +0.386 (CI [+0.356, +0.417]); PPO - greedy = +0.0007 (CI [-0.0008, +0.0025]).
**The Phase-1 kill criterion is still triggered on this synthetic pilot**: the learned policy matches but does not beat greedy.
Caveats: greedy peeks at the exact scorer, the template edits are mostly independent, and the corpus is synthetic. This validates the
machinery, not the paper's hypothesis; that needs real corpora, the real Suitability Scorer, and an LLM realizer.

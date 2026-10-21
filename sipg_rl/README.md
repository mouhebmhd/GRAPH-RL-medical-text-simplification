# SIPG-RL — reference implementation

Syntax-Induced Prompt Graph + PPO traversal policy + post-hoc fidelity verifier, following the proposal.

```
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python -m pytest tests -q                                  # 7 tests
python scripts/pilot.py --n 450 --iters 120                # Phase-1 pilot + kill-criterion check (~1 min CPU)
python scripts/demo.py "The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain."
```

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

## Model fixes (audit items A1, A2, B1-B5, C1-C4, D3, E1-wiring)
All of this is about the *model*; nothing here depends on a dataset. Numbers below are from the synthetic template corpus, which only
checks the machinery (templates favour greedy), so they are not scientific evidence.

| Item | What changed | Where |
|---|---|---|
| A1 Sim | `ContentEncoder`: length-invariant, source-anchored weighted F-measure over content tokens (recall weighted 4x), concept-normalised, numbers / units / negations / hedges / conditionals / drug names carry 3x weight and multiply the score by their recall and precision. Faithful rewrite 1.00 (old 0.84), appended explanation 0.99 (old 0.76), dropped dose 0.55, dropped negation 0.76, swapped drug 0.62. Still offline: it cannot know a layman synonym outside the lexicon; for that, use an embedding/NLI model at evaluation time. | `scoring.py` |
| A2 / B2 S | `BandSuitabilityScorer`: smooth Gaussian bands around a target (grade 6.5 default; strict above, tolerant below because critical terms are masked to 1 syllable; polysyllable share and jargon penalised only when too high). No clipped plateau, so late-stage ordering keeps a gradient; over-simplified fragments are penalised. `.fit(reference_texts)` anchors the bands to any reference distribution; `to_dict/from_dict` persist them. Saturation (S >= 0.95) on invoke-all outputs: 19% vs 45% for the old scorer. | `scoring.py` |
| B1 | `gamma` 0.01 -> 0.03 (about 25% of the median edit gain, tested to stay in 10-60%). | `env.py`, `pilot.py` |
| B3 | Coverage bonus is off everywhere; trainer defaults to 0. | `env.py`, `train_policy.py` |
| B4 | `train_policy.py --realizer llm` trains against the SAME rules-first LLM realizer (persistent cache, reject penalty 0.25). Every checkpoint writes `<ckpt>.meta.json`; `SIPGRL(checkpoint=...)` warns on legacy checkpoints and on train/deploy mismatches (policy trained with rules, deployed with LLM = inference-time augmentation). | `scripts/train_policy.py`, `pipeline.py` |
| B5 | `--seeds 0,1,2,3,4`: per-seed checkpoints, mean +- std and a t-interval across seeds. | `scripts/train_policy.py` |
| C1 / C3 | `RoutedRealizer`: rules always first; the LLM only gets span-local lexical rewrites (long words, nominalizations, jargon adjective+noun) the rules left untouched, behind `lexical_guard` (no sentence punctuation, no new numbers, length cap). Embedded clauses (CCOMP/ACL) reach the LLM only with `allow_llm_structure=True`. Prompt now says "replace only this span, do not restructure". | `realize.py`, `llm.py` |
| C2 | Lexicon 70 -> 127 reviewed plain-wording pairs. | `lexicon.py` |
| C4 | Root cause was routing, not coverage: "hypertensive urgency" already had a rule, but the old LLM-first realizer sent it to the LLM, which returned it unchanged, so the rule never ran. Test: `test_lexicon_terms_never_reach_the_llm_c4`. | `realize.py` |
| D3 | `lexicon_entry_problem` / `assert_lexicon_safe` run at import and must be called by any lexicon miner: no drug names, numbers, percentages, dose units or code-like tokens in keys or replacements. | `lexicon.py` |
| E1 wiring | The final fallback (output := source) was structurally unreachable: the per-edit gate and the final gate ran the same lexical check, and NLI only ran on sentences with a number or drug term. Now `nli_scope="all"` checks every sentence, final rejections are reported (`final_rejected`), and `adversarial.selftest` injects 6 corruption types and counts where each is stopped (edit gate / final gate / leaked). With NLI off, semantic drift leaks 100%; with an NLI model the final fallback fires. | `verifier.py`, `adversarial.py`, `pipeline.py` |
| F1 (model side) | Graphs now record `n_dropped` (nodes lost to the 24-node cap; clauses come first, so TERM nodes are what get truncated). | `graph.py` |
| F3 | The LLM cache works (key = prompt + span + full sentence, so a single pass over distinct sentences cannot hit it; running `llm` then `llm_all` does). New persistent `cache_path` reuses calls across runs and seeds. | `llm.py` |

Synthetic 5-seed result with the corrected reward (200 PPO iterations, `runs/v2_long.log`): PPO dR +0.4194 +- 0.0046; PPO - greedy
-0.0052 (95% CI [-0.0109, +0.0005]); edits PPO - greedy -0.125 ([-0.206, -0.044]); S PPO - greedy -0.009 ([-0.015, -0.003]); verifier
preservation 100%. So PPO is slightly more parsimonious and slightly less readable than greedy and still not distinguishable on reward:
gamma now makes the trade visible, but there is no demonstrated win over greedy. `sipg_policy_v2.pt` is seed 0 (declared in advance, not picked).
Checkpoints from before this change (`sipg_policy*.pt` without `.meta.json`) were trained under the old reward and will warn.
Tests: `tests/test_reward_fixes.py`, `test_routing.py`, `test_lexicon_safety.py`, `test_adversarial.py`, `test_meta_warnings.py`.

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

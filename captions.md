**Fig. 2.** Injection recall at the verifier, per injector. The verifier catches 738/738 injections whose failure is visible to lexical checks; antonym substitution (0/90) preserves every surface cue it checks. Lexical checks are structurally blind to semantic drift — this is why the final gate exists.

**Fig. 3.** Verifier ablation per violation category. Left: share of injected violations caught (higher is better). Right: unsafe-sentence rate (lower is better). The effect is not uniform; conditional-drop detection collapses from 100% to 23.1% when the verifier is ablated.

**Fig. 4.** Antonym drift case study. Each drifted output passes every lexical check listed; only entailment catches it.

**Fig. 5.** SARI inverts with the reference edit distribution. The same rule-based system beats the no-op copy baseline on a replacement-heavy fixture (n=24) and loses to it on Med-EASi (minimal-edit references). SARI's ranking depends on the references, not on the system.

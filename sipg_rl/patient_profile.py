"""Health Literacy and Patient Profile module.

ARCHITECTURAL INVARIANT: a PatientProfile conditions (1) the GENERATION layer (LLM prompts) and (2) the EVALUATION
scorer (target FKGL band, RVS/RTS targets). It is NEVER passed to `FidelityVerifier`, which stays profile-blind, so
personalisation cannot weaken a clinical safety check. A text that misses profile targets is *flagged for revision*,
never safety-rejected (only the verifier rejects). tests/test_patient_profile.py enforces this.

(Named patient_profile.py, not profile.py, to avoid shadowing the stdlib `profile` module.)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from .coherence import fkgl, rvs, rts

# ---- heuristic thresholds: DEFAULTS to calibrate on your data, not validated constants ---------------------------
LOW_PROFICIENCY, HIGH_PROFICIENCY = 0.34, 0.67
GRADE_BANDS = {"low": (4.0, 6.0), "mid": (6.0, 8.0), "high": (8.0, 10.0)}   # low = 4th-6th grade (module spec)
STEADY_CAPS = {"rvs_max": 3.0, "rts_max": 3.5}          # grade-level units; replace via calibrate_coherence()
FLEXIBLE_CAPS = {"rvs_max": None, "rts_max": None}      # no cap
TRANSITIONAL_VARIANCE = 0.60                            # topic_variance >= this -> Transitional
SPECIALIZED_GAP = 0.30                                  # |digital_preference - 0.5| >= this -> Specialized

SPIRITUAL_SEQUENCE = ("Begin by validating divine will or spiritual context. Address moral or social implications. "
                      "Present psychological support as a means of achieving spiritual peace. Finally, introduce "
                      "biomedical options as complementary tools.")
SAFETY_FOOTER = ("Never change, drop or add numbers, doses, units, drug names, negations (no, not, without) or hedges "
                 "(may, possible, if, unless). Do not add medical facts that are not in the source.")


def _clip01(x) -> float:
    return min(1.0, max(0.0, float(x)))


@dataclass
class PatientProfile:
    """Continuous, multidimensional profile (all scores in [0,1]; 1 = high)."""
    core_proficiency: float = 0.5      # integrated critical+communicative+expressed health literacy
    coherence_need: str = "auto"       # 'steady' | 'flexible' | 'auto' (auto: steady if low proficiency / transitional)
    spiritual_first: bool = False      # explanatory model: divine attribution -> moral/social -> biomedical
    digital_preference: float = 0.5    # 0 = traditional/functional text pathway, 1 = digital pathway
    adaptability: float = 0.5          # dynamic: ability to adapt to new terminology over time
    topic_variance: float = 0.0        # observed variance of the patient's expression across topics (Transitional cue)
    language: str = "en"
    name: str = ""

    def __post_init__(self):
        for f in ("core_proficiency", "digital_preference", "adaptability", "topic_variance"):
            setattr(self, f, _clip01(getattr(self, f)))
        if self.coherence_need not in ("steady", "flexible", "auto"):
            raise ValueError("coherence_need must be 'steady', 'flexible' or 'auto'")

    # ---- construction -------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict | None) -> "PatientProfile":
        if isinstance(d, PatientProfile):
            return d
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- derived ------------------------------------------------------------------------------------------------
    @property
    def archetype(self) -> str:
        """Balanced | Specialized | Transitional. Priority: transitional > specialized > balanced."""
        if self.topic_variance >= TRANSITIONAL_VARIANCE:
            return "transitional"
        if abs(self.digital_preference - 0.5) >= SPECIALIZED_GAP:
            return "specialized"
        return "balanced"

    @property
    def proficiency_level(self) -> str:
        return "low" if self.core_proficiency < LOW_PROFICIENCY else "high" if self.core_proficiency >= HIGH_PROFICIENCY else "mid"

    @property
    def pathway(self) -> str:
        return "digital" if self.digital_preference >= 0.5 else "functional"

    @property
    def steady_flow(self) -> bool:
        if self.coherence_need != "auto":
            return self.coherence_need == "steady"
        return self.archetype == "transitional" or self.proficiency_level == "low"

    def targets(self) -> "ProfileTargets":
        lo, hi = GRADE_BANDS[self.proficiency_level]
        caps = STEADY_CAPS if self.steady_flow else FLEXIBLE_CAPS
        return ProfileTargets(grade_lo=lo, grade_hi=hi, rvs_max=caps["rvs_max"], rts_max=caps["rts_max"])


@dataclass
class ProfileTargets:
    grade_lo: float
    grade_hi: float
    rvs_max: float | None = None
    rts_max: float | None = None

    @property
    def grade_mid(self) -> float:
        return (self.grade_lo + self.grade_hi) / 2


PRESETS = {
    "balanced": PatientProfile(core_proficiency=0.6, name="balanced"),
    "low_literacy": PatientProfile(core_proficiency=0.15, coherence_need="steady", name="low_literacy"),
    "specialized_digital": PatientProfile(core_proficiency=0.5, digital_preference=0.9, name="specialized_digital"),
    "specialized_functional": PatientProfile(core_proficiency=0.5, digital_preference=0.1, name="specialized_functional"),
    "transitional": PatientProfile(core_proficiency=0.4, topic_variance=0.8, adaptability=0.3, name="transitional"),
    "spiritual_first": PatientProfile(core_proficiency=0.3, spiritual_first=True, name="spiritual_first"),
}


def load_profile(spec: str | None) -> PatientProfile | None:
    """'balanced' (preset) | path to a .json file | inline JSON object | None."""
    import json, os
    if not spec:
        return None
    if spec in PRESETS:
        return PRESETS[spec]
    if os.path.exists(spec):
        return PatientProfile.from_dict(json.load(open(spec, encoding="utf8")))
    try:
        return PatientProfile.from_dict(json.loads(spec))
    except Exception as e:
        raise ValueError(f"unknown profile '{spec}' (presets: {sorted(PRESETS)}, or a JSON file / inline JSON)") from e


# ======================================================================================================================
# Generation layer: profile -> prompts
# ======================================================================================================================
def _readability_line(p: PatientProfile) -> str:
    t = p.targets()
    if p.proficiency_level == "low":
        return "Use a 4th-6th grade vocabulary: short, everyday words and short sentences."
    return f"Aim for roughly grade {t.grade_lo:.0f}-{t.grade_hi:.0f} reading level in plain language."


def _coherence_line(p: PatientProfile) -> str:
    if p.archetype == "transitional":
        return ("Use highly structured, predictable sentence formats (same pattern every time, uniform length, no abrupt "
                "topic shifts) and avoid analogies and metaphors: be a stable scaffold.")
    if p.steady_flow:
        return "Keep sentence lengths uniform and avoid abrupt topic shifts."
    return "Sentence length may vary naturally, but keep the narrative coherent."


def _pathway_line(p: PatientProfile) -> str | None:
    if p.archetype != "specialized":
        return None
    if p.pathway == "digital":
        return ("Use short labelled sections suitable for a screen. Do NOT invent links or URLs; mark where a link would "
                "help as [link: topic].")
    return "Use structured, text-only, step-by-step explanations (numbered steps, one idea per step)."


def build_system_prompt(p: PatientProfile) -> str:
    """Document-level system prompt for an LLM pass (the 'Generation Layer')."""
    lines = ["You rewrite clinical text for a specific patient.", _readability_line(p), _coherence_line(p)]
    pw = _pathway_line(p)
    if pw:
        lines.append(pw)
    if p.archetype == "balanced":
        lines.append("Address both the biological and the psychosocial context where the source supports it.")
    if p.spiritual_first:
        lines += [f"Cultural/spiritual framing - follow this narrative order: {SPIRITUAL_SEQUENCE}",
                  "Never contradict the patient's spiritual worldview."]
    lines.append(SAFETY_FOOTER)
    return " ".join(lines)


def span_style_suffix(p: PatientProfile) -> str:
    """Short style hint appended to the span-level LLM system prompts. A span rewrite cannot reorder a narrative, so
    only the readability / no-analogy / respect constraints apply here; ordering is done in the document-level pass."""
    parts = [_readability_line(p)]
    if p.archetype == "transitional":
        parts.append("Avoid analogies and metaphors.")
    if p.spiritual_first:
        parts.append("Never contradict the patient's spiritual worldview.")
    return " ".join(parts)


# ======================================================================================================================
# Evaluation layer: profile-conditioned scorer + coherence report
# ======================================================================================================================
def band_score(grade: float, lo: float, hi: float, slack: float = 4.0) -> float:
    """1.0 inside [lo, hi]; decays linearly to 0 at `slack` grades outside."""
    d = lo - grade if grade < lo else grade - hi if grade > hi else 0.0
    return max(0.0, 1.0 - d / slack)


class ProfileSuitabilityScorer:
    """S(y) = (1-w) * base(y) + w * band(FKGL(y)) : the profile only shifts the TARGET band of the readability signal.
    Works with any base scorer `callable(text)->[0,1]`. Sentence-level (RVS/RTS need a document: see profile_report)."""

    def __init__(self, base, profile: PatientProfile, weight: float = 0.5):
        self.base, self.profile, self.w = base, profile, weight
        self.t = profile.targets()

    def __call__(self, text: str) -> float:
        return float((1 - self.w) * self.base(text) + self.w * band_score(fkgl(text), self.t.grade_lo, self.t.grade_hi))

    def __getattr__(self, name):          # keep base-scorer attributes (features, tg, fit ...) reachable
        if name in ("base", "profile", "w", "t"):
            raise AttributeError(name)
        return getattr(self.base, name)


@dataclass
class ProfileReport:
    fkgl: float
    in_band: bool
    rvs: float
    rts: float
    rvs_ok: bool
    rts_ok: bool
    needs_revision: bool
    reasons: list = field(default_factory=list)
    archetype: str = ""
    target_band: tuple = (0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


def profile_report(text: str, p: PatientProfile) -> ProfileReport:
    """Readability/coherence check against the profile. 'needs_revision' is a GENERATION-layer flag; it is NOT a
    safety verdict and must never gate output the way FidelityVerifier does."""
    t, g, v, r = p.targets(), fkgl(text), rvs(text), rts(text)
    in_band = t.grade_lo - 0.5 <= g <= t.grade_hi + 0.5            # half-grade tolerance
    rvs_ok = t.rvs_max is None or v <= t.rvs_max
    rts_ok = t.rts_max is None or r <= t.rts_max
    why = []
    if not in_band:
        why.append(f"FKGL {g:.1f} outside target {t.grade_lo:.0f}-{t.grade_hi:.0f}")
    if not rvs_ok:
        why.append(f"RVS {v:.2f} > {t.rvs_max}")
    if not rts_ok:
        why.append(f"RTS {r:.2f} > {t.rts_max}")
    return ProfileReport(g, in_band, v, r, rvs_ok, rts_ok, bool(why), why, p.archetype, (t.grade_lo, t.grade_hi))

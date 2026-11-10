#!/usr/bin/env python
"""
make_figures.py — Figures 2-5 from the paper plan (vector PDF + 300-dpi PNG).

  Fig 2  injection recall (100% x5 vs antonym 0%)        data embedded (from your memo)
  Fig 3  verifier ON vs OFF per category                 data from --ablation CSV
  Fig 4  antonym drift case study                        data embedded (edit CASES / --cases JSON)
  Fig 5  SARI inversion, Med-EASi vs fixture             data embedded (from your memo)

Usage:  python make_figures.py --ablation fig_data/ablation.csv --out figures
Fig 3 is skipped (with a message) for any category whose values are blank in the CSV.
"""
import argparse, csv, json, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, RED, GREY = "#0072B2", "#E69F00", "#D55E00", "#7f7f7f"   # Okabe-Ito
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "pdf.fonttype": 42})

# ---- data from the memo ------------------------------------------------------------------
INJ = [("swap_drug", 100, 273), ("dose_x10", 100, 228), ("drop_negation", 100, 211),
       ("drop_hedge", 100, 11), ("drop_conditional", 100, 15), ("antonym", 0, 90)]
CASES = [  # source, drifted, checks passed  (replace with real injector outputs before submission)
    ("The wound is healing well.", "The wound is not healing well.", "numbers ✓   units ✓   negation-cue count ✓"),
    ("Patient tolerates the medication.", "Patient does not tolerate the medication.", "numbers ✓   units ✓   negation-cue count ✓"),
    ("Blood pressure is stable.", "Blood pressure is unstable.", "numbers ✓   units ✓"),
    ("The rash improved.", "The rash worsened.", "numbers ✓   units ✓"),
]
SARI = {"A": ("A  Med-EASi\n(minimal-edit references)", {"copy": 51.8, "rules": 49.7, "llm": 42.2}),
        "B": ("B  Fixture, n=24\n(replacement references)", {"copy": 44.7, "rules": 56.5})}


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def fig2(out):
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    ys = list(range(len(INJ)))[::-1]
    for y, (name, rec, n) in zip(ys, INJ):
        if rec > 0:
            ax.barh(y, rec, color=BLUE, height=0.62)
            ax.text(rec - 1.2, y, f"{rec:.0f}%", va="center", ha="right", color="white", fontweight="bold")
        else:
            ax.barh(y, 100, color="none", edgecolor=RED, linestyle="--", linewidth=1.1, height=0.62)
            ax.text(1.5, y, "0%", va="center", ha="left", color=RED, fontweight="bold")
            msg = textwrap.fill("lexical checks are structurally blind to semantic drift — "
                                "this is why the final gate exists", 52)
            ax.text(12, y, msg, va="center", ha="left", color=RED, fontsize=7.6, style="italic", linespacing=1.15)
        ax.text(101.5, y, f"n={n}", va="center", ha="left", color=GREY, fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in INJ], family="monospace")
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(INJ) - 0.4)
    ax.set_xlabel("Injection recall at the verifier (%)")
    ax.tick_params(axis="y", length=0)
    save(fig, out, "fig2_injection_recall")


def fig3(out, csv_path):
    rows, skipped = [], []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["full"].strip() == "" or r["ablated"].strip() == "":
                skipped.append(r["category"]); continue
            rows.append((r["category"], float(r["full"]), float(r["ablated"])))
    if skipped:
        print("Fig 3: blank values for", skipped, "-> not plotted. Fill them in", csv_path)
    catch = [r for r in rows if r[0] != "unsafe_sentence_rate"]
    unsafe = [r for r in rows if r[0] == "unsafe_sentence_rate"]
    if not catch:
        print("Fig 3 skipped: no complete rows."); return
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 3.2), gridspec_kw={"width_ratios": [max(len(catch), 1), 1.0 if unsafe else 0.001]})
    w = 0.38
    for i, (cat, full, abl) in enumerate(catch):
        a.bar(i - w / 2, full, w, color=BLUE, label="full verifier" if i == 0 else None)
        a.bar(i + w / 2, abl, w, color=ORANGE, label="verifier ablated" if i == 0 else None)
    big = max(range(len(catch)), key=lambda i: catch[i][1] - catch[i][2])
    drop = catch[big][1] - catch[big][2]
    if drop >= 20:
        a.annotate(f"{catch[big][1]:g}% → {catch[big][2]:g}%", xy=(big + w / 2, catch[big][2]),
                   xytext=(big + 0.62, min(catch[big][2] + 30, 92)), ha="center", color=RED, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    a.set_xticks(range(len(catch))); a.set_xticklabels([c[0] for c in catch], rotation=20, ha="right")
    a.set_ylim(0, 105); a.set_ylabel("Violations caught (%)  ↑ better"); a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8); a.set_xlim(-0.6, len(catch) + 0.2)
    if unsafe:
        _, full, abl = unsafe[0]
        b.bar(-w / 2, full, w, color=BLUE); b.bar(w / 2, abl, w, color=ORANGE)
        b.set_xticks([0]); b.set_xticklabels(["unsafe-sentence\nrate"]); b.set_ylabel("Unsafe sentences (%)  ↓ better")
        b.set_ylim(0, max(full, abl, 1) * 1.25)
    else:
        b.axis("off")
    fig.tight_layout()
    save(fig, out, "fig3_ablation")


def fig4(out, cases):
    fig, ax = plt.subplots(figsize=(6.9, 0.7 * len(cases) + 0.8)); ax.axis("off")
    cols = [0.0, 0.31, 0.66]; heads = ["Source", "Drifted output", "Lexical checks that passed"]
    n = len(cases); top = 1.0; rh = 0.86 / n
    for x, h in zip(cols, heads):
        ax.text(x + 0.005, top - 0.04, h, fontweight="bold", va="center", transform=ax.transAxes)
    ax.plot([0, 1], [top - 0.1] * 2, color="black", lw=0.9, transform=ax.transAxes, clip_on=False)
    for i, (src, drift, chk) in enumerate(cases):
        y = top - 0.1 - rh * (i + 0.5)
        ax.text(cols[0] + 0.005, y, textwrap.fill(src, 24), va="center", fontsize=8.5, transform=ax.transAxes)
        ax.text(cols[1] + 0.005, y, textwrap.fill(drift, 26), va="center", fontsize=8.5, color=RED, transform=ax.transAxes)
        ax.text(cols[2] + 0.005, y, chk.replace("   ", "\n"), va="center", color="#117733", transform=ax.transAxes, fontsize=8, linespacing=1.2)
        ax.plot([0, 1], [top - 0.1 - rh * (i + 1)] * 2, color="#cccccc", lw=0.6, transform=ax.transAxes, clip_on=False)
    save(fig, out, "fig4_antonym_cases")


def fig5(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), sharey=True)
    pos = {"copy": 0, "rules": 1, "llm": 2}; col = {"copy": GREY, "rules": BLUE, "llm": ORANGE}
    for ax, key in zip(axes, "AB"):
        title, d = SARI[key]
        ax.plot([pos["copy"], pos["rules"]], [d["copy"], d["rules"]], color="#444444", lw=1.4, zorder=1)
        for s, v in d.items():
            ax.scatter(pos[s], v, s=70, color=col[s], zorder=3)
            ax.text(pos[s], v + 1.6, f"{v:.1f}", ha="center", fontsize=8.5)
        delta = d["rules"] - d["copy"]
        ax.text(0.03, 0.96, ("rules − copy = " + f"{delta:+.1f}").replace("-", "−"), transform=ax.transAxes,
                ha="left", va="top", color=RED, fontweight="bold", fontsize=8.5)
        ax.set_xticks([pos[s] for s in d]); ax.set_xticklabels(list(d)); ax.set_xlim(-0.5, 2.5 if "llm" in d else 1.5)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_ylabel("SARI"); axes[0].set_ylim(36, 66)
    fig.tight_layout()
    save(fig, out, "fig5_sari_inversion")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures"); ap.add_argument("--ablation", default="fig_data/ablation.csv")
    ap.add_argument("--cases", default=None, help="optional JSON list of [source, drifted, checks]")
    a = ap.parse_args(); out = Path(a.out)
    fig2(out); fig4(out, json.load(open(a.cases)) if a.cases else CASES); fig5(out)
    if Path(a.ablation).exists(): fig3(out, a.ablation)
    else: print(f"Fig 3 skipped: {a.ablation} not found (template: fig_data/ablation_template.csv)")

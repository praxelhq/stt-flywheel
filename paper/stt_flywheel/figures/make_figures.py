"""Generate Figures 1 and 2 for the STT flywheel paper."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SC_DIR = Path("/Users/pushpak/Documents/GitHub/praxy_tts/evaluation/scorecards/stt_flywheel")
OUT_DIR = Path("/Users/pushpak/Documents/GitHub/praxy_tts/paper/stt_flywheel/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.load(open(SC_DIR / name))


# ---- Figure 1: Entity-dense EHR per system, Te headline + Hi/Ta side panels ----
te_vanilla = load("praxy_te_r2_entity_dense_cartesia_n102_scorecard.json")["vanilla_whisper_v3"]["ehr"]
te_vasista = load("vasista_te_entity_dense_cartesia_n102_scorecard.json")["ehr"]
te_deepgram = load("deepgram_te_entity_dense_cartesia_n102_scorecard.json")["ehr"]
te_praxy = load("beta_te_full_scorecard.json")["by_holdout"]["entity_dense_cartesia"]["ehr"]

hi_vasista = load("vasista_hi_entity_dense_cartesia_n86_scorecard.json")["ehr"]
hi_deepgram = load("deepgram_hi_entity_dense_cartesia_n86_scorecard.json")["ehr"]
ta_vasista = load("vasista_ta_entity_dense_cartesia_n102_scorecard.json")["ehr"]
ta_deepgram = load("deepgram_ta_entity_dense_cartesia_n102_scorecard.json")["ehr"]

systems = ["Vanilla\nWhisper-v3", "vasista22\n(open SOTA)", "Deepgram\nNova-3", "Praxy-STT-r$\\beta$\n(ours)"]
te_vals = [te_vanilla, te_vasista, te_deepgram, te_praxy]
colors = plt.get_cmap("tab10").colors
bar_colors = [colors[0], colors[1], colors[2], colors[3]]

fig, ax = plt.subplots(figsize=(5.0, 3.4))
x = np.arange(len(systems))
bars = ax.bar(x, te_vals, color=bar_colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, te_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=9)
ax.set_ylabel("Entity-Hit-Rate (EHR)", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title("Telugu entity-dense held-out (n=102)", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

# Hi/Ta annotation
ann = (
    "Hindi entity-dense (n=86): vasista=%.3f, Deepgram=%.3f.  Praxy-r$\\beta$: pending.\n"
    "Tamil entity-dense (n=102): vasista=%.3f, Deepgram=%.3f.  Praxy-r$\\beta$: pending."
) % (hi_vasista, hi_deepgram, ta_vasista, ta_deepgram)
fig.text(0.5, -0.02, ann, ha="center", va="top", fontsize=7.5, style="italic")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_ehr_te.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_ehr_te.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("Wrote", OUT_DIR / "fig_ehr_te.pdf")


# ---- Figure 2: SFR by lang on CV25 across 3 systems ----
te_sc = load("praxy_te_r2_cv25_n86_scorecard.json")
hi_sc = load("praxy_hi_r2_cv25_n100_scorecard.json")
ta_sc = load("praxy_ta_r2_cv25_n100_scorecard.json")

sfr = {
    "Te": {
        "vanilla": te_sc["vanilla_whisper_v3"]["sfr_mean"],
        "praxy_r2": te_sc["praxy_te_r2"]["sfr_mean"],
        "vasista": load("vasista_te_cv25_n86_scorecard.json")["sfr_mean"],
    },
    "Hi": {
        "vanilla": hi_sc["vanilla_whisper_v3"]["sfr_mean"],
        "praxy_r2": hi_sc["praxy_hi_r2"]["sfr_mean"],
        "vasista": load("vasista_hi_cv25_n3326_scorecard.json")["sfr_mean"],
    },
    "Ta": {
        "vanilla": ta_sc["vanilla_whisper_v3"]["sfr_mean"],
        "praxy_r2": ta_sc["praxy_ta_r2"]["sfr_mean"],
        "vasista": load("vasista_ta_cv25_n100_scorecard.json")["sfr_mean"],
    },
}

langs = ["Te", "Hi", "Ta"]
series = [
    ("Vanilla Whisper-v3", "vanilla", colors[0]),
    ("Praxy-STT-r2 (W-v3 + LoRA)", "praxy_r2", colors[3]),
    ("vasista22 (open SOTA)", "vasista", colors[1]),
]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
x = np.arange(len(langs))
width = 0.26
for i, (label, key, color) in enumerate(series):
    vals = [sfr[lang][key] for lang in langs]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=label,
                  color=color, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(["Telugu", "Hindi", "Tamil"], fontsize=10)
ax.set_ylabel("Script Fidelity Rate (SFR)", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("CV25 held-out, per-system SFR", fontsize=10)
ax.legend(fontsize=8, loc="lower right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_sfr_lang_conditional.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_sfr_lang_conditional.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("Wrote", OUT_DIR / "fig_sfr_lang_conditional.pdf")

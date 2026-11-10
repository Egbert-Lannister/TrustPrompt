"""Rebuild the two TrustPrompt website charts from preserved source JSON.

Run from any directory: python build_result_figures.py
Requires matplotlib and numpy. No evaluation or model inference is run.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets"
ROOT = HERE.parents[2]
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "test_results.source.json": ROOT / "docs/test_results.json",
    "tta_test.source.json": ROOT / "results/tta_test_20261104_202618/tta_test.json",
}
for filename, source in SOURCES.items():
    if not (HERE / filename).exists():
        shutil.copy2(source, HERE / filename)

main = json.loads((HERE / "test_results.source.json").read_text(encoding="utf-8"))
capa = json.loads((HERE / "tta_test.source.json").read_text(encoding="utf-8"))

# CHART_PLAN: two dataset facets; fixed categorical report order; raw versus
# masked estimates; main method red/white hatch; no uncertainty fabricated.
RED = "#C0392B"
GRAY = "#BFC2C4"
TEXT = "#282D30"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "hatch.color": "white", "hatch.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def finish_axes(ax, horizontal=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A5AAAC")
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.tick_params(length=0, pad=5)
    axis = ax.xaxis if horizontal else ax.yaxis
    axis.grid(True, color="#E5E7E8", linewidth=0.65)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=180)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)


models = [
    ("promptda_l", "PromptDA-L"),
    ("priorda", "PriorDA-B v1.1"),
    ("omnidc", "OMNI-DC v1.1"),
    ("cdm", "CDM"),
    ("lingbot", "LingBot-Depth v0.5"),
    ("any2full", "Any2Full-L"),
]
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharey=True)
fig.subplots_adjust(left=.175, right=.97, bottom=.22, top=.80, wspace=.12)
for ax, dataset, title, maximum, step in zip(
    axes, ["mirror3d_nyu", "hammer"],
    ["(a) NYU Mirror3D · 30 images / 23 scenes", "(b) HAMMER · 858 samples / 3 scenes"],
    [.97, .205], [.2, .05]
):
    for row, (key, label) in enumerate(models):
        entry = main[dataset][key]
        candidate_key = next(k for k in entry if k.startswith("T v1 mask"))
        raw = entry["raw"]["c"]
        candidate = entry[candidate_key]["c"]
        ax.barh(row-.17, raw, height=.29, color=GRAY, edgecolor="white", linewidth=.5)
        ax.text(raw+maximum*.012, row-.17, f"{raw:.4f}", va="center", fontsize=9)
        if candidate is None or not np.isfinite(candidate):
            ax.text(maximum*.012, row+.17, "NA · invalid prediction in 1 image", va="center", fontsize=8.8, color=RED)
        else:
            ax.barh(row+.17, candidate, height=.29, color=RED, edgecolor="white", hatch="//", linewidth=.5)
            ax.text(candidate+maximum*.012, row+.17, f"{candidate:.4f}", va="center", fontsize=9, color=RED)
    ax.set_yticks(np.arange(len(models)), [m[1] for m in models])
    ax.set_ylim(5.55, -.55)
    ax.set_xlim(0, maximum)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("C-region AbsRel ↓", labelpad=8)
    ax.set_title(title, fontsize=10.2, fontweight="bold", pad=13)
    finish_axes(ax, horizontal=True)
axes[1].tick_params(axis="y", left=False)
fig.legend(handles=[Patch(facecolor=GRAY, edgecolor="white", label="Raw prompt"),
                    Patch(facecolor=RED, edgecolor="white", hatch="//", label="T v1 mask")],
           loc="upper center", bbox_to_anchor=(.56,.96), ncol=2, frameon=False, fontsize=10)
fig.text(.175, .10, "Frozen-method test · Point estimates; panel scales differ. Lower is better.", fontsize=9)
fig.text(.175, .06, "10/12 comparisons significant after Holm correction. CDM–NYU is NA; PromptDA-L–HAMMER is not significant.", fontsize=8.6)
save(fig, "formal-test-results")

# CHART_PLAN: categorical variants, not temporal steps; independent dataset
# panels; same variant color mapping; reduced error does not imply superiority
# to the frozen prior in NYU.
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
fig.subplots_adjust(left=.08, right=.97, bottom=.25, top=.80, wspace=.20)
for ax, dataset, title, maximum, step in zip(
    axes, ["nyu", "hammer"],
    ["(a) NYU Mirror3D · 30 images / 23 scenes", "(b) HAMMER · 9 images / 3 scenes"],
    [1.28,.18], [.3,.04]
):
    keys = ["prior", "capa_raw", "tmask"]
    values = [capa[dataset]["absrel_C"][k] for k in keys]
    bars = ax.bar(np.arange(3), values, width=.56,
                  color=["#D9DADB", "#A4A8AA", RED], edgecolor="white", linewidth=.8)
    bars[-1].set_hatch("//")
    for i, v in enumerate(values):
        ax.text(i, v+maximum*.022, f"{v:.4f}", ha="center", va="bottom", fontsize=11,
                color=RED if i==2 else TEXT)
    ax.set_xticks([0,1,2], ["Frozen prior", "CAPA", "CAPA + T v1"])
    ax.set_xlim(-.55,2.55)
    ax.set_ylim(0,maximum)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylabel("C-region AbsRel ↓", labelpad=7)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=13)
    finish_axes(ax)
fig.text(.08,.94,"CAPA test comparison · Secondary, descriptive analysis",fontsize=12,fontweight="bold")
fig.text(.08,.105,"Point estimates; panel scales differ. Lower is better. No confidence intervals shown.",fontsize=9)
fig.text(.08,.055,"The mask reduces error relative to CAPA; in NYU, error remains above the frozen prior.",fontsize=9)
save(fig, "capa-test-results")

manifest = {
    "style_reference": "oral-figure/references/bar-charts.md; script/bar_spice.py",
    "sources": [{"copy": name, "original": str(path.relative_to(ROOT)).replace("\\", "/"),
                 "sha256": hashlib.sha256((HERE/name).read_bytes()).hexdigest()}
                for name,path in SOURCES.items()],
    "charts": ["formal-test-results", "capa-test-results"],
    "uncertainty": "Point estimates only; no error bars invented.",
}
(HERE/"figure_sources.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"assets":str(OUT),"charts":manifest["charts"]},indent=2))

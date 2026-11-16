"""Regenerate the result figures of the ArabicStegoGuard paper (Figs. 2-5).

Reads results/results.json and results/revision_results.json and writes
vector PDFs (plus 300 dpi PNG previews) into results/.
Usage:  python make_figures.py [--results-dir results]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

FAMS = ["ZW1", "KHN", "KSH", "PSP"]
SHORT = {"B1-Rules": "B1", "B2-CodepointNgramLR": "B2", "B3-RawCharCNN": "B3",
         "B5-ConcatFusion": "B5", "B6-ArabicStegoGuard": "B6"}
COL = {"B1": "#8172B3", "B2": "#937860", "B3": "#4C72B0", "B5": "#DD8452", "B6": "#C44E52"}
MK = {"B1": "v", "B2": "D", "B3": "o", "B5": "s", "B6": "^"}
pct = lambda x: 100.0 * x


def save(fig, out, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def fig_seed_oppoint(R, out):
    """Fig. 2: hard-negative FPR across seeds (a) and at validation-matched TPR targets (b)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.95),
                                 gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.42))
    names = ["B3-RawCharCNN", "B5-ConcatFusion", "B6-ArabicStegoGuard"]
    for k, name in enumerate(names):
        v = [pct(s["at_FPR1"]["FPR_hardneg"]) for s in R["E1_seeds"][name]]
        b = SHORT[name]
        a1.scatter(k + np.array([-0.12, 0.0, 0.12])[:len(v)], v, s=14, marker=MK[b], color=COL[b],
                   zorder=3, edgecolor="white", linewidth=0.4)
        a1.errorbar(k + 0.3, np.mean(v), yerr=np.std(v), fmt="_", ms=7, color="black", lw=0.7,
                    capsize=2, capthick=0.6, zorder=4)
    a1.axhline(1.0, ls="--", lw=0.6, color="0.4", zorder=1)
    a1.set_xticks(range(3)); a1.set_xticklabels([SHORT[n] for n in names])
    a1.set_xlim(-0.5, 2.6); a1.set_ylim(0, 1.6)
    a1.set_ylabel("Hard-negative FPR (%)")
    a1.set_title("(a) Seeds 0, 1, 2", fontsize=7.5, pad=3)

    targets = list(next(iter(R["matched_op"].values())).keys())
    for name, ops in R["matched_op"].items():
        b = SHORT[name]
        a2.plot(range(len(targets)), [pct(ops[t]["FPR_hardneg"]) for t in targets],
                marker=MK[b], ms=3.6, lw=0.9, color=COL[b], label=b)
    a2.set_yscale("symlog", linthresh=1.0, linscale=1.0)
    a2.set_yticks([0, 0.5, 1, 5, 10, 15]); a2.set_yticklabels(["0", "0.5", "1", "5", "10", "15"])
    a2.set_ylim(-0.05, 20)
    a2.set_xticks(range(len(targets)))
    a2.set_xticklabels([f"{pct(float(t.split('=')[-1])):g}" for t in targets])
    a2.set_xlabel("Validation TPR target (%)")
    a2.set_title("(b) Matched operating points", fontsize=7.5, pad=3)
    a2.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.4, columnspacing=0.6,
              borderaxespad=0.1, labelspacing=0.25)
    for a in (a1, a2):
        a.grid(axis="y", lw=0.4, color="0.9", zorder=0)
        a.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig_seed_oppoint")


def fig_fpr_shift(A, R, out):
    """Fig. 3: hard-negative FPR on the original, expanded and news benign sets."""
    names = list(SHORT)
    labels = ["B1\nRules", "B2\n$n$-gram LR", "B3\nRaw CNN", "B5\nConcat", "B6\nASG"]
    indist = [pct(A["E1"][n]["FPR_hardneg"]) for n in names]
    exp = [pct(R["expanded_hardneg"]["FPR"][n]["FPR"]) for n in names]
    lo = [pct(R["expanded_hardneg"]["FPR"][n]["CI95"][0]) for n in names]
    hi = [pct(R["expanded_hardneg"]["FPR"][n]["CI95"][1]) for n in names]
    news = [pct(R["domain_news"]["zero_shot"][n]["FPR_hardneg"]) for n in names]
    x = np.arange(len(names)); w = 0.26
    cols = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    b1 = ax.bar(x - w, indist, w, color=cols[0], label="Wikipedia hard negatives", zorder=3)
    b2 = ax.bar(x, exp, w, color=cols[1], yerr=[np.subtract(exp, lo), np.subtract(hi, exp)],
                capsize=1.8, error_kw=dict(lw=0.6, capthick=0.6),
                label="Expanded curated set (95% CI)", zorder=3)
    b3 = ax.bar(x + w, news, w, color=cols[2], label="BBC Arabic news", zorder=3)
    for bars, tops in ((b1, indist), (b2, hi), (b3, news)):
        for r, top in zip(bars, tops):
            ax.text(r.get_x() + r.get_width() / 2, max(top + 0.25, 1.2), f"{r.get_height():.1f}",
                    ha="center", va="bottom", fontsize=5.6)
    ax.axhline(1.0, ls="--", lw=0.6, color="0.35", zorder=2, label="Validation FPR target (1%)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Hard-negative FPR (%)"); ax.set_ylim(0, 15.5)
    ax.grid(axis="y", lw=0.4, color="0.88", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2,
              handlelength=1.2, columnspacing=0.8, borderaxespad=0.1)
    save(fig, out, "fig_fpr_shift")


def fig_openset(A, R, out):
    """Fig. 4: held-out family TPR of the LOFO detectors and the remedies."""
    rows = [("B1 Rules", A["E2"]["B1-Rules"]), ("B2 $n$-gram LR", A["E2"]["B2-CodepointNgramLR"]),
            ("B3 Raw CNN", A["E2"]["B3-RawCharCNN"]), ("B6 ArabicStegoGuard", A["E2"]["B6-ArabicStegoGuard"])]
    for key, lab in (("AnomalyLR", "AnomalyLR"), ("IsolationForest", "IsolationForest"),
                     ("Hybrid-B6-AnomalyLR", "Hybrid B6+AnomalyLR")):
        rows.append((lab, {f: R["E2_remedies"][f][key] for f in FAMS}))
    M = np.array([[pct(r[f]["TPR"]) for f in FAMS] for _, r in rows])
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if M[i, j] > 60 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(FAMS)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Held-out embedding family")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.axhline(3.5, color="black", lw=1.0); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("TPR (%)"); cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6.5, width=0.5)
    save(fig, out, "fig_openset")


def fig_psr(A, out):
    """Fig. 5: payload survival ratio (a) and active stego documents (b) per transformation."""
    trans = list(A["PSR"].keys())
    P = np.array([[pct(A["PSR"][t][f]) for f in FAMS] for t in trans])
    n_active = [A["E4"][t]["n_active"] for t in trans]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=True,
                                  gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.08))
    ax.imshow(P, cmap=ListedColormap(["#F2D7D5", "#2E5A87"]), vmin=0, vmax=100, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=6.8,
                    color="white" if P[i, j] > 50 else "#922B21")
    ax.set_xticks(range(4)); ax.set_xticklabels(FAMS)
    ax.set_yticks(range(len(trans))); ax.set_yticklabels([t.replace("_", " ") for t in trans])
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(trans), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0); ax.tick_params(which="both", length=0)
    ax.set_title("(a) PSR (%)", fontsize=7.5, pad=3)
    axb.barh(range(len(trans)), n_active, color="#7F8C8D", height=0.62, zorder=3)
    for i, n in enumerate(n_active):
        axb.text(n + 12, i, str(n), va="center", fontsize=6.6)
    axb.set_xlim(0, 760); axb.set_xticks([0, 300, 600]); axb.tick_params(axis="y", length=0)
    axb.grid(axis="x", lw=0.4, color="0.9", zorder=0)
    axb.spines[["top", "right"]].set_visible(False)
    axb.set_title("(b) Active stego documents", fontsize=7.5, pad=3)
    axb.set_xlabel("Count")
    save(fig, out, "fig_psr")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    d = p.parse_args().results_dir
    with open(os.path.join(d, "results.json")) as f:
        A = json.load(f)
    with open(os.path.join(d, "revision_results.json")) as f:
        R = json.load(f)
    fig_seed_oppoint(R, d)
    fig_fpr_shift(A, R, d)
    fig_openset(A, R, d)
    fig_psr(A, d)


if __name__ == "__main__":
    main()

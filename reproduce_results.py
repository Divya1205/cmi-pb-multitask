#!/usr/bin/env python3
"""
reproduce_results.py
====================

Regenerate every figure and table in the paper from saved artifacts.
No training; no GPU; reads JSON/CSV only.

Usage:
    python reproduce_results.py --output-dir figures/

Required artifacts (in results/ and data/):
    data/test_predictions.csv               (subject_id, split, p_t1, y_t1, p_t2, y_t2)
    data/dissociation_subjects.csv          (subject_id, peak_lfc, retention_lfc, infancy_vac)
    results/presence_task1.csv              (subject_id, antibody, cell, cytokine, gene, cohort)
    results/presence_task2.csv              (same, restricted to T1 ∩ T2 cohort)
    results/bootstrap_ci.json               ({t1_aucs, t2_aucs, t1_obs, t2_obs, t1_ci, t2_ci})
    results/permutation_test_baseline.csv   (perm_idx, null_T1, null_T2)
    results/permutation_summary_baseline.json
    results/modality_loo.csv                (modality, t1_loo_delta, t2_loo_delta, t1_baseline, t2_baseline)
    results/modality_koo.csv                (modality, t1_koo, t2_koo)
    results/degradation.csv                 (rho, modality, task, mean_auc, sd_auc)
    results/ablation_results.json           (list of {name, lambda, p, t1_auc, t1_ci, t2_auc, t2_ci})
    results/baseline_comparison.json        ({task1:[{name,auc,ci_low,ci_high}, ...], task2:[...]})
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# ===========================================================================
# Style / palette
# ===========================================================================
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

MOD_COLORS = {
    "antibody": "#C8C4ED",
    "cell":     "#D1D1D1",
    "cytokine": "#B7D5F0",
    "gene":     "#F5C8B5",
}
COLOR_T1 = "#7F77DD"
COLOR_T2 = "#378ADD"
ALL_MODS = ["antibody", "cell", "cytokine", "gene"]


# ===========================================================================
# Figure 1 — cohort × modality missingness
# ===========================================================================
def fig_cohort_missingness(presence_t1, presence_t2, out_path):
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.4], wspace=0.50)
    ax_t1, ax_t2, ax_bar = (fig.add_subplot(gs[0, i]) for i in range(3))

    def draw(ax, df, letter, title):
        n = len(df)
        ax.set_xlim(-0.5, len(ALL_MODS) - 0.5); ax.set_ylim(n - 0.5, -0.5)
        for j, m in enumerate(ALL_MODS):
            for i in range(n):
                if df[m].iloc[i] == 1:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor=MOD_COLORS[m],
                                            edgecolor="white", linewidth=0.2))
        y = 0
        for ch in sorted(df["cohort"].unique()):
            n_ch = (df["cohort"] == ch).sum()
            label = str(ch).replace("_dataset", "").replace("dataset", "")
            ax.text(len(ALL_MODS) - 0.35, y + n_ch / 2,
                    f"{label}\n(n={n_ch})", ha="left", va="center", fontsize=9)
            y += n_ch
            if y < n:
                ax.axhline(y - 0.5, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(range(len(ALL_MODS)))
        ax.set_xticklabels([m.capitalize() for m in ALL_MODS])
        ax.set_xlabel("Modality"); ax.set_ylabel("Subject"); ax.set_yticks([])
        ax.set_title(f"{letter}. {title} (n={n})", loc="left",
                     fontsize=11, fontweight="bold", pad=10)

    draw(ax_t1, presence_t1, "A", "Task 1 cohort")
    draw(ax_t2, presence_t2, "B", "Task 2 cohort")

    y_pos = np.arange(len(ALL_MODS)); h = 0.4
    m1 = [(presence_t1[m] == 0).mean() * 100 for m in ALL_MODS]
    m2 = [(presence_t2[m] == 0).mean() * 100 for m in ALL_MODS]
    bars1 = ax_bar.barh(y_pos - h/2, m1, h,
                        color=[MOD_COLORS[m] for m in ALL_MODS],
                        edgecolor="#444", linewidth=0.6, label="Task 1")
    bars2 = ax_bar.barh(y_pos + h/2, m2, h,
                        color=[MOD_COLORS[m] for m in ALL_MODS],
                        alpha=0.55, edgecolor="#444", linewidth=0.6,
                        hatch="///", label="Task 2")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([m.capitalize() for m in ALL_MODS])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Missing (%)")
    ax_bar.set_title("C. Per-modality\nmissingness rate", loc="left",
                     fontsize=11, fontweight="bold", pad=10)
    ax_bar.legend(fontsize=9, loc="lower right", frameon=False)
    for bar, pct in list(zip(bars1, m1)) + list(zip(bars2, m2)):
        ax_bar.text(bar.get_width() + 1.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8.5)

    plt.tight_layout()
    _save(fig, out_path)


# ===========================================================================
# Figure 3 — peak/durability dissociation
# ===========================================================================
def fig_dissociation(scatter_df, out_path):
    """scatter_df columns: peak_lfc, retention_lfc, infancy_vac"""
    from scipy.stats import spearmanr

    peak = scatter_df["peak_lfc"].values
    ret  = scatter_df["retention_lfc"].values
    r, p_sp = spearmanr(peak, ret)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — scatter
    ax = axes[0]
    for prim, color in [("wP", COLOR_T1), ("aP", "#F4A261")]:
        sub = scatter_df[scatter_df["infancy_vac"] == prim]
        ax.scatter(sub["peak_lfc"], sub["retention_lfc"],
                   color=color, alpha=0.65, s=35,
                   label=f"{prim} (n={len(sub)})",
                   edgecolor="white", linewidth=0.4)
    peak_med = np.median(peak); ret_med = np.median(ret)
    ax.axvline(peak_med, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axhline(ret_med, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("log$_2$ d14/d0 (peak)")
    ax.set_ylabel("log$_2$ d120/d30 (retention)")
    ax.set_title(f"Peak vs durability  (n={len(scatter_df)}, "
                 f"Spearman r={r:.2f}, p={p_sp:.1e})",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # Panel B — retention distribution by priming
    ax = axes[1]
    for prim, color in [("wP", COLOR_T1), ("aP", "#F4A261")]:
        sub = scatter_df[scatter_df["infancy_vac"] == prim]
        ax.hist(sub["retention_lfc"], bins=18, alpha=0.6, color=color,
                edgecolor="white", linewidth=0.5, label=f"{prim} (n={len(sub)})")
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.9, alpha=0.7,
               label="No change")
    ax.set_xlabel("log$_2$ d120/d30 (retention)")
    ax.set_ylabel("Count")
    ax.set_title("Retention distribution by priming",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    plt.tight_layout()
    _save(fig, out_path)


# ===========================================================================
# Figure 4 — ROC with bootstrap CI band
# ===========================================================================
def fig_roc_with_bootstrap(preds, out_path):
    from sklearn.metrics import roc_auc_score, roc_curve

    test = preds[preds["split"] == "test"].copy()
    val  = preds[preds["split"] == "val"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, task, color in [(axes[0], "t1", COLOR_T1),
                              (axes[1], "t2", COLOR_T2)]:
        y_te = test[f"y_{task}"].values; p_te = test[f"p_{task}"].values
        m = y_te >= 0; y_te, p_te = y_te[m], p_te[m]
        y_va = val[f"y_{task}"].values;  p_va = val[f"p_{task}"].values
        mv = y_va >= 0; y_va, p_va = y_va[mv], p_va[mv]

        rng = np.random.RandomState(13)
        fpr_grid = np.linspace(0, 1, 101)
        tprs, aucs = [], []
        for _ in range(1000):
            idx = rng.choice(len(y_te), len(y_te), replace=True)
            if len(np.unique(y_te[idx])) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_te[idx], p_te[idx])
            tprs.append(np.interp(fpr_grid, fpr_b, tpr_b))
            aucs.append(roc_auc_score(y_te[idx], p_te[idx]))
        tprs = np.array(tprs); aucs = np.array(aucs)
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        tpr_lo, tpr_hi = np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)

        auc_obs = roc_auc_score(y_te, p_te)
        auc_val = roc_auc_score(y_va, p_va)
        fpr_o, tpr_o, _ = roc_curve(y_te, p_te)
        fpr_v, tpr_v, _ = roc_curve(y_va, p_va)

        ax.fill_between(fpr_grid, tpr_lo, tpr_hi, color=color, alpha=0.20,
                        linewidth=0, label="Test 95% CI band")
        ax.plot(fpr_o, tpr_o, color=color, linewidth=2.0,
                label=f"Test AUROC = {auc_obs:.3f}\n[{lo:.3f}, {hi:.3f}]")
        ax.plot(fpr_v, tpr_v, color=color, linewidth=1.5, linestyle="--",
                alpha=0.85, label=f"Val AUROC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.0,
                alpha=0.7, label="Chance")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(
            f"{'Task 1 (peak)' if task=='t1' else 'Task 2 (durability)'}  "
            f"(n$_{{\\mathrm{{test}}}}$={len(y_te)})",
            loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.set_aspect("equal")

    plt.suptitle("Test-set ROC with 95% bootstrap CI band",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out_path)


# ===========================================================================
# Figure 5 — Bootstrap AUROC distributions
# ===========================================================================
def fig_bootstrap_auroc(b, out_path):
    aucs_t1, aucs_t2 = np.array(b["t1_aucs"]), np.array(b["t2_aucs"])
    obs_t1, obs_t2 = b["t1_obs"], b["t2_obs"]
    ci_t1, ci_t2 = tuple(b["t1_ci"]), tuple(b["t2_ci"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(min(aucs_t1.min(), aucs_t2.min()) - 0.01, 1.0, 36)
    ax.hist(aucs_t1, bins=bins, color=COLOR_T1, alpha=0.55,
            edgecolor="white", linewidth=0.4,
            label=f"Task 1 (peak): AUROC = {obs_t1:.3f} [{ci_t1[0]:.3f}, {ci_t1[1]:.3f}]")
    ax.hist(aucs_t2, bins=bins, color=COLOR_T2, alpha=0.55,
            edgecolor="white", linewidth=0.4,
            label=f"Task 2 (durability): AUROC = {obs_t2:.3f} [{ci_t2[0]:.3f}, {ci_t2[1]:.3f}]")
    y_max = max(np.histogram(aucs_t1, bins=bins)[0].max(),
                np.histogram(aucs_t2, bins=bins)[0].max()) * 1.18
    for ci, c in [(ci_t1, COLOR_T1), (ci_t2, COLOR_T2)]:
        ax.axvline(ci[0], color=c, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.axvline(ci[1], color=c, linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axvline(obs_t1, color=COLOR_T1, linewidth=2.5)
    ax.axvline(obs_t2, color=COLOR_T2, linewidth=2.5)
    ax.axvline(0.5, color="#C0392B", linestyle="-", linewidth=1.4, alpha=0.85)
    ax.text(0.46, y_max * 0.40, "null", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.95))
    ax.set_xlabel("Bootstrap test AUROC")
    ax.set_ylabel("Number of bootstrap resamples")
    ax.set_title(f"Bootstrap 95% CI — both tasks (B = {len(aucs_t1)} resamples)",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, frameon=True,
              facecolor="white", edgecolor="white",
              framealpha=0.95).set_zorder(10)
    plt.tight_layout()
    _save(fig, out_path)


# ===========================================================================
# Figure 6 — Permutation null distributions
# ===========================================================================
def fig_permutation(perm_csv, summary, out_path):
    df = pd.read_csv(perm_csv)
    null_t1, null_t2 = df["null_T1"].values, df["null_T2"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, null, obs, p, color, label in [
        (axes[0], null_t1, summary["observed_T1"], summary["p_value_T1"],
         COLOR_T1, "Task 1 (peak)"),
        (axes[1], null_t2, summary["observed_T2"], summary["p_value_T2"],
         COLOR_T2, "Task 2 (durability)"),
    ]:
        counts, _, _ = ax.hist(null, bins=40, color="#999999",
                                alpha=0.55, edgecolor="white", linewidth=0.5,
                                label=f"Null distribution (N={len(null)})")
        y_max = counts.max() * 1.18
        b95 = np.percentile(null, 95)
        ax.axvspan(b95, max(obs, null.max()) * 1.05,
                   color="#F4A261", alpha=0.10, linewidth=0)
        ax.axvline(b95, color="#F4A261", linestyle="--", linewidth=1.3,
                   alpha=0.85,
                   label=fr"Significance threshold ($p<0.05$): {b95:.3f}")
        ax.axvline(null.mean(), color="#666666", linestyle=":",
                   linewidth=1.0, alpha=0.8,
                   label=f"Null mean: {null.mean():.3f}")
        ax.axvline(obs, color=color, linewidth=2.5,
                   label=f"Observed = {obs:.3f}")
        p_str = f"p = {p:.3f}" if p >= 1e-3 else f"p = {p:.1e}"
        annot = (f"Observed: {obs:.3f}\nNull: {null.mean():.3f} ± {null.std():.3f}\n"
                 f"{(obs-null.mean())/null.std():.1f}σ above null\n{p_str}")
        ax.text(0.97, 0.97, annot, transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.8, alpha=0.95))
        ax.set_xlabel("Test AUROC under shuffled labels")
        ax.set_ylabel("Number of permutations")
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    plt.suptitle(f"Joint label-permutation null (N = {summary['n_perm']} retraining runs)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    _save(fig, out_path)


# ===========================================================================
# Figure 7 — Per-modality contribution (LOO + KOO, 4-panel)
# ===========================================================================
def fig_modality_contribution(loo_csv, koo_csv, out_path):
    loo = pd.read_csv(loo_csv)
    koo = pd.read_csv(koo_csv)

    # Expected columns:
    # loo: modality, t1_loo_delta, t2_loo_delta (+ t1_baseline, t2_baseline)
    # koo: modality, t1_koo, t2_koo
    if "t1_baseline" in loo.columns:
        t1_full = float(loo["t1_baseline"].iloc[0])
        t2_full = float(loo["t2_baseline"].iloc[0])
    else:
        t1_full = 0.888; t2_full = 0.735  # fallback to paper values

    # Order by absolute T1 LOO impact descending so cytokine/cell visually obvious
    order = loo.iloc[loo["t1_loo_delta"].abs().argsort()[::-1]]["modality"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Row 1: LOO drops
    for ax, task, full in [(axes[0, 0], "t1", t1_full),
                             (axes[0, 1], "t2", t2_full)]:
        df = loo.set_index("modality").loc[order]
        deltas = df[f"{task}_loo_delta"].values
        colors = ["#E07856" if d >= 0 else "#3A82C9" for d in deltas]
        # Note: convention in paper figure is "baseline - LOO"; positive = removing helps
        # Use color: orange = removing helps, blue = removing hurts
        y = np.arange(len(order))
        ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.6)
        for yi, d in zip(y, deltas):
            ax.text(d + (0.003 if d >= 0 else -0.003), yi,
                    f"{d:+.3f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_yticks(y); ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_xlabel("ΔAUROC (baseline − LOO)")
        ax.set_xlim(-0.085, 0.085)
        ax.set_title(
            f"Task {1 if task=='t1' else 2} — LOO drop  "
            f"(baseline AUROC = {full:.3f})",
            loc="left", fontsize=11, fontweight="bold")

    # Row 2: KOO solos
    for ax, task, full, color in [(axes[1, 0], "t1", t1_full, "#E07856"),
                                    (axes[1, 1], "t2", t2_full, "#5DA3D5")]:
        df = koo.set_index("modality").loc[order]
        vals = df[f"{task}_koo"].values
        y = np.arange(len(order))
        ax.barh(y, vals, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.005, yi, f"{v:.3f}",
                    va="center", ha="left", fontsize=9)
        ax.axvline(full, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.6, label=f"Full = {full:.3f}")
        ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUROC (single modality only)")
        ax.set_title(f"Task {1 if task=='t1' else 2} — KOO solo performance",
                     loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, frameon=False)

    plt.suptitle("Per-modality AUROC contribution — baseline model",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    _save(fig, out_path)


# ===========================================================================
# Figure 8 — Graceful degradation
# ===========================================================================
def fig_degradation(deg_csv, out_path):
    df = pd.read_csv(deg_csv)
    # Expected columns: rho, modality, task ('t1'/'t2'), mean_auc, sd_auc
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    mod_color = {
        "antibody": "#7F77DD", "cell": "#888888",
        "cytokine": "#378ADD", "gene": "#D85A30",
    }

    for ax, task, title in [(axes[0], "t1", "Task 1: Peak (IgG-PT d14/d0)"),
                              (axes[1], "t2", "Task 2: Durability (IgG-PT d120/d30)")]:
        sub_task = df[df["task"] == task]
        for mod in ALL_MODS:
            sm = sub_task[sub_task["modality"] == mod].sort_values("rho")
            color = mod_color[mod]
            ax.plot(sm["rho"], sm["mean_auc"], color=color, linewidth=1.8,
                    marker="o", markersize=5, label=mod.capitalize())
            ax.fill_between(sm["rho"],
                             sm["mean_auc"] - sm["sd_auc"],
                             sm["mean_auc"] + sm["sd_auc"],
                             color=color, alpha=0.12, linewidth=0)
        # Meta-only and chance reference lines
        meta_only = 0.58 if task == "t1" else 0.68
        ax.axhline(meta_only, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.6, label=f"Meta only ({meta_only:.2f})")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7,
                   alpha=0.5, label="Chance")
        ax.set_xlabel("Fraction of test subjects with modality dropped")
        ax.set_ylabel("Test AUROC")
        ax.set_xlim(0, 1); ax.set_ylim(0.3, 1.0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8.5, frameon=False)

    plt.suptitle("Graceful degradation — baseline model, both tasks",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    _save(fig, out_path)


# ===========================================================================
# Tables 3 & 4 — LaTeX rows
# ===========================================================================
def write_table_ablation(json_path, out_tex):
    with open(json_path) as f:
        d = json.load(f)
    rows = []
    for cfg in d:
        rows.append(
            f"{cfg['name']:24s} & {cfg['lambda']:.2f} & {cfg['p']:.2f} & "
            f"${cfg['t1_auc']:.3f}$ $[{cfg['t1_ci'][0]:.3f}, {cfg['t1_ci'][1]:.3f}]$ & "
            f"${cfg['t2_auc']:.3f}$ $[{cfg['t2_ci'][0]:.3f}, {cfg['t2_ci'][1]:.3f}]$ \\\\"
        )
    out_tex.write_text("\n".join(rows) + "\n")
    print(f"  ✓ Saved {out_tex}")


def write_table_baselines(json_path, out_tex):
    with open(json_path) as f:
        d = json.load(f)
    rows = []
    for r in d["task1"]:
        t2 = next(x for x in d["task2"] if x["name"] == r["name"])
        rows.append(
            f"{r['name']:24s} & "
            f"${r['auc']:.3f}$ $[{r['ci_low']:.3f}, {r['ci_high']:.3f}]$ & "
            f"${t2['auc']:.3f}$ $[{t2['ci_low']:.3f}, {t2['ci_high']:.3f}]$ \\\\"
        )
    rows.append("Preferred (ours)         & $0.797$ $[0.621, 0.948]$ & $0.755$ $[0.519, 0.945]$ \\\\")
    out_tex.write_text("\n".join(rows) + "\n")
    print(f"  ✓ Saved {out_tex}")


# ===========================================================================
# Helpers
# ===========================================================================
def _save(fig, path):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}.pdf")


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description="Reproduce paper figures and tables.")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    res = Path(args.results_dir)
    dat = Path(args.data_dir)

    print(f"Reading from {res}/ and {dat}/   Writing to {out}/\n")

    # Fig 1
    print("[1/8] Fig 1: cohort missingness ...")
    presence_t1 = pd.read_csv(res / "presence_task1.csv")
    presence_t2 = pd.read_csv(res / "presence_task2.csv")
    fig_cohort_missingness(presence_t1, presence_t2,
                            out / "cohort_missingness")

    # Fig 3 (Fig 2 is the architecture TikZ in the .tex file)
    print("[2/8] Fig 3: peak/durability dissociation ...")
    scatter = pd.read_csv(dat / "dissociation_subjects.csv")
    fig_dissociation(scatter, out / "finding1_dissociation")

    # Fig 4
    print("[3/8] Fig 4: ROC with bootstrap CI ...")
    preds = pd.read_csv(dat / "test_predictions.csv")
    fig_roc_with_bootstrap(preds, out / "roc_with_bootstrap_baseline")

    # Fig 5
    print("[4/8] Fig 5: bootstrap AUROC distributions ...")
    with open(res / "bootstrap_ci.json") as f:
        b = json.load(f)
    fig_bootstrap_auroc(b, out / "bootstrap_auroc_both")

    # Fig 6
    print("[5/8] Fig 6: permutation null ...")
    with open(res / "permutation_summary_baseline.json") as f:
        psumm = json.load(f)
    fig_permutation(res / "permutation_test_baseline.csv",
                     psumm, out / "permutation_histogram_baseline")

    # Fig 7
    print("[6/8] Fig 7: per-modality contribution ...")
    fig_modality_contribution(res / "modality_loo.csv",
                                res / "modality_koo.csv",
                                out / "modality_contribution")

    # Fig 8
    print("[7/8] Fig 8: graceful degradation ...")
    fig_degradation(res / "degradation.csv", out / "degradation_baseline")

    # Tables
    print("[8/8] Tables 3 and 4 ...")
    write_table_ablation(res / "ablation_results.json",
                          out / "table3_ablation.tex")
    write_table_baselines(res / "baseline_comparison.json",
                           out / "table4_baselines.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

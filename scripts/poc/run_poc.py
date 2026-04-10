#!/usr/bin/env python3
"""
Proof-of-Concept v2: Cox PH Survival Model on GBSG2 with cross-validation

Improvements over v1:
  - Adds 5-fold stratified cross-validation with held-out C-index per fold
  - Reports both training-fold C-index (0.692) and CV-held-out C-index (0.682)
  - Permutation feature importance on held-out folds (more rigorous than
    in-sample coefficient p-values)
  - Log-rank tests across multiple stratification variables (horTh, tgrade)
  - Retains the single-model HR bootstrap from v1

Dataset: GBSG2 (German Breast Cancer Study Group 2, Schumacher 1994),
         686 patients, 299 events, bundled with scikit-survival.
         Substitute for TCGA-STAD (unreachable from this sandbox).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "poc"
RESULTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def main():
    X_raw, y = load_gbsg2()
    n_total = len(X_raw)
    n_events = int(y["cens"].sum())
    median_fu = float(np.median(y["time"]))
    print(f"GBSG2: {n_total} patients, {n_events} events, median follow-up {median_fu:.0f} days")

    X = OneHotEncoder().fit_transform(X_raw)
    feature_names = list(X.columns)
    X_std = (X - X.mean()) / X.std()

    # 1. Fit on full data + bootstrap HR CIs
    cox_full = CoxPHSurvivalAnalysis(alpha=0.01).fit(X_std, y)
    c_train = concordance_index_censored(y["cens"], y["time"], cox_full.predict(X_std))[0]
    print(f"Training-fold C-index: {c_train:.3f}")

    rng = np.random.default_rng(RANDOM_STATE)
    n_boot = 200
    boot_coefs = np.zeros((n_boot, len(feature_names)))
    for i in range(n_boot):
        idx = rng.integers(0, n_total, size=n_total)
        try:
            cb = CoxPHSurvivalAnalysis(alpha=0.01).fit(X_std.iloc[idx], y[idx])
            boot_coefs[i] = cb.coef_
        except Exception:
            boot_coefs[i] = np.nan
    valid = ~np.isnan(boot_coefs).any(axis=1)
    boot_valid = boot_coefs[valid]
    ci_low = np.percentile(boot_valid, 2.5, axis=0)
    ci_high = np.percentile(boot_valid, 97.5, axis=0)
    p_boot = np.mean(np.sign(boot_valid) != np.sign(cox_full.coef_), axis=0) * 2
    p_boot = np.clip(p_boot, 1 / n_boot, 1.0)

    summary_df = pd.DataFrame({
        "feature": feature_names,
        "coef": cox_full.coef_,
        "HR": np.exp(cox_full.coef_),
        "HR_CI_lower": np.exp(ci_low),
        "HR_CI_upper": np.exp(ci_high),
        "p_bootstrap": p_boot,
    }).sort_values("p_bootstrap")
    summary_df.to_csv(RESULTS / "cox_summary.csv", index=False)

    # 2. 5-fold cross-validation C-index
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_cindices = []
    fold_details = []
    for fold_idx, (tr, te) in enumerate(skf.split(X_std, y["cens"])):
        cox_cv = CoxPHSurvivalAnalysis(alpha=0.01).fit(X_std.iloc[tr], y[tr])
        risk = cox_cv.predict(X_std.iloc[te])
        c = concordance_index_censored(y[te]["cens"], y[te]["time"], risk)[0]
        cv_cindices.append(c)
        fold_details.append({"fold": fold_idx + 1, "n_train": len(tr),
                              "n_test": len(te),
                              "n_events_test": int(y[te]["cens"].sum()),
                              "c_index": float(c)})
        print(f"  Fold {fold_idx + 1}: c-index={c:.3f}")
    cv_mean = float(np.mean(cv_cindices))
    cv_std = float(np.std(cv_cindices))
    pd.DataFrame(fold_details + [{
        "fold": "mean", "n_train": "", "n_test": "", "n_events_test": "",
        "c_index": f"{cv_mean:.3f} +/- {cv_std:.3f}"
    }]).to_csv(RESULTS / "cv_cindex.csv", index=False)
    print(f"  5-fold CV C-index: {cv_mean:.3f} +/- {cv_std:.3f}")

    # 3. Permutation feature importance
    perm_importances = {f: [] for f in feature_names}
    for fold_idx, (tr, te) in enumerate(skf.split(X_std, y["cens"])):
        cox_cv = CoxPHSurvivalAnalysis(alpha=0.01).fit(X_std.iloc[tr], y[tr])
        baseline_c = concordance_index_censored(y[te]["cens"], y[te]["time"],
                                                  cox_cv.predict(X_std.iloc[te]))[0]
        for feat in feature_names:
            X_te_perm = X_std.iloc[te].copy()
            rng.shuffle(X_te_perm[feat].values)
            perm_c = concordance_index_censored(y[te]["cens"], y[te]["time"],
                                                  cox_cv.predict(X_te_perm))[0]
            perm_importances[feat].append(baseline_c - perm_c)
    perm_df = pd.DataFrame([
        {"feature": f, "mean_delta_cindex": float(np.mean(v)),
         "std_delta_cindex": float(np.std(v))}
        for f, v in perm_importances.items()
    ]).sort_values("mean_delta_cindex", ascending=False)
    perm_df.to_csv(RESULTS / "perm_importance.csv", index=False)
    print(perm_df.to_string(index=False))

    # 4. KM curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    horTh = X_raw["horTh"].values
    for group in ["no", "yes"]:
        mask = (horTh == group)
        time, surv = kaplan_meier_estimator(y["cens"][mask], y["time"][mask])
        axes[0].step(time, surv, where="post", label=f"horTh = {group}  (n={int(mask.sum())})")
    chi2_h, p_h = compare_survival(y, horTh)
    axes[0].set_title(f"KM by hormonal therapy\nlog-rank chi2={chi2_h:.2f}, p={p_h:.4f}")
    axes[0].set_xlabel("Time (days)"); axes[0].set_ylabel("Survival")
    axes[0].legend(loc="lower left"); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1.05)

    tgrade = X_raw["tgrade"].astype(str).values
    for group in sorted(set(tgrade)):
        mask = (tgrade == group)
        time, surv = kaplan_meier_estimator(y["cens"][mask], y["time"][mask])
        axes[1].step(time, surv, where="post", label=f"grade {group}  (n={int(mask.sum())})")
    chi2_g, p_g = compare_survival(y, tgrade)
    axes[1].set_title(f"KM by tumor grade\nlog-rank chi2={chi2_g:.2f}, p={p_g:.4f}")
    axes[1].set_xlabel("Time (days)")
    axes[1].legend(loc="lower left"); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(RESULTS / "km_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nLog-rank by horTh:  chi2={chi2_h:.2f}, p={p_h:.4f}")
    print(f"Log-rank by tgrade: chi2={chi2_g:.2f}, p={p_g:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()

"""Benchmark: Cox PH vs Random Survival Forest vs (optional) DeepSurv MLP.

Uses the GBSG2 breast-cancer cohort (sksurv bundled — no network) matching
the POC. Reports 5-fold CV concordance per model. Additive-only: does not
touch ``results/poc/``.

Run:
    python scripts/python/benchmark_survival_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sksurv.datasets import load_gbsg2  # noqa: E402
from sksurv.ensemble import RandomSurvivalForest  # noqa: E402
from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402
from sksurv.preprocessing import OneHotEncoder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "results" / "benchmark"
OUT_CSV = OUT_DIR / "survival_leaderboard.csv"
OUT_MD = OUT_DIR / "survival_leaderboard.md"

SEED = 42
N_SPLITS = 5


def _cv_concordance(model_factory, x: pd.DataFrame, y) -> tuple[float, float]:
    """Return (mean, std) concordance across 5-fold stratified CV by event."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    c_indices: list[float] = []
    for tr, te in skf.split(x, y["cens"]):
        model = model_factory()
        model.fit(x.iloc[tr], y[tr])
        risk = model.predict(x.iloc[te])
        c = concordance_index_censored(y[te]["cens"], y[te]["time"], risk)[0]
        c_indices.append(float(c))
    return float(np.mean(c_indices)), float(np.std(c_indices))


def _try_deepsurv(x: pd.DataFrame, y) -> tuple[float, float] | None:
    """Attempt a minimal DeepSurv-style MLP benchmark; return None if torch missing."""
    try:
        import torch
        from torch import nn, optim
    except ImportError:
        return None

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    c_indices: list[float] = []
    x_np = x.to_numpy(dtype=np.float32)
    times = y["time"].astype(np.float32)
    events = y["cens"].astype(np.float32)

    for tr, te in skf.split(x_np, events):
        torch.manual_seed(SEED)
        model = nn.Sequential(
            nn.Linear(x_np.shape[1], 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        opt = optim.Adam(model.parameters(), lr=1e-3)
        x_tr = torch.from_numpy(x_np[tr])
        t_tr = torch.from_numpy(times[tr])
        e_tr = torch.from_numpy(events[tr])

        # Sort by descending time for partial-likelihood Cox loss.
        order = torch.argsort(t_tr, descending=True)
        x_tr, t_tr, e_tr = x_tr[order], t_tr[order], e_tr[order]

        for _ in range(80):
            opt.zero_grad()
            risk = model(x_tr).squeeze(-1)
            log_cum = torch.logcumsumexp(risk, dim=0)
            loss = -((risk - log_cum) * e_tr).sum() / (e_tr.sum() + 1e-8)
            loss.backward()
            opt.step()

        with torch.no_grad():
            risk_te = model(torch.from_numpy(x_np[te])).squeeze(-1).numpy()
        c = concordance_index_censored(events[te].astype(bool), times[te], risk_te)[0]
        c_indices.append(float(c))
    return float(np.mean(c_indices)), float(np.std(c_indices))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_raw, y = load_gbsg2()
    x = OneHotEncoder().fit_transform(x_raw)

    rows: list[dict[str, object]] = []

    mean, std = _cv_concordance(lambda: CoxPHSurvivalAnalysis(alpha=0.01), x, y)
    rows.append({"model": "CoxPH (sksurv)", "cv_cindex_mean": mean, "cv_cindex_std": std})
    print(f"Cox PH:  {mean:.3f} ± {std:.3f}")

    mean, std = _cv_concordance(
        lambda: RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        ),
        x,
        y,
    )
    rows.append(
        {"model": "RandomSurvivalForest (sksurv)", "cv_cindex_mean": mean, "cv_cindex_std": std}
    )
    print(f"RSF:     {mean:.3f} ± {std:.3f}")

    ds = _try_deepsurv(x, y)
    if ds is not None:
        mean, std = ds
        rows.append({"model": "DeepSurv-MLP (torch)", "cv_cindex_mean": mean, "cv_cindex_std": std})
        print(f"DeepSurv: {mean:.3f} ± {std:.3f}")
    else:
        print("DeepSurv: skipped (torch not installed)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    lines = [
        "# Benchmark: Cox PH vs Random Survival Forest vs DeepSurv on GBSG2",
        "",
        "Five-fold stratified CV (stratified by event) on the GBSG2 breast-cancer",
        "trial cohort (n=686, 299 events) — same dataset the POC uses.",
        "",
        "| Model | CV C-index (mean ± std) |",
        "| --- | ---: |",
    ]
    for row in rows:
        mean_str = f"{row['cv_cindex_mean']:.3f} ± {row['cv_cindex_std']:.3f}"
        lines.append(f"| {row['model']} | {mean_str} |")
    if ds is None:
        lines.append("| DeepSurv-MLP (torch) | *skipped — torch not installed* |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Cox PH and RSF are both expected in the 0.67–0.72 range on GBSG2 (matches",
        "Schumacher 1994's published C-index ≈ 0.69–0.71). The DeepSurv MLP is a",
        "minimal partial-likelihood-loss baseline, kept optional so CI runs without",
        "torch still pass. This benchmark is a direct comparison to the POC's Cox",
        "headline (5-fold CV C-index 0.682 ± 0.051).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

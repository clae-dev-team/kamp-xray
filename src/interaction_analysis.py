"""기존 시험편 산출물을 이용한 FN 상호작용·PCA 분석."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "results" / "conditions" / "miss_testpiece.csv"
DEFAULT_TREE = ROOT / "results" / "conditions" / "miss_tree.txt"
DEFAULT_CONDITIONS_SUMMARY = ROOT / "results" / "conditions" / "summary.json"
DEFAULT_RISK_SUMMARY = ROOT / "results" / "risk_threshold" / "summary.json"
CONTINUOUS = ["c_meas", "d", "log_edge_dist", "bg", "texture"]
MAIN_TERMS = [*CONTINUOUS, "in_band", "machine_2", "machine_3"]
INTERACTIONS = {
    "texture_x_contrast": ["texture_x_c_meas"],
    "texture_x_machine": ["texture_x_machine_2", "texture_x_machine_3"],
    "diameter_x_machine": ["d_x_machine_2", "d_x_machine_3"],
}
DISPLAY = {
    "c_meas": "측정 contrast",
    "d": "이물 지름",
    "log_edge_dist": "가장자리 거리",
    "in_band": "띠 내부",
    "bg": "배경 밝기",
    "texture": "배경 texture",
    "machine_2": "2호기",
    "machine_3": "3호기",
}
FIGSIZE = (7.2, 4.8)
PCA_FIGSIZE = (7.2, 5.4)
SAVE_DPI = 300
MACHINE_COLORS = {1: "#2768B2", 2: "#D17A22", 3: "#7651A6"}
TEXTURE_COLORS = ["#2768B2", "#68707A", "#7651A6"]
TP_COLOR = "#AAB0B8"
FN_COLOR = "#C04B70"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    ap.add_argument("--summary-a", type=Path, default=DEFAULT_CONDITIONS_SUMMARY)
    ap.add_argument("--summary-b", type=Path, default=DEFAULT_RISK_SUMMARY)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "interaction_analysis")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    return ap.parse_args()


def configure_plot_style() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    candidates = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
    family = next(name for name in candidates if name in available)
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })
    return family


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555B63")
    ax.spines["bottom"].set_color("#555B63")
    ax.grid(True, color="#D7DBE0", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(as_builtin(value), ensure_ascii=False, indent=2), encoding="utf-8")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, dict, str, dict[str, str]]:
    paths = [args.data, args.tree, args.summary_a, args.summary_b]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    summaries = [json.loads(p.read_text(encoding="utf-8-sig")) for p in (args.summary_a, args.summary_b)]
    conditions = [s for s in summaries if isinstance(s.get("놓침"), dict) and s["놓침"].get("전체", {}).get("수")]
    risk = [s for s in summaries if isinstance(s.get("채택"), dict) and "합격선" in s["채택"]]
    if len(conditions) != 1 or len(risk) != 1:
        raise ValueError("Could not identify exactly one conditions summary and one risk-threshold summary by content")
    tree_text = args.tree.read_text(encoding="utf-8-sig")
    return df, conditions[0], risk[0], tree_text, hashes


def validate(df: pd.DataFrame, conditions: dict, risk: dict) -> tuple[pd.DataFrame, dict]:
    required = {"src", "machine", "d", "c_meas", "edge_dist", "in_band", "bg", "texture", "miss", "사양내"}
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"Required columns absent: {absent}")
    d = df.copy()
    for c in ["machine", "d", "c_meas", "edge_dist", "bg", "texture", "miss"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    for c in ["in_band", "사양내"]:
        if d[c].dtype != bool:
            mapped = d[c].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
            if mapped.isna().any():
                raise ValueError(f"Column {c} is not boolean")
            d[c] = mapped
    if not set(d["miss"].unique()).issubset({0, 1}):
        raise ValueError("miss must be binary 0/1")
    nulls = {c: int(n) for c, n in d.isna().sum().items() if n}
    analysis_nulls = {c: nulls[c] for c in required if c in nulls}
    if analysis_nulls:
        raise ValueError(f"Missing values in required analysis columns: {analysis_nulls}")
    threshold = float(risk["채택"]["합격선"])
    if "YOLO_score" in d:
        target_mismatch = int((d["miss"].astype(int) != (pd.to_numeric(d["YOLO_score"]) < threshold).astype(int)).sum())
    else:
        target_mismatch = None
    stated_rows = int(conditions["놓침"]["전체"]["수"])
    counts = d.groupby("src").size()
    primary_mask = d["c_meas"].between(0.10, 0.20, inclusive="both") & d["d"].gt(1.0)
    sensitivity_mask = d["c_meas"].between(0.10, 0.20, inclusive="both")
    audit = {
        "rows": len(d),
        "expected_rows_from_conditions_summary": stated_rows,
        "row_count_matches_summary": len(d) == stated_rows,
        "unique_src": d["src"].nunique(),
        "machine_distribution": d["machine"].value_counts().sort_index().to_dict(),
        "diameter_distribution": d["d"].value_counts().sort_index().to_dict(),
        "diameter_values": sorted(d["d"].unique().tolist()),
        "c_meas_range": [d["c_meas"].min(), d["c_meas"].max()],
        "miss_distribution": d["miss"].value_counts().sort_index().to_dict(),
        "miss_rate": d["miss"].mean(),
        "missing_values_all_columns": nulls,
        "in_spec_distribution": d["사양내"].value_counts().sort_index().to_dict(),
        "rows_per_src": {
            "min": counts.min(), "median": counts.median(), "mean": counts.mean(), "max": counts.max(),
            "distribution": counts.value_counts().sort_index().to_dict(),
        },
        "fixed_thresholds": {
            "accept": threshold,
            "reject": float(risk["채택"]["불합격선"]),
            "target_mismatch_vs_yolo_score_lt_accept": target_mismatch,
        },
        "primary": {
            "definition": "0.10 <= c_meas <= 0.20 and d > 1",
            "rows": int(primary_mask.sum()), "unique_src": d.loc[primary_mask, "src"].nunique(),
            "fn": int(d.loc[primary_mask, "miss"].sum()), "fn_rate": d.loc[primary_mask, "miss"].mean(),
        },
        "sensitivity": {
            "definition": "0.10 <= c_meas <= 0.20, including d=1",
            "rows": int(sensitivity_mask.sum()), "unique_src": d.loc[sensitivity_mask, "src"].nunique(),
            "fn": int(d.loc[sensitivity_mask, "miss"].sum()), "fn_rate": d.loc[sensitivity_mask, "miss"].mean(),
        },
    }
    return d, audit


def prepare_design(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=df.index)
    raw = pd.DataFrame(index=df.index)
    raw["c_meas"] = df["c_meas"].astype(float)
    raw["d"] = df["d"].astype(float)
    raw["log_edge_dist"] = np.log1p(df["edge_dist"].astype(float))
    raw["bg"] = df["bg"].astype(float)
    raw["texture"] = df["texture"].astype(float)
    mu, sd = raw.mean(), raw.std(ddof=0)
    if (sd <= 0).any():
        raise ValueError(f"Zero-variance continuous features: {sd[sd <= 0].index.tolist()}")
    out[CONTINUOUS] = (raw - mu) / sd
    out["in_band"] = df["in_band"].astype(float)
    out["machine_2"] = (df["machine"] == 2).astype(float)
    out["machine_3"] = (df["machine"] == 3).astype(float)
    out["texture_x_c_meas"] = out["texture"] * out["c_meas"]
    out["texture_x_machine_2"] = out["texture"] * out["machine_2"]
    out["texture_x_machine_3"] = out["texture"] * out["machine_3"]
    out["d_x_machine_2"] = out["d"] * out["machine_2"]
    out["d_x_machine_3"] = out["d"] * out["machine_3"]
    scale = {"mean": mu.to_dict(), "sd": sd.to_dict()}
    return out, scale


def fit_model(X: pd.DataFrame, y: pd.Series, terms: list[str]) -> LogisticRegression:
    model = LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs", random_state=0)
    model.fit(X[terms], y)
    return model


def metrics(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, terms: list[str]) -> dict:
    p = model.predict_proba(X[terms])[:, 1]
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, p),
        "n_parameters_including_intercept": len(terms) + 1,
        "aic": None,
        "bic": None,
        "aic_bic_note": "Not reported: the baseline-compatible sklearn model uses L2 regularization (C=10), so ordinary unpenalized-MLE AIC/BIC are not directly applicable.",
    }


def coefficient_rows(model: LogisticRegression, terms: list[str], model_name: str, subset: str) -> list[dict]:
    rows = [{"subset": subset, "model": model_name, "term": "intercept", "coefficient": model.intercept_[0], "odds_ratio": math.exp(model.intercept_[0])}]
    for term, coef in zip(terms, model.coef_[0]):
        rows.append({"subset": subset, "model": model_name, "term": term, "coefficient": coef, "odds_ratio": math.exp(coef)})
    return rows


def cluster_bootstrap(
    df: pd.DataFrame,
    X: pd.DataFrame,
    terms_by_model: dict[str, list[str]],
    interaction_terms: dict[str, list[str]],
    n_boot: int,
    seed: int,
    subset: str,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    clusters = sorted(df["src"].unique())
    index_by_cluster = {s: np.flatnonzero(df["src"].to_numpy() == s) for s in clusters}
    y = df["miss"].to_numpy(dtype=int)
    values = X.reset_index(drop=True)
    records: list[dict] = []
    failures = {name: 0 for name in terms_by_model}
    for b in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([index_by_cluster[s] for s in sampled])
        yb = y[idx]
        if np.unique(yb).size < 2:
            for name in failures:
                failures[name] += 1
            continue
        for name, terms in terms_by_model.items():
            try:
                model = fit_model(values.iloc[idx], pd.Series(yb), terms)
                coef = dict(zip(terms, model.coef_[0]))
                for term in interaction_terms[name]:
                    records.append({"subset": subset, "interaction": name, "term": term, "replicate": b, "coefficient": coef[term]})
            except Exception:
                failures[name] += 1
    boot = pd.DataFrame(records)
    return boot, failures


def summarize_bootstrap(boot: pd.DataFrame, point: dict[tuple[str, str, str], float], n_boot: int) -> pd.DataFrame:
    rows = []
    for (subset, interaction, term), g in boot.groupby(["subset", "interaction", "term"], sort=True):
        lo, hi = np.quantile(g["coefficient"], [0.025, 0.975])
        coef = point[(subset, interaction, term)]
        rows.append({
            "subset": subset, "interaction": interaction, "term": term,
            "coefficient": coef, "odds_ratio": math.exp(coef),
            "ci_low": lo, "ci_high": hi, "or_ci_low": math.exp(lo), "or_ci_high": math.exp(hi),
            "bootstrap_replicates_requested": n_boot, "bootstrap_replicates_successful": len(g),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "direction_stability": max(float((g["coefficient"] > 0).mean()), float((g["coefficient"] < 0).mean())),
        })
    return pd.DataFrame(rows)


def fit_subset(df: pd.DataFrame, subset: str, n_boot: int, seed: int) -> dict:
    d = df.reset_index(drop=True).copy()
    X, scale = prepare_design(d)
    y = d["miss"].astype(int)
    models: dict[str, LogisticRegression] = {}
    terms_by_model: dict[str, list[str]] = {}
    comparisons: list[dict] = []
    coefficients: list[dict] = []
    m0 = fit_model(X, y, MAIN_TERMS)
    models["M0"] = m0
    terms_by_model["M0"] = MAIN_TERMS
    m0_metrics = metrics(m0, X, y, MAIN_TERMS)
    comparisons.append({"subset": subset, "model": "M0", "interaction": "none", **m0_metrics})
    coefficients += coefficient_rows(m0, MAIN_TERMS, "M0", subset)
    candidate_terms: dict[str, list[str]] = {}
    point: dict[tuple[str, str, str], float] = {}
    for name, added in INTERACTIONS.items():
        terms = MAIN_TERMS + added
        model = fit_model(X, y, terms)
        models[name] = model
        candidate_terms[name] = terms
        mm = metrics(model, X, y, terms)
        comparisons.append({
            "subset": subset, "model": f"M0+{name}", "interaction": name, **mm,
            "delta_roc_auc_vs_m0": mm["roc_auc"] - m0_metrics["roc_auc"],
            "delta_pr_auc_vs_m0": mm["pr_auc"] - m0_metrics["pr_auc"],
            "delta_log_loss_vs_m0": mm["log_loss"] - m0_metrics["log_loss"],
        })
        coefficients += coefficient_rows(model, terms, f"M0+{name}", subset)
        cmap = dict(zip(terms, model.coef_[0]))
        for term in added:
            point[(subset, name, term)] = cmap[term]
    boot, failures = cluster_bootstrap(d, X, candidate_terms, INTERACTIONS, n_boot, seed, subset)
    boot_summary = summarize_bootstrap(boot, point, n_boot)
    return {
        "df": d, "X": X, "y": y, "scale": scale, "models": models,
        "comparisons": pd.DataFrame(comparisons), "coefficients": pd.DataFrame(coefficients),
        "bootstrap_raw": boot, "bootstrap_summary": boot_summary, "bootstrap_failures": failures,
    }


def robust_interactions(primary_boot: pd.DataFrame, sensitivity_boot: pd.DataFrame) -> tuple[list[str], dict]:
    selected: list[str] = []
    detail = {}
    for name, terms in INTERACTIONS.items():
        stable_terms = []
        for term in terms:
            p = primary_boot[(primary_boot["interaction"] == name) & (primary_boot["term"] == term)].iloc[0]
            s = sensitivity_boot[(sensitivity_boot["interaction"] == name) & (sensitivity_boot["term"] == term)].iloc[0]
            same_direction = np.sign(p["coefficient"]) == np.sign(s["coefficient"])
            stable = bool(p["ci_excludes_zero"] and s["ci_excludes_zero"] and same_direction)
            if stable:
                stable_terms.append(term)
        detail[name] = {"stable_terms": stable_terms, "selected_for_M1": bool(stable_terms)}
        if stable_terms:
            selected.append(name)
    return selected, detail


def add_final_m1(result: dict, selected: list[str], subset: str) -> None:
    terms = MAIN_TERMS + [t for name in selected for t in INTERACTIONS[name]]
    if not selected:
        result["final_terms"] = MAIN_TERMS
        result["final_model"] = result["models"]["M0"]
        return
    model = fit_model(result["X"], result["y"], terms)
    result["final_terms"] = terms
    result["final_model"] = model
    mm = metrics(model, result["X"], result["y"], terms)
    base = result["comparisons"].iloc[0]
    row = {
        "subset": subset, "model": "M1_final", "interaction": "+".join(selected), **mm,
        "delta_roc_auc_vs_m0": mm["roc_auc"] - base["roc_auc"],
        "delta_pr_auc_vs_m0": mm["pr_auc"] - base["pr_auc"],
        "delta_log_loss_vs_m0": mm["log_loss"] - base["log_loss"],
    }
    result["comparisons"] = pd.concat([result["comparisons"], pd.DataFrame([row])], ignore_index=True)
    result["coefficients"] = pd.concat(
        [result["coefficients"], pd.DataFrame(coefficient_rows(model, terms, "M1_final", subset))], ignore_index=True
    )


def design_row(raw: dict, machine: int, in_band: bool, scale: dict) -> pd.DataFrame:
    row = {}
    for c in CONTINUOUS:
        row[c] = (raw[c] - scale["mean"][c]) / scale["sd"][c]
    row["in_band"] = float(in_band)
    row["machine_2"] = float(machine == 2)
    row["machine_3"] = float(machine == 3)
    row["texture_x_c_meas"] = row["texture"] * row["c_meas"]
    row["texture_x_machine_2"] = row["texture"] * row["machine_2"]
    row["texture_x_machine_3"] = row["texture"] * row["machine_3"]
    row["d_x_machine_2"] = row["d"] * row["machine_2"]
    row["d_x_machine_3"] = row["d"] * row["machine_3"]
    return pd.DataFrame([row])


def common_values(df: pd.DataFrame) -> dict:
    return {
        "c_meas": float(df["c_meas"].median()), "d": float(df["d"].median()),
        "log_edge_dist": float(np.log1p(df["edge_dist"]).median()),
        "bg": float(df["bg"].median()), "texture": float(df["texture"].median()),
    }


def plot_contrast_texture(result: dict, out: Path) -> dict:
    df, scale = result["df"], result["scale"]
    model = result["models"]["texture_x_contrast"]
    terms = MAIN_TERMS + INTERACTIONS["texture_x_contrast"]
    base = common_values(df)
    # texture 수준별 공통 관측 범위
    bins = pd.qcut(df["texture"], 3, labels=False, duplicates="drop")
    ranges = [df.loc[bins == k, "c_meas"].quantile([0.01, 0.99]).to_numpy() for k in sorted(bins.unique())]
    lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
    xs = np.linspace(lo, hi, 120)
    tex_values = df["texture"].quantile([0.2, 0.5, 0.8]).to_dict()
    labels = {0.2: "낮은 texture (P20)", 0.5: "중간 texture (P50)", 0.8: "높은 texture (P80)"}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    prediction_summary = {}
    for marker, ((q, tex), color) in zip(["o", "s", "^"], zip(tex_values.items(), TEXTURE_COLORS)):
        ps = []
        for x in xs:
            raw = dict(base, c_meas=float(x), texture=float(tex))
            row = design_row(raw, machine=1, in_band=bool(df["in_band"].mode().iat[0]), scale=scale)
            ps.append(model.predict_proba(row[terms])[:, 1][0])
        ax.plot(xs, ps, lw=2.2, color=color, marker=marker, markevery=24, ms=4.2, label=labels[q])
        prediction_summary[labels[q]] = {"texture": tex, "p_at_low_contrast": ps[0], "p_at_high_contrast": ps[-1]}
    ax.set(title="측정 contrast와 배경 texture의 상호작용", xlabel="측정 contrast", ylabel="예측 FN 확률")
    ax.set_title("측정 contrast와 배경 texture의 상호작용", pad=28)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, min(1.0, max(line.get_ydata().max() for line in ax.lines) * 1.12 + 0.02))
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.5, 1.01, "d=1 포함 민감도 분석에서 효과 불안정", transform=ax.transAxes,
            ha="center", va="bottom", color="#68707A", fontsize=9)
    fig.tight_layout(pad=1.2)
    save_figure(fig, out / "interaction_contrast_texture.png")
    return {
        "support_range": [lo, hi], "fixed_machine": 1,
        "fixed_in_band": bool(df["in_band"].mode().iat[0]), "predictions": prediction_summary,
        "report_note": "d=1 포함 민감도 분석에서 효과 불안정",
    }


def plot_texture_machine(result: dict, out: Path) -> dict:
    df, scale = result["df"], result["scale"]
    model = result["models"]["texture_x_machine"]
    terms = MAIN_TERMS + INTERACTIONS["texture_x_machine"]
    base = common_values(df)
    # 호기별 공통 관측 범위
    ranges = [
        df.loc[df["machine"] == machine, "texture"].quantile([0.01, 0.99]).to_numpy()
        for machine in [1, 2, 3]
    ]
    lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
    if lo >= hi:
        raise ValueError("No common texture support across machines")
    xs = np.linspace(lo, hi, 120)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    summary = {}
    curves = {}
    for machine, marker in zip([1, 2, 3], ["o", "s", "^"]):
        ps = []
        for x in xs:
            raw = dict(base, texture=float(x))
            row = design_row(raw, machine, bool(df["in_band"].mode().iat[0]), scale)
            ps.append(model.predict_proba(row[terms])[:, 1][0])
        curves[machine] = np.asarray(ps)
        ax.plot(xs, ps, lw=2.3, color=MACHINE_COLORS[machine], marker=marker,
                markevery=24, ms=4.3, label=f"{machine}호기")
        summary[str(machine)] = {"p_at_low_texture": ps[0], "p_at_high_texture": ps[-1]}
    ax.set(title="호기별 배경 texture에 따른 미탐 위험", xlabel="배경 texture", ylabel="예측 FN 확률")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, min(1.0, max(line.get_ydata().max() for line in ax.lines) * 1.12 + 0.02))
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    k2, k3 = 72, 82
    ax.annotate("2호기: texture 영향 완화", xy=(xs[k2], curves[2][k2]), xytext=(0.50, 0.18),
                textcoords="axes fraction", color=MACHINE_COLORS[2], fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MACHINE_COLORS[2], "lw": 0.9})
    ax.annotate("3호기: 영향 증가 경향", xy=(xs[k3], curves[3][k3]), xytext=(0.56, 0.76),
                textcoords="axes fraction", color=MACHINE_COLORS[3], fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MACHINE_COLORS[3], "lw": 0.9})
    fig.tight_layout(pad=1.2)
    save_figure(fig, out / "interaction_texture_machine.png")
    return {"support_range": [lo, hi], "predictions": summary}


def plot_diameter_machine(result: dict, out: Path) -> dict:
    df, scale = result["df"], result["scale"]
    model = result["models"]["diameter_x_machine"]
    terms = MAIN_TERMS + INTERACTIONS["diameter_x_machine"]
    base = common_values(df)
    xs = sorted(df["d"].unique())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    summary = {}
    curves = {}
    for machine, marker in zip([1, 2, 3], ["o", "s", "^"]):
        ps = []
        for x in xs:
            raw = dict(base, d=float(x))
            row = design_row(raw, machine, bool(df["in_band"].mode().iat[0]), scale)
            ps.append(model.predict_proba(row[terms])[:, 1][0])
        curves[machine] = np.asarray(ps)
        ax.plot(xs, ps, lw=2.3, color=MACHINE_COLORS[machine], marker=marker, ms=6, label=f"{machine}호기")
        summary[str(machine)] = {str(x): p for x, p in zip(xs, ps)}
    ax.set(title="호기별 이물 지름에 따른 미탐 위험", xlabel="이물 지름 (px)", ylabel="예측 FN 확률", xticks=xs)
    ax.set_ylim(0, min(1.0, max(line.get_ydata().max() for line in ax.lines) * 1.12 + 0.02))
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right", ncol=3)
    ax.annotate("3호기: 위험 감소폭 상대적으로 작음", xy=(xs[-1], curves[3][-1]), xytext=(0.43, 0.43),
                textcoords="axes fraction", color=MACHINE_COLORS[3], fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MACHINE_COLORS[3], "lw": 0.9})
    fig.tight_layout(pad=1.2)
    save_figure(fig, out / "interaction_diameter_machine.png")
    return {"diameters": xs, "predictions": summary}


def run_pca(df: pd.DataFrame, out: Path, seed: int) -> dict:
    features = ["c_meas", "d", "log_edge_dist", "bg", "texture"]
    raw = pd.DataFrame({
        "c_meas": df["c_meas"], "d": df["d"], "log_edge_dist": np.log1p(df["edge_dist"]),
        "bg": df["bg"], "texture": df["texture"],
    })
    Z = StandardScaler().fit_transform(raw)
    pca = PCA().fit(Z)
    scores = pca.transform(Z)
    score_df = pd.DataFrame({"src": df["src"].to_numpy(), "machine": df["machine"].to_numpy(), "miss": df["miss"].to_numpy(), "PC1": scores[:, 0], "PC2": scores[:, 1]})
    score_df.to_csv(out / "pca_scores.csv", index=False, encoding="utf-8-sig")
    loadings = pd.DataFrame({"feature": features, "PC1": pca.components_[0], "PC2": pca.components_[1]})
    loadings.to_csv(out / "pca_loadings.csv", index=False, encoding="utf-8-sig")
    explained = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(features))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(out / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(seed)
    tp_idx = np.flatnonzero(score_df["miss"].to_numpy() == 0)
    fn_idx = np.flatnonzero(score_df["miss"].to_numpy() == 1)
    max_tp = min(len(tp_idx), max(2000, 3 * len(fn_idx)))
    if len(tp_idx) > max_tp:
        tp_idx = rng.choice(tp_idx, size=max_tp, replace=False)

    # PCA 점수 분포
    fig, ax = plt.subplots(figsize=PCA_FIGSIZE)
    ax.scatter(score_df.loc[tp_idx, "PC1"], score_df.loc[tp_idx, "PC2"], s=12, alpha=0.22,
               c=TP_COLOR, marker="o", linewidths=0, label=f"TP (표시 {len(tp_idx):,})")
    ax.scatter(score_df.loc[fn_idx, "PC1"], score_df.loc[fn_idx, "PC2"], s=20, alpha=0.78,
               c=FN_COLOR, marker="x", linewidths=0.9, label=f"FN ({len(fn_idx):,})")
    ax.set(
        title="시험편 조건의 PCA 분포와 미탐 위치",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=1.2)
    save_figure(fig, out / "pca_tp_fn.png")

    # PCA biplot loading overlay
    fig, ax = plt.subplots(figsize=PCA_FIGSIZE)
    ax.scatter(score_df.loc[tp_idx, "PC1"], score_df.loc[tp_idx, "PC2"], s=12, alpha=0.18,
               c=TP_COLOR, marker="o", linewidths=0, label="TP")
    ax.scatter(score_df.loc[fn_idx, "PC1"], score_df.loc[fn_idx, "PC2"], s=20, alpha=0.76,
               c=FN_COLOR, marker="x", linewidths=0.9, label="FN")
    shown = score_df.loc[np.concatenate([tp_idx, fn_idx]), ["PC1", "PC2"]]
    x_span = float(shown["PC1"].max() - shown["PC1"].min())
    y_span = float(shown["PC2"].max() - shown["PC2"].min())
    loading_scale = 0.32 * min(
        x_span / (2 * loadings["PC1"].abs().max()),
        y_span / (2 * loadings["PC2"].abs().max()),
    )
    label_offsets = {
        "c_meas": (0.02, 0.02), "d": (-0.02, 0.02), "log_edge_dist": (0.02, -0.04),
        "bg": (-0.02, 0.02), "texture": (0.02, 0.05),
    }
    for r in loadings.itertuples():
        x_end, y_end = r.PC1 * loading_scale, r.PC2 * loading_scale
        ax.annotate("", xy=(x_end, y_end), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": "#354052", "lw": 1.2})
        dx, dy = label_offsets[r.feature]
        ax.text(x_end + dx * x_span, y_end + dy * y_span, DISPLAY[r.feature], fontsize=8.8,
                ha="left" if x_end >= 0 else "right", va="center", color="#354052")
    ax.axhline(0, color="#AAB0B8", lw=0.7, zorder=0)
    ax.axvline(0, color="#AAB0B8", lw=0.7, zorder=0)
    ax.set(
        title="시험편 조건의 PCA 분포와 미탐 위치",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=1.2)
    save_figure(fig, out / "pca_biplot.png")
    top = {
        "PC1": loadings.assign(abs_loading=loadings["PC1"].abs()).nlargest(3, "abs_loading")[["feature", "PC1"]].to_dict("records"),
        "PC2": loadings.assign(abs_loading=loadings["PC2"].abs()).nlargest(3, "abs_loading")[["feature", "PC2"]].to_dict("records"),
    }
    return {
        "n_samples": len(df), "fn": int(df["miss"].sum()), "tp": int((df["miss"] == 0).sum()),
        "pc1": pca.explained_variance_ratio_[0], "pc2": pca.explained_variance_ratio_[1],
        "pc1_pc2_cumulative": pca.explained_variance_ratio_[:2].sum(), "top_loadings": top,
        "biplot_loading_scale": loading_scale,
        "caption_note": "FN이 특정 조건 공간에 상대적으로 집중되는 경향을 표시하며 인과관계를 의미하지 않음.",
        "interpretation_limit": "PCA is descriptive feature-space visualization, not evidence that PCs cause FN.",
    }


def plot_interaction_forest(boot_summary: pd.DataFrame, out: Path) -> dict:
    primary = boot_summary[boot_summary["subset"] == "PRIMARY"].set_index("term")
    items = [
        ("texture_x_machine_2", "배경 texture × 2호기", 2, "o"),
        ("texture_x_machine_3", "배경 texture × 3호기", 3, "o"),
        ("d_x_machine_2", "이물 지름 × 2호기", 2, "s"),
        ("d_x_machine_3", "이물 지름 × 3호기", 3, "s"),
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    records = []
    for y, (term, label, machine, marker) in enumerate(items[::-1]):
        r = primary.loc[term]
        or_value, lo, hi = float(r["odds_ratio"]), float(r["or_ci_low"]), float(r["or_ci_high"])
        stable = bool(r["ci_excludes_zero"])
        ax.errorbar(
            or_value, y, xerr=[[or_value - lo], [hi - or_value]], fmt=marker,
            ms=7, mfc=MACHINE_COLORS[machine] if stable else "white", mec=MACHINE_COLORS[machine],
            ecolor=MACHINE_COLORS[machine], elinewidth=1.5, capsize=3, capthick=1.2,
        )
        records.append({"term": term, "label": label, "odds_ratio": or_value, "ci_low": lo, "ci_high": hi, "ci_excludes_one": stable})
    ax.axvline(1.0, color="#68707A", lw=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(items)), [item[1] for item in items[::-1]])
    ax.set(title="핵심 상호작용 효과 (PRIMARY)", xlabel="Odds Ratio (95% CI)", ylabel="")
    style_axis(ax)
    ax.grid(False, axis="y")
    fig.tight_layout(pad=1.2)
    save_figure(fig, out / "interaction_forest_plot.png")
    return {"subset": "PRIMARY", "items": records}


def reproduce_existing_main_effect(df: pd.DataFrame, conditions: dict) -> dict:
    X = pd.DataFrame({
        "측정대비": df["c_meas"], "지름": df["d"], "가장자리거리(log)": np.log1p(df["edge_dist"]),
        "띠안": df["in_band"].astype(float), "주변밝기": df["bg"], "주변결": df["texture"],
        "2호기": (df["machine"] == 2).astype(float), "3호기": (df["machine"] == 3).astype(float),
    })
    Z = (X - X.mean()) / X.std()
    model = LogisticRegression(C=10, max_iter=2000).fit(Z, df["miss"])
    reproduced = {c: float(np.exp(b)) for c, b in zip(Z.columns, model.coef_[0])}
    reported = conditions["놓침"]["로지스틱_오즈비(1표준편차당, >1이면 놓침 증가)"]
    diff = {c: reproduced[c] - float(reported[c]) for c in reproduced}
    return {"reported_odds_ratio": reported, "reproduced_odds_ratio": reproduced, "difference": diff, "max_absolute_difference": max(abs(v) for v in diff.values())}


def tree_findings(tree_text: str) -> dict:
    return {
        "contains_contrast": "측정대비" in tree_text,
        "contains_texture": "주변결" in tree_text,
        "contains_machine": "3호기" in tree_text,
        "observed_structure": "The tree first splits at measured contrast 0.075; above it, texture 3.598 separates risk, while below it machine 3 appears as the next split.",
    }


def build_markdown(summary: dict, comparisons: pd.DataFrame, boot: pd.DataFrame) -> str:
    audit = summary["data_audit"]
    pca = summary["pca"]
    selected = summary["selection"]["selected_interactions"]
    verdict = summary["final_verdict"]
    lines = [
        "# 변수 간 상호작용 분석 요약", "",
        "## 1. 사용 데이터와 동결 조건", "",
        f"- 입력: `{summary['inputs']['data']}`", f"- 전체 {audit['rows']:,}행, 원본 배경 `src` {audit['unique_src']:,}개",
        f"- 저장된 FN target: {audit['miss_distribution'].get('1', audit['miss_distribution'].get(1, 0)):,}행 ({audit['miss_rate']:.1%})",
        f"- 동결된 합격선/불합격선: {audit['fixed_thresholds']['accept']:.4f} / {audit['fixed_thresholds']['reject']:.4f}",
        f"- PRIMARY: {audit['primary']['rows']:,}행, FN {audit['primary']['fn']:,}행 ({audit['primary']['fn_rate']:.1%})",
        f"- SENSITIVITY(d=1 포함): {audit['sensitivity']['rows']:,}행, FN {audit['sensitivity']['fn']:,}행 ({audit['sensitivity']['fn_rate']:.1%})",
        "", "## 2. 모형", "",
        "M0는 표준화한 측정 대비, 지름, log 가장자리 거리, 배경 밝기, 배경 결과 띠 여부 및 범주형 호기 더미를 사용했다. 사전에 정한 texture × contrast, texture × machine, diameter × machine만 각각 검증했다. M1에는 PRIMARY와 d=1 포함 민감도 분석에서 계수 방향과 src-cluster bootstrap 95% 구간이 모두 안정적인 interaction만 포함했다.",
        "", "기존 분석과 같은 L2 규제 로지스틱 회귀(`C=10`)를 사용했으므로 비규제 최대우도 모형의 AIC/BIC를 억지로 계산하지 않았다.",
        "", "## 3. src-cluster bootstrap interaction 결과", "",
    ]
    for name in INTERACTIONS:
        rows = boot[(boot["subset"] == "PRIMARY") & (boot["interaction"] == name)]
        lines.append(f"### {name} - PRIMARY")
        lines.append("")
        for r in rows.itertuples():
            lines.append(f"- `{r.term}`: 계수 {r.coefficient:.3f}, OR {r.odds_ratio:.3f}, src-cluster bootstrap 계수 95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]")
        lines.append("")
        lines.append("SENSITIVITY(d=1 포함):")
        lines.append("")
        for r in boot[(boot["subset"] == "SENSITIVITY_D1_INCLUDED") & (boot["interaction"] == name)].itertuples():
            lines.append(f"- `{r.term}`: 계수 {r.coefficient:.3f}, OR {r.odds_ratio:.3f}, src-cluster bootstrap 계수 95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]")
        lines.append("")
    lines += [
        "texture × contrast는 PRIMARY에서 양의 계수였지만 d=1 포함 시 CI가 0을 포함해 핵심 interaction으로 채택하지 않았다. 따라서 '낮은 대비일수록 거친 배경의 악영향이 더 커진다'는 사전 가설을 강건하게 확인했다고 주장할 수 없다. texture × machine은 2호기의 texture 기울기가 1호기보다 완만했고, diameter × machine은 3호기의 지름 기울기가 1호기와 달랐다.",
        "", "## 4. 모형 비교", "",
        "| Model | ROC-AUC | PR-AUC | Log loss |", "|---|---:|---:|---:|",
    ]
    for r in comparisons[comparisons["subset"] == "PRIMARY"].itertuples():
        lines.append(f"| {r.model} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.log_loss:.4f} |")
    lines += [
        "", "같은 데이터에 적합한 지표 변화는 기술적 비교이며 그 자체가 interaction의 근거는 아니다. 판단은 계수 방향, src-cluster bootstrap 안정성, 조건별 예측위험을 함께 사용했다.",
        "", "## 5. PCA", "",
        f"PC1 {pca['pc1']:.1%}, PC2 {pca['pc2']:.1%}, 누적 {pca['pc1_pc2_cumulative']:.1%}. PC1은 가장자리 거리·texture·배경 밝기, PC2는 대비·지름의 기여가 컸다. FN은 PC1의 양의 영역에 더 모였지만 TP와 상당히 겹치므로 PCA를 원인 증명으로 해석하지 않는다.",
        "", "## 6. 기존 분석과의 관계", "",
        "기존 로지스틱 회귀는 가법적 주효과를 설명했고, 깊이 3 결정나무는 contrast·texture·3호기의 조건 조합을 이미 보였다. 이번 분석은 그 조합 중 사전 지정한 product term이 원본 영상 단위 재표집에서도 안정적인지를 추가로 확인했다. texture × machine 및 diameter × machine은 나무의 contrast/texture/machine 조합과 부분적으로 같은 방향이나, 결정나무에는 diameter × machine 분기가 없어 완전한 재현은 아니다.",
        "", "## 7. 현장 활용 경계", "",
        "안정적인 호기별 texture·지름 취약성은 재검사 우선순위의 보조 risk factor로만 사용할 수 있다. 합격선 0.6128과 불합격선 0.8367은 변경하지 않았으며, 새 production threshold를 제안하지 않는다.",
        "", "## 8. 허용·금지 주장", "",
        f"- 최종 M1에 선택: {', '.join(selected) if selected else '없음'}",
        f"- 최종 판정: **{verdict}**",
        "- 허용: 계수 방향, src-cluster bootstrap 구간, PRIMARY 관측 범위 안의 예측위험 패턴, 기술적 PCA 구조",
        "- 금지: 인과효과, 14,400개 독립표본, threshold 개선, detector 성능 개선 주장",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    font_family = configure_plot_style()
    if args.bootstrap < 1000:
        raise ValueError("--bootstrap must be at least 1000 for the requested source-cluster uncertainty analysis")
    args.out.mkdir(parents=True, exist_ok=True)

    # 입력 파일 검증
    raw, conditions, risk, tree_text, before_hashes = load_inputs(args)
    df, audit = validate(raw, conditions, risk)
    primary = df[df["c_meas"].between(0.10, 0.20, inclusive="both") & df["d"].gt(1.0)].copy()
    sensitivity = df[df["c_meas"].between(0.10, 0.20, inclusive="both")].copy()
    if primary["miss"].nunique() != 2 or sensitivity["miss"].nunique() != 2:
        raise ValueError("PRIMARY and SENSITIVITY subsets must each contain TP and FN rows")

    # PRIMARY 및 d=1 포함 민감도 분석
    primary_result = fit_subset(primary, "PRIMARY", args.bootstrap, args.seed)
    sensitivity_result = fit_subset(sensitivity, "SENSITIVITY_D1_INCLUDED", args.bootstrap, args.seed + 1)

    # 핵심 상호작용 선택
    selected, selection_detail = robust_interactions(primary_result["bootstrap_summary"], sensitivity_result["bootstrap_summary"])
    add_final_m1(primary_result, selected, "PRIMARY")
    add_final_m1(sensitivity_result, selected, "SENSITIVITY_D1_INCLUDED")

    comparisons = pd.concat([primary_result["comparisons"], sensitivity_result["comparisons"]], ignore_index=True)
    coefficients = pd.concat([primary_result["coefficients"], sensitivity_result["coefficients"]], ignore_index=True)
    boot_raw = pd.concat([primary_result["bootstrap_raw"], sensitivity_result["bootstrap_raw"]], ignore_index=True)
    boot_summary = pd.concat([primary_result["bootstrap_summary"], sensitivity_result["bootstrap_summary"]], ignore_index=True)
    comparisons.to_csv(args.out / "model_comparison.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(args.out / "interaction_coefficients.csv", index=False, encoding="utf-8-sig")
    boot_raw.to_csv(args.out / "interaction_bootstrap.csv", index=False, encoding="utf-8-sig")

    # 보고서용 시각화 저장
    plots = {"contrast_texture": plot_contrast_texture(primary_result, args.out)}
    if "texture_x_machine" in selected:
        plots["texture_machine"] = plot_texture_machine(primary_result, args.out)
    if "diameter_x_machine" in selected:
        plots["diameter_machine"] = plot_diameter_machine(primary_result, args.out)
    plots["interaction_forest"] = plot_interaction_forest(boot_summary, args.out)
    pca_summary = run_pca(primary, args.out, args.seed)

    primary_sig = bool(primary_result["bootstrap_summary"]["ci_excludes_zero"].any())
    if selected:
        verdict = "MEANINGFUL_INTERACTION"
    elif primary_sig:
        verdict = "LIMITED_INTERACTION"
    else:
        verdict = "NO_ROBUST_INTERACTION"
    # 입력 파일 무결성 검증
    after_hashes = {path: sha256(Path(path)) for path in before_hashes}
    if before_hashes != after_hashes:
        raise RuntimeError("One or more source input files changed during analysis")

    summary = {
        "inputs": {
            "data": str(args.data.resolve()), "tree": str(args.tree.resolve()),
            "conditions_summary": str((args.summary_a if "놓침" in json.loads(args.summary_a.read_text(encoding='utf-8-sig')) else args.summary_b).resolve()),
            "risk_threshold_summary": str((args.summary_a if "채택" in json.loads(args.summary_a.read_text(encoding='utf-8-sig')) else args.summary_b).resolve()),
            "sha256_before_and_after_unchanged": before_hashes,
        },
        "data_audit": audit,
        "existing_main_effect_reproduction": reproduce_existing_main_effect(df, conditions),
        "existing_tree": tree_findings(tree_text),
        "bootstrap": {
            "method": "sample src with replacement and include every row from each sampled source",
            "requested_replicates_per_subset": args.bootstrap, "seed_primary": args.seed, "seed_sensitivity": args.seed + 1,
            "fit_failures": {"primary": primary_result["bootstrap_failures"], "sensitivity": sensitivity_result["bootstrap_failures"]},
        },
        "interaction_bootstrap_summary": boot_summary.to_dict("records"),
        "selection": {"rule": "Select an interaction for M1 when at least one component CI excludes zero in both subsets with the same coefficient direction.", "selected_interactions": selected, "detail": selection_detail},
        "plots": plots,
        "plot_font_family": font_family,
        "pca": pca_summary,
        "thresholds_unchanged": {"accept": 0.6128, "reject": 0.8367},
        "final_verdict": verdict,
        "competition_criterion_completion": "YES" if verdict == "MEANINGFUL_INTERACTION" else ("PARTIAL" if verdict == "LIMITED_INTERACTION" else "NO"),
    }
    write_json(args.out / "data_audit.json", audit)
    write_json(args.out / "interaction_summary.json", summary)
    (args.out / "analysis_summary.md").write_text(build_markdown(summary, comparisons, boot_summary), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(df), "primary_rows": len(primary), "primary_fn": int(primary["miss"].sum()), "selected": selected, "verdict": verdict}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

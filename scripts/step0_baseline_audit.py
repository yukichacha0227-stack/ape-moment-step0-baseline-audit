#!/usr/bin/env python3
"""Audit the unchanged F+TO3 baseline against its source spectral observations.

This script does not fit or modify the baseline. Run ``hourly_hybrid_to3.py``
first, then point ``--baseline-run`` at that run's output directory. Source
spectra are supplied separately because the historical baseline CLI only takes
precomputed APE targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


HC_EV_NM_LEGACY = 1239.84193
WAVELENGTHS_NM = np.arange(350, 1051, dtype=np.float64)
SPECTRAL_HEADER = re.compile(r"^日射強度\((\d+)nm\)\[W/m2/μm\]$")
SURFACES = ("HSR", "TSR")
BASELINE_MODEL = "fixed_beta_plus_residual_mlp_recursive_calibrated"
ROLES = ("train_oof", "validation_early", "validation_calibration", "test")
EXPECTED_TEST = {
    "HSR": {"r2": 0.6567, "rmse_meV": 13.8417, "mae_meV": 10.6759,
            "mbe_truth_minus_prediction_meV": -1.0063},
    "TSR": {"r2": 0.7795, "rmse_meV": 14.1903, "mae_meV": 10.3116,
            "mbe_truth_minus_prediction_meV": -0.4819},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_moments(irradiance: np.ndarray) -> dict[str, float]:
    """Calculate both trapezoidal physical moments and legacy 1-nm sums.

    The CSV's E(lambda) unit is W m^-2 um^-1, whereas lambda is in nm.
    Dividing the nm-domain integrals by 1000 yields S0 in W m^-2 and S1 in
    W m^-2 nm. The factor cancels in the APE ratio.
    """
    values = np.asarray(irradiance, dtype=np.float64)
    if values.shape != (len(WAVELENGTHS_NM),):
        raise ValueError(f"Expected 701 spectral bins; got shape {values.shape}")
    if not np.isfinite(values).all() or (values == -9999).any():
        raise ValueError("An accepted source spectrum contains missing values")
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    s0_trap = float(trapezoid(values, WAVELENGTHS_NM) / 1000.0)
    s1_trap = float(trapezoid(values * WAVELENGTHS_NM, WAVELENGTHS_NM) / 1000.0)
    s0_legacy = float(np.sum(values) / 1000.0)
    s1_legacy = float(np.sum(values * WAVELENGTHS_NM) / 1000.0)
    if min(s0_trap, s1_trap, s0_legacy, s1_legacy) <= 0:
        raise ValueError("An accepted source spectrum has a nonpositive moment")
    return {
        "S0_trapezoid_W_m2": s0_trap,
        "S1_trapezoid_W_m2_nm": s1_trap,
        "ape_trapezoid_eV": HC_EV_NM_LEGACY * s0_trap / s1_trap,
        "S0_legacy_W_m2": s0_legacy,
        "S1_legacy_W_m2_nm": s1_legacy,
        "ape_legacy_eV": HC_EV_NM_LEGACY * s0_legacy / s1_legacy,
    }


def parse_source_datetime(date_text: str, time_text: str) -> pd.Timestamp:
    text = str(time_text).strip()
    rollover = bool(re.fullmatch(r"24:00(?::00)?", text))
    stamp = pd.Timestamp(f"{str(date_text).strip()} {'00:00' if rollover else text}")
    return stamp + (pd.Timedelta(days=1) if rollover else pd.Timedelta(0))


def load_target(path: Path, surface: str) -> tuple[pd.DataFrame, dict]:
    # Only trusted, locally generated joblib files may be supplied here.
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "target_dataframe" not in payload:
        raise TypeError(f"No target_dataframe in {path}")
    frame = payload["target_dataframe"].copy()
    required = {"Datetime", "SiteNum", "Remark", "APE", "source_file", "source_row_index"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{surface} target is missing provenance fields: {sorted(missing)}")
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="raise")
    if frame["Datetime"].dt.tz is not None:
        raise ValueError("Target Datetime must be JST-local and timezone-naive")
    frame["surface"] = surface
    frame["source_row_index"] = frame["source_row_index"].astype(int)
    if frame.duplicated(["Datetime", "SiteNum", "surface"]).any():
        raise ValueError(f"Duplicate instantaneous {surface} target keys")
    return frame, {
        "sha256": sha256(path),
        "rows": len(frame),
        "source_files": int(frame["source_file"].nunique()),
        "remark_counts": frame["Remark"].value_counts().sort_index().to_dict(),
        "processor_config": payload.get("processor_config", {}),
    }


def source_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(relative)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Source file escapes spectral root: {relative}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Spectral source not found: {candidate}")
    return candidate


def audit_target_spectra(
    target: pd.DataFrame, surface: str, spectral_root: Path
) -> tuple[pd.DataFrame, dict]:
    results: list[dict] = []
    header_count = Counter()
    missing_source_rows = 0
    for relative, group in target.groupby("source_file", sort=True):
        path = source_path(spectral_root, str(relative))
        wanted = {int(row.source_row_index): row for row in group.itertuples(index=False)}
        if len(wanted) != len(group):
            raise ValueError(f"Duplicate source row index in {surface}: {relative}")
        with path.open("r", encoding="shift_jis", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if header[:5] != ["地点番号", "地点名", "年月日", "時分", "リマーク"]:
                raise ValueError(f"Unexpected spectral metadata columns: {relative}")
            if len(header) != 1356:
                raise ValueError(f"Expected 5 metadata + 1351 wavelength columns: {relative}")
            waves = []
            for name in header[5:]:
                match = SPECTRAL_HEADER.fullmatch(name)
                if match is None:
                    raise ValueError(f"Unexpected wavelength/irradiance unit: {relative}: {name}")
                waves.append(int(match.group(1)))
            if waves != list(range(350, 1701)):
                raise ValueError(f"Wavelengths are not 350–1700 nm at 1 nm: {relative}")
            header_count[(waves[0], waves[-1], len(waves))] += 1
            for row_index, cells in enumerate(reader):
                if row_index not in wanted:
                    continue
                provenance = wanted.pop(row_index)
                if len(cells) != len(header):
                    raise ValueError(f"Incorrect field count: {relative} row {row_index}")
                stamp = parse_source_datetime(cells[2], cells[3])
                if stamp != provenance.Datetime:
                    raise ValueError(f"Timestamp mismatch: {relative} row {row_index}")
                if int(cells[0]) != int(provenance.SiteNum):
                    raise ValueError(f"Site mismatch: {relative} row {row_index}")
                if int(cells[4]) != int(provenance.Remark):
                    raise ValueError(f"Remark mismatch: {relative} row {row_index}")
                try:
                    irradiance = np.asarray(cells[5:706], dtype=np.float64)
                except ValueError as exc:
                    raise ValueError(f"Invalid spectral value: {relative} row {row_index}") from exc
                results.append({
                    "datetime": stamp.tz_localize("Asia/Tokyo").tz_convert("UTC"),
                    "surface": surface,
                    "SiteNum": int(provenance.SiteNum),
                    "Remark": int(provenance.Remark),
                    "source_file": str(relative),
                    "source_row_index": row_index,
                    "ape_target_eV": float(provenance.APE),
                    **spectral_moments(irradiance),
                })
        missing_source_rows += len(wanted)
        if wanted:
            raise ValueError(f"Source rows absent from {relative}: {sorted(wanted)[:5]}")
    frame = pd.DataFrame(results)
    if len(frame) != len(target) or missing_source_rows:
        raise ValueError(f"Incomplete {surface} spectral audit: {len(frame)} / {len(target)}")
    if frame.duplicated(["datetime", "surface"]).any():
        raise ValueError(f"Duplicate {surface} timestamps after spectral audit")
    frame = frame.sort_values(["datetime", "surface"]).reset_index(drop=True)
    return frame, {
        "surface": surface,
        "spectral_files_read": int(sum(header_count.values())),
        "headers": {str(key): value for key, value in header_count.items()},
        "audited_spectra": len(frame),
        "source_row_index_and_timestamp_verified": True,
    }


def numeric_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(reference, dtype=float)
    y = np.asarray(candidate, dtype=float)
    if len(x) == 0 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid comparison arrays")
    difference = y - x
    return {
        "n": int(len(x)),
        "r2": float(r2_score(x, y)),
        "mae_eV": float(mean_absolute_error(x, y)),
        "max_abs_difference_eV": float(np.max(np.abs(difference))),
        "mean_difference_eV": float(np.mean(difference)),
    }


def hourly_moments(instant: pd.DataFrame) -> pd.DataFrame:
    work = instant.copy()
    local = work["datetime"].dt.tz_convert("Asia/Tokyo")
    work["datetime"] = (local.dt.floor("h") + pd.Timedelta(minutes=30)).dt.tz_convert("UTC")
    group = work.groupby(["datetime", "SiteNum", "surface"], as_index=False)
    hourly = group.agg(
        n_spectra=("ape_target_eV", "size"),
        ape_target_A_eV=("ape_target_eV", "mean"),
        ape_legacy_A_eV=("ape_legacy_eV", "mean"),
        ape_trapezoid_A_eV=("ape_trapezoid_eV", "mean"),
        S0_trapezoid_mean_W_m2=("S0_trapezoid_W_m2", "mean"),
        S1_trapezoid_mean_W_m2_nm=("S1_trapezoid_W_m2_nm", "mean"),
        S0_legacy_mean_W_m2=("S0_legacy_W_m2", "mean"),
        S1_legacy_mean_W_m2_nm=("S1_legacy_W_m2_nm", "mean"),
    )
    hourly["ape_trapezoid_B_eV"] = (
        HC_EV_NM_LEGACY * hourly["S0_trapezoid_mean_W_m2"]
        / hourly["S1_trapezoid_mean_W_m2_nm"]
    )
    hourly["ape_legacy_B_eV"] = (
        HC_EV_NM_LEGACY * hourly["S0_legacy_mean_W_m2"]
        / hourly["S1_legacy_mean_W_m2_nm"]
    )
    return hourly


def plot_baseline(test: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, surface in zip(axes, SURFACES):
        part = test.loc[test["surface"].eq(surface)]
        truth = part["ape_true"].to_numpy(float)
        prediction = part[BASELINE_MODEL].to_numpy(float)
        axis.scatter(truth, prediction, s=10, alpha=0.32, color="#2878B5")
        low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        axis.plot([low, high], [low, high], "k--", linewidth=1, label="y = x")
        axis.set(title=f"{surface} (n={len(part):,})", xlabel="Measured APE (eV)",
                 ylabel="Predicted APE (eV)")
        axis.legend(loc="lower right")
    fig.suptitle("Unchanged F+TO3 baseline: independent chronological test")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_recalculation(instant: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for surface, color in (("HSR", "#2878B5"), ("TSR", "#D95F02")):
        part = instant.loc[instant["surface"].eq(surface)]
        axes[0].scatter(part["ape_target_eV"], part["ape_legacy_eV"],
                        s=2, alpha=0.08, color=color, label=surface)
        axes[1].hist(1000 * (part["ape_trapezoid_eV"] - part["ape_target_eV"]),
                     bins=100, alpha=0.5, color=color, label=surface)
    axes[0].plot([1.2, 2.2], [1.2, 2.2], "k--", linewidth=1)
    axes[0].set(xlabel="Stored instantaneous APE (eV)",
                ylabel="Recalculated legacy-sum APE (eV)", title="Exact formula check")
    axes[1].set(xlabel="Trapezoid minus stored APE (meV)",
                ylabel="Number of spectra", title="Quadrature sensitivity")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            f"{value:.8g}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--hsr-target", type=Path, required=True)
    parser.add_argument("--tsr-target", type=Path, required=True)
    parser.add_argument("--hsr-spectral-root", type=Path, required=True)
    parser.add_argument("--tsr-spectral-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Audit output must be empty: {output}")
    baseline = args.baseline_run.resolve()
    metrics_path = baseline / "metrics_all_splits.csv"
    predictions_path = baseline / "predictions_all_models.csv"
    split_path = baseline / "split_report.json"
    config_path = baseline / "run_config.json"
    for path in (metrics_path, predictions_path, split_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_report = json.loads(split_path.read_text(encoding="utf-8"))
    model_report = json.loads(
        (baseline / "model_and_calibration_report.json").read_text(encoding="utf-8")
    )
    if config.get("experiment") != "F+TO3":
        raise ValueError("The baseline run is not F+TO3")
    settings = config["settings"]
    if settings.get("seed") != 42 or settings.get("purge_hours") != 6 or settings.get("residual_lags_hours") != [1, 2, 3, 6]:
        raise ValueError("The baseline run changed the published seed/purge/lag protocol")
    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig", parse_dates=["datetime"])
    if predictions["datetime"].dt.tz is None:
        raise ValueError("Baseline timestamps must carry UTC timezone")
    if predictions.duplicated(["datetime", "SiteNum", "surface"]).any():
        raise ValueError("Duplicate baseline prediction keys")
    roles_by_time = predictions.groupby("datetime")["split"].nunique()
    if not roles_by_time.eq(1).all():
        raise ValueError("HSR and TSR timestamps have different split roles")
    paired = predictions.groupby("datetime")["surface"].nunique()
    if not paired.eq(2).all():
        raise ValueError("Unpaired HSR/TSR baseline timestamps")
    test_metrics = metrics.loc[metrics["split"].eq("test") & metrics["surface"].isin(SURFACES)
                               & metrics["model"].eq(BASELINE_MODEL)].copy()
    if len(test_metrics) != 2:
        raise ValueError("Missing HSR or TSR baseline test metrics")
    test_metrics["readme_consistent"] = False
    for index, row in test_metrics.iterrows():
        expected = EXPECTED_TEST[row["surface"]]
        test_metrics.at[index, "readme_consistent"] = all(
            abs(float(row[key]) - value) <= (0.0001 if key == "r2" else 0.1)
            for key, value in expected.items()
        )
        test_rows = predictions.loc[
            predictions["split"].eq("test") & predictions["surface"].eq(row["surface"])
        ]
        truth = test_rows["ape_true"].to_numpy(float)
        predicted = test_rows[BASELINE_MODEL].to_numpy(float)
        residual = truth - predicted
        recomputed = {
            "r2": r2_score(truth, predicted),
            "rmse_meV": 1000 * np.sqrt(np.mean(residual ** 2)),
            "mae_meV": 1000 * np.mean(np.abs(residual)),
            "mbe_truth_minus_prediction_meV": 1000 * np.mean(residual),
        }
        if int(row["n"]) != len(test_rows) or any(
            abs(float(row[key]) - value) > 1e-9 for key, value in recomputed.items()
        ):
            raise ValueError(f"Test predictions and metrics disagree for {row['surface']}")
    test_metrics.to_csv(output / "metrics_baseline.csv", index=False)
    if not test_metrics["readme_consistent"].all():
        raise ValueError("Baseline test result differs materially from README")
    predictions[["datetime", "SiteNum", "surface", "split", "ape_true", BASELINE_MODEL,
                 "ape_samples_in_hour"]].to_csv(output / "predictions_baseline.csv", index=False)
    plot_baseline(predictions.loc[predictions["split"].eq("test")],
                  output / "baseline_measured_vs_predicted_ape.png")

    instant_parts = []
    source_reports = {}
    target_reports = {}
    for surface, target_path, spectral_root in (
        ("HSR", args.hsr_target, args.hsr_spectral_root),
        ("TSR", args.tsr_target, args.tsr_spectral_root),
    ):
        target, target_reports[surface] = load_target(target_path, surface)
        print(f"Auditing {surface}: {len(target):,} source spectra...", flush=True)
        instant, source_reports[surface] = audit_target_spectra(target, surface, spectral_root)
        instant_parts.append(instant)
    instant = pd.concat(instant_parts, ignore_index=True)
    instant.to_csv(output / "spectral_moment_validation.csv", index=False)
    plot_recalculation(instant, output / "ape_recalculation_validation.png")

    comparisons = []
    for surface, part in instant.groupby("surface", sort=True):
        for name, column in (("legacy_sum", "ape_legacy_eV"),
                             ("trapezoid", "ape_trapezoid_eV")):
            comparisons.append({"surface": surface, "level": "instantaneous", "method": name,
                **numeric_comparison(part["ape_target_eV"], part[column])})
    hourly = hourly_moments(instant)
    # Compare with the baseline's exact hourly target and enforce equal source counts.
    joined = predictions.merge(hourly, on=["datetime", "SiteNum", "surface"],
                               how="left", validate="one_to_one", indicator=True)
    joined["moment_available"] = (
        joined["_merge"].eq("both")
        & joined["n_spectra"].eq(joined["ape_samples_in_hour"])
        & np.isfinite(joined["ape_legacy_A_eV"])
        & np.isfinite(joined["ape_trapezoid_A_eV"])
        & np.isfinite(joined["S0_trapezoid_mean_W_m2"])
        & np.isfinite(joined["S1_trapezoid_mean_W_m2_nm"])
    )
    available = joined.loc[joined["moment_available"]]
    if len(available) == 0:
        raise ValueError("No baseline rows have matching spectral moments")
    for surface, part in available.groupby("surface", sort=True):
        for name, column in (("legacy_A", "ape_legacy_A_eV"),
                             ("legacy_B", "ape_legacy_B_eV"),
                             ("trapezoid_A", "ape_trapezoid_A_eV"),
                             ("trapezoid_B", "ape_trapezoid_B_eV")):
            comparisons.append({"surface": surface, "level": "baseline_hourly", "method": name,
                **numeric_comparison(part["ape_true"], part[column])})
        comparisons.append({"surface": surface, "level": "A_vs_B", "method": "trapezoid_B_minus_A",
            **numeric_comparison(part["ape_trapezoid_A_eV"], part["ape_trapezoid_B_eV"])})
        comparisons.append({"surface": surface, "level": "A_vs_B", "method": "legacy_B_minus_A",
            **numeric_comparison(part["ape_legacy_A_eV"], part["ape_legacy_B_eV"])})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "ape_recalculation_metrics.csv", index=False)

    # Freeze only timestamps for which both surfaces have complete spectral
    # moments and which belong to one of the four modeling roles.
    eligible = joined.loc[joined["split"].isin(ROLES)].copy()
    complete_times = eligible.groupby("datetime").agg(
        surfaces=("surface", "nunique"), valid_rows=("moment_available", "sum")
    )
    common_times = complete_times.index[(complete_times["surfaces"] == 2)
                                        & (complete_times["valid_rows"] == 2)]
    cohort = (eligible.loc[eligible["datetime"].isin(common_times)]
              .sort_values(["datetime", "surface"]).copy())
    cohort_path = output / "common_cohort_keys.csv"
    cohort[["datetime", "SiteNum", "surface", "split", "n_spectra",
            "ape_target_A_eV", "ape_legacy_A_eV", "ape_trapezoid_A_eV",
            "ape_trapezoid_B_eV", "ape_legacy_B_eV",
            "S0_trapezoid_mean_W_m2", "S1_trapezoid_mean_W_m2_nm",
            "S0_legacy_mean_W_m2", "S1_legacy_mean_W_m2_nm"]].to_csv(
                cohort_path, index=False)
    cohort_sha256 = sha256(cohort_path)
    summary = (eligible.groupby(["split", "surface"], as_index=False)
               .agg(baseline_rows=("surface", "size"), moment_valid_rows=("moment_available", "sum")))
    cohort_counts = (cohort.groupby(["split", "surface"]).size().rename("common_rows")
                     .reset_index())
    summary = summary.merge(cohort_counts, on=["split", "surface"], how="left")
    summary["common_rows"] = summary["common_rows"].fillna(0).astype(int)
    summary["excluded_rows"] = summary["baseline_rows"] - summary["common_rows"]
    summary["unique_common_hours"] = summary["common_rows"]
    summary.to_csv(output / "common_cohort_summary.csv", index=False)

    legacy_instant_max = comparison.loc[comparison["method"].eq("legacy_sum"),
                                        "max_abs_difference_eV"].max()
    legacy_hourly_max = comparison.loc[comparison["method"].eq("legacy_A"),
                                       "max_abs_difference_eV"].max()
    same_cohort = len(cohort) == len(eligible)
    verified = bool(legacy_instant_max < 1e-9 and legacy_hourly_max < 1e-9)
    status = "PASS" if verified and same_cohort else "FAIL"
    if verified and not same_cohort:
        status = "CONDITIONAL PASS"
    gap = comparison.loc[comparison["level"].eq("A_vs_B")]
    legacy_gap = gap.loc[gap["method"].eq("legacy_B_minus_A")].set_index("surface")
    split_table = pd.DataFrame([
        {
            "role": role,
            "start_utc": split_report[role]["start_utc"],
            "end_utc": split_report[role]["end_utc"],
            "paired_hours": split_report[role]["unique_hours"],
        }
        for role in ROLES
    ])
    report = f"""# Step 0 — baseline reproduction and spectral-moment audit

## 1. 実験目的

直接APE回帰と低次元スペクトルモーメント経由予測を比較する前に、既存F+TO3モデル、スペクトル由来のS0/S1、APEの算式、共通評価母集団を監査する。

## 2. 設計理念

既存baselineのコード・5大気変数・seed・split・MLP・再帰推論・校正を一切変更しない。新しい処理は独立した監査スクリプトに限定し、比較対象の時刻と表面を先に固定する。

## 3. 仮説

元スペクトルに対して既存の1 nm単純和を適用すれば、pickleの瞬時APEおよびbaselineの時間算術平均APEを数値精度で再現できる。台形積分は端点重みが異なるため、単純和との差が残る可能性がある。

## 4. 使用データ

- 筑波地点301、HSR水平面とTSR傾斜32°。各日付の`10HSR*.csv`/`10TSR*.csv`から、pickleの`source_file`と`source_row_index`で元行を厳密に参照した。
- 元CSVはShift-JIS、先頭5列が地点番号・地点名・年月日・時分・リマーク、後続1351列が350–1700 nmの1 nm刻み、列単位は`W/m2/μm`。
- 解析対象は350–1050 nmの701列。既存target生成はRemark 1/2、全対象波長の有限値かつ非`-9999`、正の分子・分母、APE 1.2–2.2 eVを採用する。実際に採用されたRemark内訳はHSR {target_reports['HSR']['remark_counts']}、TSR {target_reports['TSR']['remark_counts']}。
- HSR/TSRともpickleの時刻はJSTのtimezone-naive値であり、監査では元CSV時刻、行番号、地点、Remarkとの一致を全採用行で検証した。入力ハッシュは`audit_manifest.json`に保存。
- 検証した元ファイルはHSR {source_reports['HSR']['spectral_files_read']}件・TSR {source_reports['TSR']['spectral_files_read']}件、採用スペクトルはHSR {source_reports['HSR']['audited_spectra']:,}件・TSR {source_reports['TSR']['audited_spectra']:,}件。元スペクトルは既存baseline CLIのAPE入力に含まれないため、独立監査CLIの`--hsr-spectral-root`/`--tsr-spectral-root`で明示的に受け取った。
- 添付の参照`model.py`は0バイトだったため、既存実験スクリプトと元APE生成実装を基準にした。

## 5. APE / S0 / S1の定義

`E(λ)`の単位を`W m^-2 μm^-1`、波長をnmとし、`S0 = ∫ E(λ)dλ/1000` (`W m^-2`)、`S1 = ∫ λE(λ)dλ/1000` (`W m^-2 nm`) とする。`APE = 1239.84193 S0/S1` eV。積分は`numpy.trapezoid`（利用可能なNumPy）を原則とし、既存生成式の`sum(E)`/`sum(λE)`も別列に保存した。定数`1239.84193 eV nm`は既存TSR生成実装と一致する。

## 6. hourly aggregation上の注意

- A: 各瞬時スペクトルからAPEを計算し、JSTの`[hh:00, hh+1:00)`で表面別に算術平均。既存baseline targetの再現はAで行う。
- B: 同じ時間窓のS0/S1をそれぞれ平均した後に比をとる。これはAと一般に一致しない。
- 同じ積分規則の中で比較したB−Aを以下に示す（差の単位eV）。legacy規則ではHSRのMAEが{1000 * legacy_gap.loc['HSR', 'mae_eV']:.3f} meV、最大差が{1000 * legacy_gap.loc['HSR', 'max_abs_difference_eV']:.3f} meV、TSRのMAEが{1000 * legacy_gap.loc['TSR', 'mae_eV']:.3f} meV、最大差が{1000 * legacy_gap.loc['TSR', 'max_abs_difference_eV']:.3f} meVであった。これは単なる積分規則の差ではない。

{markdown_table(gap[['surface', 'method', 'n', 'mae_eV', 'max_abs_difference_eV', 'mean_difference_eV']])}

## 7. 現行baselineモデル

Fixed-beta Ridge → Train OOF beta prediction → residual target → residual MLP → 表面別の因果的recursive residual rollout → validation後半だけによるsurface-wise affine calibration → untouched chronological Test。一次大気入力はAM、AOD550、omega(TQV)、TOTANGSTR、TO3のみ。対照HGBも既存スクリプトのまま実行した。

## 8. chronological split

{markdown_table(split_table)}

Residual MLPのbest epochは{model_report['best_epoch']}、validation earlyの最良lossは{model_report['best_validation_loss']:.8g}。early stoppingとsurface-wise affine calibrationは別のvalidation期間を使用し、Testは学習・選択・校正に使っていない（実行ログ上: `{model_report['test_used_for_training_selection_or_calibration']}`）。

seed={settings['seed']}、70/15/15%の時系列分割、6時間purge、Validation前半でearly stopping、後半で校正。HSR/TSR同時刻は同一role。元のsplitラベルを行単位予測CSVから継承し、共通母集団の作成時にも変更していない。

{markdown_table(summary[['split', 'surface', 'baseline_rows', 'moment_valid_rows', 'common_rows', 'excluded_rows']])}

## 9. APE再計算検証

stored APEを基準としたR²、MAE、最大絶対差、平均差を記録した。legacy_sumは既存生成式、trapezoidは台形積分である。時間Aは瞬時APEの算術平均、Bは平均S0/S1の比。

{markdown_table(comparison[['surface', 'level', 'method', 'n', 'r2', 'mae_eV', 'max_abs_difference_eV', 'mean_difference_eV']])}

![APE再計算検証](ape_recalculation_validation.png)

## 10. baseline結果

READMEの既存結果と比較した独立Test。MBEは正解−予測、単位meV。

{markdown_table(test_metrics[['surface', 'n', 'r2', 'rmse_meV', 'mae_meV', 'mbe_truth_minus_prediction_meV', 'readme_consistent']])}

## 11. 予測APE vs 正解APEの散布図の解釈

独立Testのみ。横軸は正解APE、縦軸は予測APE、破線は`y=x`。HSRとTSRは別パネルとした。高APE側の平均への回帰・過小予測は残るため、R²だけで上側誤差を隠さない。

![baseline独立Test散布図](baseline_measured_vs_predicted_ape.png)

## 12. common cohort

固定キー集合のSHA-256: `{cohort_sha256}`。時刻はUTCで保存する。

baselineに存在する4 roleのうち、元スペクトル由来のS0/S1と瞬時APEが全件一致し、同時刻HSR/TSRが両方揃う時刻だけを採用した。`common_cohort_keys.csv`を後続のBaseline/Raw Moment/Normalized Moment/Physics-guided Multi-taskの固定キー集合とする。後続モデルがこの集合以外を評価することは禁止する。baseline全role行と同一か: **{same_cohort}**。

## 13. 問題点・異常

- 既存APEは台形積分ではなく1 nm単純和である。両者の差は積分端点の扱いによる数値定義差であり、未解決のデータ異常と混同しない。
- BはAと異なり得る。後続のS0/S1学習でBを予測して既存AのR²と比較する設計は不公平である。
- raw prediction、スペクトル行由来のモーメント、共通キーは公開データセットではない。リポジトリの`.gitignore`で非公開とする。

## 14. 次のRaw Moment実験を実施可能か

**{status}**。legacy_sumによる瞬時/時間Aの最大差がそれぞれ{legacy_instant_max:.3g}/{legacy_hourly_max:.3g} eV。次段階では既存APEと同じ数値定義のS0/S1を教師にするか、台形積分のAPEへ全モデルのtargetを統一してbaselineを再学習する必要がある。どちらを採るか実験設定に明記する。common cohortが縮小した場合は、このcohortでbaselineを同じ条件で再学習するまで性能比較を開始しない。

## 15. 結論

既存baselineとスペクトル積分量の対応を検証した。判定: **{status}**。研究仮説に不利な台形積分差・hourly A/B差・高APE過小予測を含め、結果をそのまま報告した。
"""
    (output / "00_baseline_audit_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "status": status,
        "baseline_run_config_sha256": sha256(config_path),
        "baseline_metrics_sha256": sha256(metrics_path),
        "baseline_predictions_sha256": sha256(predictions_path),
        "target_reports": target_reports,
        "spectral_source_reports": source_reports,
        "common_rows": len(cohort),
        "common_unique_hours": int(cohort["datetime"].nunique()),
        "common_cohort_keys_sha256": cohort_sha256,
        "all_baseline_modeling_rows": len(eligible),
        "ape_legacy_instant_max_abs_difference_eV": float(legacy_instant_max),
        "ape_legacy_hourly_max_abs_difference_eV": float(legacy_hourly_max),
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Step 0 audit: {status}; common rows={len(cohort):,}; report={output / '00_baseline_audit_report.md'}")
    return 0 if status != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())


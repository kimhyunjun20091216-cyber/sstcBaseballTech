"""측정-검증 통계 프레임워크 (Phase 4).

CV로 측정한 관절각을 참값(ground truth, 수동 주석)과 대조하여 일치도·신뢰도를
정량 평가한다. 계획서 §3.2가 요구하는 지표를 변수별로 산출한다:
  - Bland-Altman: 편향(bias) + 95% 일치한계(LoA)
  - ICC(2,1): 이원 랜덤·절대일치·단일평정 + 95% CI
  - RMSE, MAE, Pearson r

의존성: numpy, scipy, pandas, matplotlib 만 사용(pingouin 등 무거운 패키지 배제 →
이식성↑). ICC는 Shrout & Fleiss(1979) 공식을 scipy.stats.f 로 직접 구현.

출력:
  - <out_md>            검증 리포트(변수별 표)
  - <out_png_dir>/*.png Bland-Altman 플롯(변수별)

사용:
    # 실제 참값 대조
    python src/validation.py --cv out/영상1_측면_angles.csv \
        --ref annotations/영상1_측면_ref.csv --out out/영상1_검증.md
    # 데모(합성 참값으로 파이프라인 동작 증명)
    python src/validation.py --demo

주의: 참값 없이 --demo 로 만든 결과는 SYNTHETIC(합성)이며 실제 검증 아님.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats


# analyze/kinematics 산출 CSV에서 검증 대상이 되는 관절각 컬럼(°).
ANGLE_COLUMNS = [
    "L_elbow_deg", "R_elbow_deg", "L_knee_deg", "R_knee_deg",
    "L_arm_slot_deg", "R_arm_slot_deg", "trunk_tilt_deg",
]


# ============================ 오차 지표 ============================

def _pair(a, b):
    """두 배열을 정렬·NaN 제거하여 쌍으로 반환."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    good = ~(np.isnan(a) | np.isnan(b))
    return a[good], b[good]


def rmse(a, b):
    """Root Mean Squared Error. 큰 오차에 민감(이상치 노출)."""
    a, b = _pair(a, b)
    if len(a) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    """Mean Absolute Error. 평균적 오차 크기."""
    a, b = _pair(a, b)
    if len(a) == 0:
        return np.nan
    return float(np.mean(np.abs(a - b)))


def pearson_r(a, b):
    """Pearson 상관계수. 연관성만 나타냄(일치 아님 → 반드시 Bland-Altman 병행)."""
    a, b = _pair(a, b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ============================ Bland-Altman ============================

def bland_altman(method, reference):
    """두 측정법의 일치도 분석[Bland & Altman, Lancet 1986].

    반환 dict:
      bias       평균 차이(method - reference), 계통오차.
      sd_diff    차이의 표준편차.
      loa_lower  bias - 1.96*sd_diff (95% 하한 일치한계).
      loa_upper  bias + 1.96*sd_diff.
      mean       각 쌍의 평균값 배열(x축).
      diff       각 쌍의 차이 배열(y축).
      n          유효 쌍 개수.
    """
    m, r = _pair(method, reference)
    diff = m - r
    mean = (m + r) / 2.0
    bias = float(np.mean(diff)) if len(diff) else np.nan
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_lower": bias - 1.96 * sd if not np.isnan(sd) else np.nan,
        "loa_upper": bias + 1.96 * sd if not np.isnan(sd) else np.nan,
        "mean": mean,
        "diff": diff,
        "n": int(len(diff)),
    }


def bland_altman_plot(method, reference, png_path, title="", unit="°"):
    """Bland-Altman 산점도(평균 vs 차이)를 PNG로 저장."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fam in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic"):
        try:
            matplotlib.rcParams["font.family"] = fam
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    ba = bland_altman(method, reference)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ba["mean"], ba["diff"], s=18, alpha=0.6, color="tab:blue")
    ax.axhline(ba["bias"], color="tab:red", lw=1.5,
               label=f"편향 {ba['bias']:.2f}{unit}")
    ax.axhline(ba["loa_upper"], color="gray", ls="--", lw=1.2,
               label=f"+1.96SD {ba['loa_upper']:.2f}")
    ax.axhline(ba["loa_lower"], color="gray", ls="--", lw=1.2,
               label=f"-1.96SD {ba['loa_lower']:.2f}")
    ax.set_xlabel(f"두 측정 평균 ({unit})")
    ax.set_ylabel(f"차이 (CV - 참값, {unit})")
    ax.set_title(title or "Bland-Altman")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


# ============================ ICC(2,1) ============================

def icc21(matrix):
    """ICC(2,1): 이원 랜덤효과·절대일치·단일평정 신뢰도.

    행 = 대상(target, 예: 프레임), 열 = 평정자/방법(rater, 예: [CV, 참값]).
    Shrout & Fleiss(1979) Case 2, 단일 측정치(single measure) 공식:

        ICC(2,1) = (BMS - EMS) /
                   (BMS + (k-1)*EMS + k*(JMS - EMS)/n)

      BMS 대상 간 평균제곱, JMS 평정자 간 평균제곱,
      EMS 잔차 평균제곱, n 대상 수, k 평정자 수.

    95% CI는 F 분포 기반[McGraw & Wong 1996; Koo & Li 2016].
    반환: dict(icc, ci_low, ci_high, n, k).
    """
    x = np.asarray(matrix, dtype=float)
    # 결측 행 제거(완전 케이스).
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        return {"icc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n, "k": k}

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    # 제곱합 분해.
    ss_total = ((x - grand) ** 2).sum()
    ss_row = k * ((row_means - grand) ** 2).sum()      # 대상(between-target)
    ss_col = n * ((col_means - grand) ** 2).sum()      # 평정자(between-rater)
    ss_err = ss_total - ss_row - ss_col                # 잔차

    df_row = n - 1
    df_col = k - 1
    df_err = (n - 1) * (k - 1)

    bms = ss_row / df_row
    jms = ss_col / df_col
    ems = ss_err / df_err if df_err > 0 else np.nan

    denom = bms + (k - 1) * ems + k * (jms - ems) / n
    icc = (bms - ems) / denom if denom != 0 else np.nan

    # 95% CI (McGraw & Wong 1996, ICC(2,1) = ICC(A,1)).
    # a, b 항으로 자유도 v를 구한 뒤 F 임계값으로 하·상한 산출.
    alpha = 0.05
    try:
        fl = bms / ems                                  # 관측 F(대상)
        a = (k * icc) / (n * (1 - icc))
        b = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
        v = (a * jms + b * ems) ** 2 / \
            ((a * jms) ** 2 / (k - 1) + (b * ems) ** 2 / ((n - 1) * (k - 1)))
        f_lower = stats.f.ppf(1 - alpha / 2, n - 1, v)
        f_upper = stats.f.ppf(1 - alpha / 2, v, n - 1)
        lo = n * (bms - f_lower * ems) / \
            (f_lower * (k * jms + (k * n - k - n) * ems) + n * bms)
        hi = n * (f_upper * bms - ems) / \
            (k * jms + (k * n - k - n) * ems + n * f_upper * bms)
        ci_low, ci_high = float(lo), float(hi)
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    return {"icc": float(icc), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "n": n, "k": k}


def interpret_icc(icc):
    """Koo & Li(2016) 해석 기준."""
    if np.isnan(icc):
        return "판정불가"
    if icc < 0.5:
        return "나쁨(poor)"
    if icc < 0.75:
        return "보통(moderate)"
    if icc < 0.9:
        return "좋음(good)"
    return "우수(excellent)"


# ============================ 통합 검증 ============================

def validate_variable(name, cv_values, ref_values):
    """한 변수에 대해 전체 지표 산출."""
    cv, ref = _pair(cv_values, ref_values)
    ba = bland_altman(cv, ref)
    icc = icc21(np.column_stack([cv, ref])) if len(cv) >= 2 else \
        {"icc": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return {
        "name": name,
        "n": ba["n"],
        "bias": ba["bias"],
        "loa_lower": ba["loa_lower"],
        "loa_upper": ba["loa_upper"],
        "icc": icc["icc"],
        "icc_ci": (icc["ci_low"], icc["ci_high"]),
        "rmse": rmse(cv, ref),
        "mae": mae(cv, ref),
        "r": pearson_r(cv, ref),
        "interp": interpret_icc(icc["icc"]),
    }


# ====================== 참값(수동주석) 로드/정렬 ======================
#
# 참값 CSV 스키마 (수동 주석):
#   - `frame` 컬럼(정수)으로 CV 출력 CSV와 프레임 정렬.
#   - 각 검증 대상 각도는 CV CSV와 동일한 컬럼명 사용(예: L_elbow_deg).
#     주석자가 프레임별로 관절각을 직접 측정(각도기/수동 디지타이징)하여 기입.
#   - 전 프레임을 주석할 필요는 없음. 주석된 프레임만 남기면 됨(내부 조인 처리).
#   예)
#     frame,L_elbow_deg,R_elbow_deg,trunk_tilt_deg
#     30,128.0,,6.0
#     60,95.0,,-22.0
#     70,80.0,,-38.0

def load_paired(cv_csv, ref_csv, columns=None):
    """CV·참값 CSV를 frame 기준으로 정렬하여 변수별 (cv, ref) 배열 dict 반환."""
    import pandas as pd

    cv = pd.read_csv(cv_csv)
    ref = pd.read_csv(ref_csv)
    if "frame" not in cv.columns or "frame" not in ref.columns:
        raise ValueError("두 CSV 모두 'frame' 컬럼이 필요합니다.")

    cols = columns or [c for c in ANGLE_COLUMNS if c in cv.columns and c in ref.columns]
    merged = ref.merge(cv, on="frame", suffixes=("_ref", "_cv"))
    pairs = {}
    for c in cols:
        cv_col, ref_col = f"{c}_cv", f"{c}_ref"
        if cv_col in merged and ref_col in merged:
            sub = merged[[ref_col, cv_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 2:
                pairs[c] = (sub[cv_col].to_numpy(), sub[ref_col].to_numpy())
    return pairs


# ============================ 리포트 ============================

def validate_report(cv_angles_csv, ref_angles_csv, out_md, out_png_dir,
                     columns=None, synthetic=False):
    """검증 리포트(마크다운 표 + Bland-Altman PNG) 생성."""
    pairs = load_paired(cv_angles_csv, ref_angles_csv, columns)
    os.makedirs(out_png_dir, exist_ok=True)

    rows, pngs = [], []
    for name, (cv, ref) in pairs.items():
        res = validate_variable(name, cv, ref)
        png = os.path.join(out_png_dir, f"ba_{name}.png")
        bland_altman_plot(cv, ref, png, title=f"Bland-Altman: {name}")
        rows.append(res)
        pngs.append((name, png))

    banner = ("> ⚠ **SYNTHETIC/DEMO** — 합성 참값으로 만든 결과입니다. 실제 검증 아님.\n\n"
              if synthetic else "")
    lines = [
        f"# 투구 CV 측정-검증 리포트 (Phase 4)",
        "",
        banner + f"- CV 출력: `{os.path.basename(cv_angles_csv)}`",
        f"- 참값: `{os.path.basename(ref_angles_csv)}`",
        f"- 검증 변수 수: {len(rows)}",
        "",
        "## 변수별 일치도·신뢰도",
        "",
        "| 변수 | n | 편향(bias) | 95% LoA | ICC(2,1) [95% CI] | RMSE | MAE | r | 판정 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for res in rows:
        ci = res["icc_ci"]
        ci_s = (f"[{ci[0]:.2f}, {ci[1]:.2f}]"
                if not (np.isnan(ci[0]) or np.isnan(ci[1])) else "[–]")
        lines.append(
            f"| {res['name']} | {res['n']} | {res['bias']:.2f}° | "
            f"[{res['loa_lower']:.1f}, {res['loa_upper']:.1f}]° | "
            f"{res['icc']:.3f} {ci_s} | {res['rmse']:.2f}° | {res['mae']:.2f}° | "
            f"{res['r']:.3f} | {res['interp']} |"
        )
    lines += [
        "",
        "## Bland-Altman 플롯",
        "",
    ]
    for name, png in pngs:
        lines.append(f"### {name}")
        lines.append(f"![{name}]({os.path.relpath(png, os.path.dirname(out_md))})")
        lines.append("")
    lines += [
        "## 해석 지침",
        "",
        "- **ICC**(Koo & Li 2016): <0.5 나쁨 / 0.5–0.75 보통 / 0.75–0.9 좋음 / >0.9 우수.",
        "- **Bland-Altman**: 편향은 계통오차, LoA는 개별 측정이 놓일 95% 범위.",
        "- **r 단독으로 일치 주장 불가** — 반드시 Bland-Altman·ICC와 함께 해석.",
        "- **RMSE ≥ MAE**: 둘의 격차가 크면 소수의 큰 오차(이상치) 존재.",
        "",
        "> 통계 출처: Bland & Altman(1986); Shrout & Fleiss(1979); "
        "McGraw & Wong(1996); Koo & Li(2016).",
    ]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"  → 리포트: {out_md}  (변수 {len(rows)}, 플롯 {len(pngs)})")
    return {"variables": [r["name"] for r in rows], "results": rows}


# ============================ 신뢰도(참고) ============================

def reliability_icc(*raters):
    """test-retest / inter-rater / intra-rater 신뢰도.

    각 rater 인자는 동일 길이의 측정 배열(같은 대상들에 대한 반복/평정).
    ICC(2,1)로 계산하여 dict 반환.
    """
    mat = np.column_stack([np.asarray(r, dtype=float) for r in raters])
    res = icc21(mat)
    res["interp"] = interpret_icc(res["icc"])
    return res


# ============================ ICC 정합성 점검 ============================

def _icc_sanity_check():
    """Shrout & Fleiss(1979) Table 2의 표준 예제로 ICC(2,1) 검증.

    알려진 값: ICC(2,1) ≈ 0.290.
    """
    data = np.array([
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ], dtype=float)
    res = icc21(data)
    return res["icc"]


# ============================ 데모 ============================

def _run_demo():
    """합성 참값으로 전체 파이프라인 동작을 증명한다(실제 검증 아님)."""
    import pandas as pd

    src_csv = "out/영상1_측면_angles.csv"
    if not os.path.exists(src_csv):
        raise FileNotFoundError(
            f"{src_csv} 없음. 먼저 kinematics.py 로 각도 CSV를 생성하세요.")

    out_dir = "out/validation_demo"
    os.makedirs(out_dir, exist_ok=True)

    # ICC 정합성 점검.
    icc_check = _icc_sanity_check()
    print(f"[ICC 정합성] Shrout&Fleiss 예제 → ICC(2,1)={icc_check:.3f} "
          f"(기대 ≈0.290)")

    # 합성 참값: CV 각도에 계통편향 + 관측노이즈 부가(고정 시드).
    rng = np.random.default_rng(42)
    cv = pd.read_csv(src_csv)
    ref = cv[["frame"]].copy()
    biases = {"L_elbow_deg": 2.0, "R_elbow_deg": -1.5, "L_knee_deg": 1.0,
              "R_knee_deg": 3.0, "L_arm_slot_deg": -2.0, "R_arm_slot_deg": 2.5,
              "trunk_tilt_deg": 1.5}
    for c in ANGLE_COLUMNS:
        if c in cv.columns:
            noise = rng.normal(0, 3.0, len(cv))         # 관측노이즈 SD=3°
            ref[c] = cv[c].to_numpy() + biases.get(c, 0) + noise
    # 일부 프레임만 주석된 상황을 모사(20프레임 샘플).
    ref = ref.iloc[::6].reset_index(drop=True)
    ref_csv = os.path.join(out_dir, "SYNTHETIC_ref_angles.csv")
    ref.to_csv(ref_csv, index=False)
    print(f"[합성 참값] {ref_csv} (주석 프레임 {len(ref)}개, 편향+노이즈 부가)")

    validate_report(src_csv, ref_csv,
                    out_md=os.path.join(out_dir, "검증리포트_DEMO.md"),
                    out_png_dir=os.path.join(out_dir, "plots"),
                    synthetic=True)
    print("[완료] 데모 산출물: out/validation_demo/  (SYNTHETIC — 실제 검증 아님)")


def main():
    ap = argparse.ArgumentParser(description="투구 CV 측정-검증 통계(Phase 4)")
    ap.add_argument("--cv", help="CV 출력 각도 CSV (kinematics.py)")
    ap.add_argument("--ref", help="참값(수동주석) CSV")
    ap.add_argument("--out", default="out/검증리포트.md", help="리포트 md 경로")
    ap.add_argument("--png-dir", default="out/validation_plots", help="플롯 저장 폴더")
    ap.add_argument("--demo", action="store_true", help="합성 참값 데모 실행")
    args = ap.parse_args()

    if args.demo:
        _run_demo()
        return
    if not (args.cv and args.ref):
        ap.error("--cv 와 --ref 를 지정하거나 --demo 를 사용하세요.")
    validate_report(args.cv, args.ref, args.out, args.png_dir)


if __name__ == "__main__":
    main()

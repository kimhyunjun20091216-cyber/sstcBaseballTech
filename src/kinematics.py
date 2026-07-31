"""운동학 분석 — 관절각 계산 · 투구 단계 자동 분할 · 리포트 생성.

입력: analyze.py가 생성한 keypoint CSV (프레임별 COCO 17관절 x,y,visibility).
출력:
  - <stem>_angles.csv   프레임별 관절각·속도
  - <stem>_kinematics.png  시계열 그래프(단계 마커 포함)
  - <stem>_report.md    핵심 지표 + 정상값 대조 리포트

사용:
    python src/kinematics.py out/영상1_측면_keypoints.csv --fps 30 --hand auto

주의(2D 단일 카메라 한계):
  - 각도는 영상 평면 투영값 → 카메라 각도에 의존. 측면 영상의 팔꿈치·무릎·
    몸통 시상면 각도가 가장 신뢰성 높다.
  - 어깨 내·외회전(횡단면)은 2D로 신뢰 불가 → 산출하지 않음(계획서 §1.2).
"""

from __future__ import annotations

import argparse
import os

import numpy as np


# COCO 17 인덱스.
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14
L_AN, R_AN = 15, 16

# 성인/프로 정상값(계획서 §1.1, Diffendaffer & Fleisig 2022 등).
NORMS = {
    "elbow_flexion_release": (20, 30, "팔꿈치 굽힘(릴리스)"),
    "lead_knee_flexion_release": (25, 35, "앞무릎 굽힘(릴리스)"),
    "trunk_forward_tilt_release": (28, 42, "전방 몸통 기울기(릴리스)"),
    "stride_pct_height": (77, 90, "스트라이드 길이(키 대비 %)"),
}


# ----------------------------- 로드/전처리 -----------------------------

def load_keypoints(csv_path):
    """CSV → (kpts (N,17,3), times (N,), fps)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    names = [c[:-2] for c in df.columns if c.endswith("_x")]
    n = len(df)
    kpts = np.full((n, len(names), 3), np.nan, dtype=np.float64)
    for j, nm in enumerate(names):
        kpts[:, j, 0] = df[f"{nm}_x"].to_numpy(dtype=np.float64)
        kpts[:, j, 1] = df[f"{nm}_y"].to_numpy(dtype=np.float64)
        kpts[:, j, 2] = df[f"{nm}_v"].to_numpy(dtype=np.float64)
    times = df["time_s"].to_numpy(dtype=np.float64) if "time_s" in df else np.arange(n)
    fps = 1.0 / np.median(np.diff(times)) if n > 1 and np.any(np.diff(times) > 0) else 30.0
    return kpts, times, fps


def _interp_nan(a):
    """1D 배열의 NaN을 선형 보간(양끝은 최근접 값 유지)."""
    a = a.copy()
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    if good.sum() < 2:
        return a
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def clean(kpts, vis_thresh=0.3):
    """visibility 낮은 좌표를 NaN 처리 후 x,y를 시간축으로 보간."""
    k = kpts.copy()
    low = k[:, :, 2] < vis_thresh
    k[low, 0] = np.nan
    k[low, 1] = np.nan
    for j in range(k.shape[1]):
        k[:, j, 0] = _interp_nan(k[:, j, 0])
        k[:, j, 1] = _interp_nan(k[:, j, 1])
    return k


# ------------------------------- 각도 -------------------------------

def joint_angle(a, b, c):
    """정점 B에서 A-B-C가 이루는 각(도). a,b,c: (N,2)."""
    ba = a - b
    bc = c - b
    cosang = (ba * bc).sum(1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def tilt_from_vertical(top, bottom):
    """top·bottom(어깨중점·골반중점)을 잇는 몸통 벡터의 연직 대비 기울기(도).

    부호: 화면 오른쪽으로 기울면 +. 측면 영상에서는 전방 기울기의 대리값.
    """
    v = top - bottom            # 아래→위 방향
    ang = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))   # 연직 위=0
    return ang


def xy(kpts, idx):
    return kpts[:, idx, :2]


def compute_angles(k):
    """관절각·속도 딕셔너리 반환.

    팔꿈치·무릎은 생체역학 표준인 **굴곡각(flexion)** 으로 반환한다:
    flexion = 180 − 내각 → 완전히 편 상태 0°, 굽힐수록 증가.
    """
    sh_mid = (xy(k, L_SH) + xy(k, R_SH)) / 2
    hip_mid = (xy(k, L_HIP) + xy(k, R_HIP)) / 2
    out = {
        "L_elbow": 180 - joint_angle(xy(k, L_SH), xy(k, L_EL), xy(k, L_WR)),
        "R_elbow": 180 - joint_angle(xy(k, R_SH), xy(k, R_EL), xy(k, R_WR)),
        "L_knee": 180 - joint_angle(xy(k, L_HIP), xy(k, L_KN), xy(k, L_AN)),
        "R_knee": 180 - joint_angle(xy(k, R_HIP), xy(k, R_KN), xy(k, R_AN)),
        # 팔 슬롯: 상완(어깨→팔꿈치)과 몸통(어깨→골반)이 이루는 각 ≈ 어깨 외전.
        "L_arm_slot": joint_angle(xy(k, L_EL), xy(k, L_SH), hip_mid),
        "R_arm_slot": joint_angle(xy(k, R_EL), xy(k, R_SH), hip_mid),
        "trunk_tilt": tilt_from_vertical(sh_mid, hip_mid),
    }
    return out, sh_mid, hip_mid


def speed(pt, fps):
    """(N,2) 좌표 → 속도 크기(px/s)."""
    v = np.zeros(len(pt))
    d = np.linalg.norm(np.diff(pt, axis=0), axis=1) * fps
    v[1:] = d
    return v


# --------------------------- 단계/이벤트 분할 ---------------------------

def detect_handedness(k, fps):
    """던지는 팔·리드 다리 자동 판정.

    던지는 팔 = 손목 최고속이 큰 쪽. 리드 다리 = 던지는 팔의 반대쪽
    (일반적 투구역학). 반환: dict(throw='L'|'R', lead='L'|'R').
    """
    ls = speed(xy(k, L_WR), fps).max()
    rs = speed(xy(k, R_WR), fps).max()
    throw = "L" if ls > rs else "R"
    lead = "R" if throw == "L" else "L"
    return {"throw": throw, "lead": lead}


def detect_events(k, hand, fps):
    """주요 이벤트 프레임 인덱스 추정.

    - leg_lift_peak: 리드 무릎이 골반 대비 가장 높이 올라간 프레임.
    - release: 던지는 손목 속도 최대 프레임(후반부 탐색).
    - foot_contact: leg_lift_peak~release 사이, 리드 발목이 가장 낮아진(착지) 프레임.
    """
    n = len(k)
    lead_kn = L_KN if hand["lead"] == "L" else R_KN
    lead_an = L_AN if hand["lead"] == "L" else R_AN
    throw_wr = L_WR if hand["throw"] == "L" else R_WR
    hip_mid_y = ((xy(k, L_HIP) + xy(k, R_HIP)) / 2)[:, 1]

    # 무릎 높이(골반 대비, 값이 작을수록 높음).
    knee_rel = k[:, lead_kn, 1] - hip_mid_y
    leg_lift_peak = int(np.nanargmin(knee_rel))

    wr_speed = speed(xy(k, throw_wr), fps)
    # 릴리스는 다리 들기 이후에서 손목 최고속.
    lo = min(leg_lift_peak + 1, n - 1)
    release = lo + int(np.nanargmax(wr_speed[lo:])) if lo < n else int(np.nanargmax(wr_speed))

    # 앞발 착지: 다리들기~릴리스 사이 발목이 가장 낮은(y 최대) 지점.
    if release > leg_lift_peak + 1:
        seg = k[leg_lift_peak:release, lead_an, 1]
        foot_contact = leg_lift_peak + int(np.nanargmax(seg))
    else:
        foot_contact = leg_lift_peak

    return {
        "leg_lift_peak": leg_lift_peak,
        "foot_contact": foot_contact,
        "release": release,
        "_wr_speed": wr_speed,
    }


def stride_pct_height(k, hand, events):
    """스트라이드 길이(양 발목 간 수평거리)를 키(발목~코) 대비 %로 근사.

    2D·원근 영향 큰 근사값 — 참고용.
    """
    fc = events["foot_contact"]
    lift = events["leg_lift_peak"]
    rel = events["release"]
    # 스트라이드 폭: 들기~릴리스 구간의 최대 발목 수평거리(착지 순간 근사).
    seg = slice(min(lift, rel), max(rel, lift) + 1)
    ankle_dx = np.nanmax(np.abs(k[seg, L_AN, 0] - k[seg, R_AN, 0])) if rel != lift \
        else abs(k[fc, L_AN, 0] - k[fc, R_AN, 0])
    # 키 근사: 코~아래발목 수직거리(FC 프레임).
    ank_y = max(k[fc, L_AN, 1], k[fc, R_AN, 1])
    height_px = abs(ank_y - k[fc, NOSE, 1])
    if height_px < 1:
        return np.nan
    return 100.0 * ankle_dx / height_px


# ------------------------------- 리포트 -------------------------------

def _smooth(a, win=5):
    if len(a) < win:
        return a
    kern = np.ones(win) / win
    return np.convolve(_interp_nan(a), kern, mode="same")


def build_report(stem, angles, hand, events, fps, n, stride_pct):
    thr, lead = hand["throw"], hand["lead"]
    rel = events["release"]
    fc = events["foot_contact"]

    elbow_rel = angles[f"{thr}_elbow"][rel]
    knee_rel = angles[f"{lead}_knee"][rel]
    knee_fc = angles[f"{lead}_knee"][fc]
    trunk_rel = abs(angles["trunk_tilt"][rel])
    arm_slot_rel = angles[f"{thr}_arm_slot"][rel]

    def norm_str(key, val, unit="°"):
        lo, hi, label = NORMS[key]
        mark = "✓ 정상범위" if lo <= val <= hi else "⚠ 범위밖"
        return f"| {label} | **{val:.0f}{unit}** | {lo}–{hi}{unit} | {mark} |"

    lines = [
        f"# 투구 운동학 분석 리포트 — {os.path.basename(stem)}",
        "",
        f"- 프레임 수: {n} · 프레임레이트: {fps:.0f} fps · 길이: {n/fps:.1f}s",
        f"- 던지는 팔(추정): **{'왼손' if thr=='L' else '오른손'}** · "
        f"리드 다리: **{'왼다리' if lead=='L' else '오른다리'}**",
        "",
        "## 투구 단계 이벤트 (자동 검출)",
        "",
        "| 이벤트 | 프레임 | 시각(s) |",
        "|---|---|---|",
        f"| 다리 들기 최고점 | {events['leg_lift_peak']} | {events['leg_lift_peak']/fps:.2f} |",
        f"| 앞발 착지(추정) | {fc} | {fc/fps:.2f} |",
        f"| 릴리스(손목 최고속) | {rel} | {rel/fps:.2f} |",
        "",
        "## 핵심 지표 vs 정상값 (성인/프로 기준)",
        "",
        "| 지표 | 측정값 | 정상범위 | 판정 |",
        "|---|---|---|---|",
        norm_str("elbow_flexion_release", elbow_rel),
        norm_str("lead_knee_flexion_release", knee_rel),
        norm_str("trunk_forward_tilt_release", trunk_rel),
        norm_str("stride_pct_height", stride_pct, "%") if not np.isnan(stride_pct)
        else "| 스트라이드 길이(키 대비 %) | (측정불가) | 77–90% | – |",
        "",
        f"- 앞무릎 굽힘(착지 시): {knee_fc:.0f}°  (정상 ~45°, 착지→릴리스 신전이 정상 패턴)",
        f"- 던지는 팔 슬롯(릴리스): {arm_slot_rel:.0f}°  (어깨 외전 대리값, ~90° 부근)",
        "",
        "## 해석 주의 (2D 단일 카메라 한계)",
        "",
        "- 위 각도는 영상 평면 투영값으로, 카메라 각도·원근에 따라 오차가 있음.",
        "- **어깨 내·외회전**(투구 부상 핵심 변수)은 2D로 신뢰 측정 불가 → 미산출.",
        "- 정량 검증은 참값 대조(Phase 4) 후에만 결론 가능. 현 값은 상대 비교·추세용.",
        "",
        "> 정상값 출처: Diffendaffer & Fleisig, *Sports Health* 2022; Fleisig et al. 1995.",
    ]
    return "\n".join(lines)


def plot_kinematics(png_path, angles, hand, events, fps, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 한글 라벨용 폰트(라벨 생성 전에 설정해야 적용됨). 없으면 무시.
    for fam in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic"):
        try:
            matplotlib.rcParams["font.family"] = fam
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    thr, lead = hand["throw"], hand["lead"]
    t = np.arange(n) / fps
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    def mark(a):
        for key, col, lab in [("leg_lift_peak", "tab:green", "다리들기"),
                              ("foot_contact", "tab:orange", "앞발착지"),
                              ("release", "tab:red", "릴리스")]:
            a.axvline(events[key] / fps, color=col, ls="--", lw=1.2, label=lab)

    ax[0].plot(t, _smooth(angles[f"{thr}_elbow"]), color="tab:blue")
    ax[0].set_ylabel("던지는 팔꿈치\n굽힘(°)"); mark(ax[0])
    ax[0].legend(loc="upper left", fontsize=8, ncol=3)

    ax[1].plot(t, _smooth(angles[f"{lead}_knee"]), color="tab:purple")
    ax[1].set_ylabel("리드 무릎\n굽힘(°)"); mark(ax[1])

    ax[2].plot(t, _smooth(angles["trunk_tilt"]), color="tab:brown")
    ax[2].axhline(0, color="gray", lw=0.6)
    ax[2].set_ylabel("몸통 기울기\n(연직=0°)"); mark(ax[2])

    ax[3].plot(t, _smooth(events["_wr_speed"]), color="tab:red")
    ax[3].set_ylabel("던지는 손목\n속도(px/s)"); mark(ax[3])
    ax[3].set_xlabel("시간 (s)")

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("투구 운동학 시계열 (단계 마커)", fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def analyze(csv_path, out_dir=None, hand_mode="auto", vis_thresh=0.3):
    kpts, times, fps = load_keypoints(csv_path)
    k = clean(kpts, vis_thresh)
    n = len(k)

    angles, sh_mid, hip_mid = compute_angles(k)
    if hand_mode == "auto":
        hand = detect_handedness(k, fps)
    else:
        thr = hand_mode.upper()[0]
        hand = {"throw": thr, "lead": "R" if thr == "L" else "L"}
    events = detect_events(k, hand, fps)
    stride_pct = stride_pct_height(k, hand, events)

    stem = os.path.splitext(csv_path)[0].replace("_keypoints", "")
    if out_dir:
        stem = os.path.join(out_dir, os.path.basename(stem))
        os.makedirs(out_dir, exist_ok=True)

    # 각도 CSV.
    _save_angle_csv(stem + "_angles.csv", angles, events["_wr_speed"], times, fps)
    # 그래프.
    plot_kinematics(stem + "_kinematics.png", angles, hand, events, fps, n)
    # 리포트.
    report = build_report(stem, angles, hand, events, fps, n, stride_pct)
    with open(stem + "_report.md", "w") as f:
        f.write(report)

    print(f"[운동학] {csv_path}")
    print(f"  던지는 팔={hand['throw']} 리드다리={hand['lead']} | "
          f"이벤트: 들기={events['leg_lift_peak']} 착지={events['foot_contact']} "
          f"릴리스={events['release']} (fps={fps:.0f})")
    print(f"  → {stem}_angles.csv / _kinematics.png / _report.md")
    return {"hand": hand, "events": {k_: v for k_, v in events.items() if not k_.startswith("_")}}


def _save_angle_csv(path, angles, wr_speed, times, fps):
    import pandas as pd

    data = {"frame": np.arange(len(times)), "time_s": times}
    for name, arr in angles.items():
        data[name + "_deg"] = np.round(arr, 2)
    data["throw_wrist_speed_pxs"] = np.round(wr_speed, 1)
    pd.DataFrame(data).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="투구 운동학 분석(관절각·단계분할)")
    ap.add_argument("csv", help="keypoint CSV (analyze.py 출력)")
    ap.add_argument("--out", help="출력 디렉토리(기본: CSV와 동일 위치)")
    ap.add_argument("--hand", default="auto", help="던지는 팔: auto|L|R")
    ap.add_argument("--vis", type=float, default=0.3, help="visibility 임계값")
    args = ap.parse_args()
    analyze(args.csv, args.out, args.hand, args.vis)


if __name__ == "__main__":
    main()

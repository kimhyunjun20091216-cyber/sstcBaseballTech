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
    - 어깨 내·외회전(횡단면)은 2D로 신뢰 불가 -> 산출하지 않음(계획서 1.2).
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

# 성인/프로 정상값(계획서 1.1, Diffendaffer & Fleisig 2022 등).
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
    flexion = 180 - 내각 -> 완전히 편 상태 0 deg, 굽힐수록 증가.
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


def _unwrap_deg(theta_deg):
    return np.degrees(np.unwrap(np.radians(theta_deg)))


def _ang_speed_deg_s(theta_deg, fps):
    unwrapped = _unwrap_deg(theta_deg)
    return np.gradient(unwrapped) * fps


def _line_angle_deg(a, b):
    vec = b - a
    return np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))


def _smallest_angle_diff_deg(a_deg, b_deg):
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


def compute_student_metrics(k, angles, hand, events, fps):
    """학생용 3축 지표(스피드/구조/타이밍) 계산."""
    thr = hand["throw"]
    lead = hand["lead"]

    l_hip = xy(k, L_HIP)
    r_hip = xy(k, R_HIP)
    l_sh = xy(k, L_SH)
    r_sh = xy(k, R_SH)

    pelvis_angle = _line_angle_deg(l_hip, r_hip)
    trunk_angle = _line_angle_deg(l_sh, r_sh)

    pelvis_speed = np.abs(_ang_speed_deg_s(pelvis_angle, fps))
    trunk_speed = np.abs(_ang_speed_deg_s(trunk_angle, fps))

    if thr == "R":
        sh = xy(k, R_SH)
        el = xy(k, R_EL)
        wr = xy(k, R_WR)
    else:
        sh = xy(k, L_SH)
        el = xy(k, L_EL)
        wr = xy(k, L_WR)

    upper_arm_angle = _line_angle_deg(sh, el)
    forearm_angle = _line_angle_deg(el, wr)
    upper_arm_speed = np.abs(_ang_speed_deg_s(upper_arm_angle, fps))
    forearm_speed = np.abs(_ang_speed_deg_s(forearm_angle, fps))
    arm_speed = 0.6 * upper_arm_speed + 0.4 * forearm_speed

    hip_shoulder_sep = np.abs(_smallest_angle_diff_deg(trunk_angle, pelvis_angle))

    phase_start = events["leg_lift_peak"]
    phase_end = events["release"]
    if phase_end <= phase_start:
        phase_start = 0
        phase_end = len(k) - 1
    seg = slice(phase_start, phase_end + 1)

    pelvis_peak = phase_start + int(np.argmax(pelvis_speed[seg]))
    trunk_peak = phase_start + int(np.argmax(trunk_speed[seg]))
    arm_peak = phase_start + int(np.argmax(arm_speed[seg]))

    metrics = {
        "pelvis_speed": pelvis_speed,
        "trunk_speed": trunk_speed,
        "arm_speed": arm_speed,
        "hip_shoulder_sep": hip_shoulder_sep,
        "lead_knee_flex": angles[f"{lead}_knee"],
        "trunk_tilt": angles["trunk_tilt"],
        "throw_elbow_flex": angles[f"{thr}_elbow"],
    }

    timing = {
        "pelvis_peak": pelvis_peak,
        "trunk_peak": trunk_peak,
        "arm_peak": arm_peak,
        "leg_lift_peak": events["leg_lift_peak"],
        "foot_contact": events["foot_contact"],
        "release": events["release"],
    }
    return metrics, timing


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


def summarize_speed_comparison(metrics, timing, fps):
    """세 속도 곡선의 피크 순서와 릴리스 대비 간격을 요약한다."""
    peak_frames = {
        "pelvis": int(timing["pelvis_peak"]),
        "trunk": int(timing["trunk_peak"]),
        "arm": int(timing["arm_peak"]),
    }
    peak_times_s = {name: frame / fps for name, frame in peak_frames.items()}
    release_frame = int(timing["release"])
    release_lag_s = {name: (release_frame - frame) / fps for name, frame in peak_frames.items()}
    peak_sequence = [name for name, _ in sorted(peak_frames.items(), key=lambda item: item[1])]
    labels = {"pelvis": "골반", "trunk": "몸통", "arm": "팔"}
    summary_text = (
        f"{labels[peak_sequence[0]]} -> {labels[peak_sequence[1]]} -> {labels[peak_sequence[2]]} 순으로 속도 피크가 나타났고, "
        f"릴리스 대비 차이는 골반 {release_lag_s['pelvis']:+.2f}s, 몸통 {release_lag_s['trunk']:+.2f}s, 팔 {release_lag_s['arm']:+.2f}s입니다."
    )
    return {
        "peak_frames": peak_frames,
        "peak_times_s": peak_times_s,
        "pelvis_peak_s": peak_times_s["pelvis"],
        "trunk_peak_s": peak_times_s["trunk"],
        "arm_peak_s": peak_times_s["arm"],
        "release_lag_s": release_lag_s,
        "peak_sequence": peak_sequence,
        "summary_text": summary_text,
    }


def _window_bounds(timing, n):
    start = int(max(0, min(timing.get("leg_lift_peak", 0), n - 1)))
    end = int(max(start, min(timing.get("release", n - 1), n - 1)))
    return start, end


def _curve_phase_stats(values, timing, fps, start_ratio=0.2, sustain_ratio=0.8):
    smooth = _smooth(values)
    n = len(smooth)
    start, end = _window_bounds(timing, n)
    seg = np.asarray(smooth[start:end + 1], dtype=np.float64)
    if len(seg) == 0:
        return {
            "peak_idx": 0,
            "peak_value": 0.0,
            "peak_time_s": 0.0,
            "accel_start_idx": 0,
            "accel_start_time_s": 0.0,
            "ramp_duration_s": 0.0,
            "sustain_duration_s": 0.0,
            "release_gap_s": 0.0,
        }

    peak_offset = int(np.nanargmax(seg))
    peak_idx = start + peak_offset
    peak_value = float(seg[peak_offset])
    start_threshold = peak_value * start_ratio
    accel_candidates = np.where(seg[:peak_offset + 1] >= start_threshold)[0]
    accel_start_offset = int(accel_candidates[0]) if len(accel_candidates) else 0
    accel_start_idx = start + accel_start_offset

    sustain_threshold = peak_value * sustain_ratio
    left = peak_offset
    while left > 0 and seg[left - 1] >= sustain_threshold:
        left -= 1
    right = peak_offset
    while right < len(seg) - 1 and seg[right + 1] >= sustain_threshold:
        right += 1

    return {
        "peak_idx": peak_idx,
        "peak_value": peak_value,
        "peak_time_s": peak_idx / fps,
        "accel_start_idx": accel_start_idx,
        "accel_start_time_s": accel_start_idx / fps,
        "ramp_duration_s": max(0.0, (peak_idx - accel_start_idx) / fps),
        "sustain_duration_s": max(0.0, (right - left) / fps),
        "release_gap_s": max(0.0, (timing["release"] - peak_idx) / fps),
    }


def summarize_reference_comparison(user_result, ref_result):
    labels = {"pelvis": "골반", "trunk": "몸통", "arm": "팔"}
    metric_names = {
        "pelvis": "pelvis_speed",
        "trunk": "trunk_speed",
        "arm": "arm_speed",
    }

    sections = {}
    for key, metric_name in metric_names.items():
        user_stats = _curve_phase_stats(user_result["metrics"][metric_name], user_result["timing"], user_result["fps"])
        ref_stats = _curve_phase_stats(ref_result["metrics"][metric_name], ref_result["timing"], ref_result["fps"])
        peak_gap = ref_stats["peak_value"] - user_stats["peak_value"]
        peak_ratio = (user_stats["peak_value"] / ref_stats["peak_value"] * 100.0) if ref_stats["peak_value"] > 1e-9 else 0.0
        sections[key] = {
            "label": labels[key],
            "user": user_stats,
            "reference": ref_stats,
            "peak_gap": peak_gap,
            "peak_ratio_pct": peak_ratio,
            "peak_gain_needed_pct": max(0.0, 100.0 - peak_ratio),
            "start_diff_s": user_stats["accel_start_time_s"] - ref_stats["accel_start_time_s"],
            "ramp_diff_s": user_stats["ramp_duration_s"] - ref_stats["ramp_duration_s"],
            "sustain_diff_s": user_stats["sustain_duration_s"] - ref_stats["sustain_duration_s"],
            "release_gap_diff_s": user_stats["release_gap_s"] - ref_stats["release_gap_s"],
        }

    user_intervals = {
        "pelvis_to_trunk_s": (user_result["timing"]["trunk_peak"] - user_result["timing"]["pelvis_peak"]) / user_result["fps"],
        "trunk_to_arm_s": (user_result["timing"]["arm_peak"] - user_result["timing"]["trunk_peak"]) / user_result["fps"],
    }
    ref_intervals = {
        "pelvis_to_trunk_s": (ref_result["timing"]["trunk_peak"] - ref_result["timing"]["pelvis_peak"]) / ref_result["fps"],
        "trunk_to_arm_s": (ref_result["timing"]["arm_peak"] - ref_result["timing"]["trunk_peak"]) / ref_result["fps"],
    }
    transfer = {
        "user_trunk_from_pelvis": user_result["metrics"]["trunk_speed"].max() / max(user_result["metrics"]["pelvis_speed"].max(), 1e-9),
        "user_arm_from_trunk": user_result["metrics"]["arm_speed"].max() / max(user_result["metrics"]["trunk_speed"].max(), 1e-9),
        "ref_trunk_from_pelvis": ref_result["metrics"]["trunk_speed"].max() / max(ref_result["metrics"]["pelvis_speed"].max(), 1e-9),
        "ref_arm_from_trunk": ref_result["metrics"]["arm_speed"].max() / max(ref_result["metrics"]["trunk_speed"].max(), 1e-9),
    }

    biggest_peak_deficit = max(sections.values(), key=lambda item: item["peak_gain_needed_pct"])
    latest_start = max(sections.values(), key=lambda item: item["start_diff_s"])
    shortest_sustain = min(sections.values(), key=lambda item: item["sustain_diff_s"])

    insights = []
    if biggest_peak_deficit["peak_gap"] > 0:
        insights.append(
            f"가장 큰 최고속도 격차는 {biggest_peak_deficit['label']}에서 나타납니다. "
            f"선수 대비 {biggest_peak_deficit['peak_ratio_pct']:.0f}% 수준이라 약 {biggest_peak_deficit['peak_gain_needed_pct']:.0f}% 추가 향상이 필요합니다."
        )
    if latest_start["start_diff_s"] > 0.01:
        insights.append(
            f"{latest_start['label']} 가속 시작이 선수보다 {latest_start['start_diff_s']:+.2f}s 늦습니다. "
            f"동작 초반에 이 분절을 더 일찍 열어야 합니다."
        )
    if shortest_sustain["sustain_diff_s"] < -0.01:
        insights.append(
            f"{shortest_sustain['label']} 고속 유지시간이 선수보다 {abs(shortest_sustain['sustain_diff_s']):.2f}s 짧습니다. "
            f"최고속도 근처의 힘 전달을 더 오래 유지할 필요가 있습니다."
        )

    chain_notes = []
    pelvis_gap = user_intervals["pelvis_to_trunk_s"] - ref_intervals["pelvis_to_trunk_s"]
    trunk_gap = user_intervals["trunk_to_arm_s"] - ref_intervals["trunk_to_arm_s"]
    if pelvis_gap < -0.01:
        chain_notes.append("골반에서 몸통으로 넘어가는 간격이 선수보다 짧아 하체 리드가 급하게 소모됩니다.")
    elif pelvis_gap > 0.01:
        chain_notes.append("골반 이후 몸통이 붙는 시점이 선수보다 늦어 에너지 전달이 끊길 수 있습니다.")
    if trunk_gap < -0.01:
        chain_notes.append("몸통에서 팔로 넘어가는 간격이 너무 짧아 상체가 급하게 열리는 패턴입니다.")
    elif trunk_gap > 0.01:
        chain_notes.append("팔이 선수보다 늦게 따라와 릴리스 직전 가속이 뒤로 밀릴 수 있습니다.")

    priorities = [
        f"1) {biggest_peak_deficit['label']} 최고속도를 선수 대비 {biggest_peak_deficit['peak_ratio_pct']:.0f}%에서 더 끌어올리기",
        f"2) {latest_start['label']} 가속 시작을 현재보다 {max(0.0, latest_start['start_diff_s']):.2f}s 빠르게 가져가기",
        f"3) {shortest_sustain['label']} 고속 유지시간을 선수 수준에 가깝게 늘리기",
    ]

    return {
        "sections": sections,
        "user_intervals": user_intervals,
        "reference_intervals": ref_intervals,
        "transfer": transfer,
        "insights": insights,
        "chain_notes": chain_notes,
        "priorities": priorities,
    }


def write_reference_comparison_summary(path, comparison):
    lines = [
        "# 유저 vs 선수 비교 요약",
        "",
        "| 분절 | 유저 최고속도 | 선수 최고속도 | 차이 | 선수 대비 수준 | 더 필요한 증가량 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("pelvis", "trunk", "arm"):
        section = comparison["sections"][key]
        lines.append(
            f"| {section['label']} | {section['user']['peak_value']:.1f} | {section['reference']['peak_value']:.1f} | "
            f"{section['peak_gap']:+.1f} | {section['peak_ratio_pct']:.0f}% | {section['peak_gain_needed_pct']:.0f}% |"
        )
    lines.extend([
        "",
        "| 분절 | 가속 시작 차이(s) | 피크까지 지속시간 차이(s) | 고속 유지시간 차이(s) |",
        "|---|---:|---:|---:|",
    ])
    for key in ("pelvis", "trunk", "arm"):
        section = comparison["sections"][key]
        lines.append(
            f"| {section['label']} | {section['start_diff_s']:+.2f} | {section['ramp_diff_s']:+.2f} | {section['sustain_diff_s']:+.2f} |"
        )
    lines.extend([
        "",
        f"- 골반 -> 몸통 간격: 유저 {comparison['user_intervals']['pelvis_to_trunk_s']:.2f}s / 선수 {comparison['reference_intervals']['pelvis_to_trunk_s']:.2f}s",
        f"- 몸통 -> 팔 간격: 유저 {comparison['user_intervals']['trunk_to_arm_s']:.2f}s / 선수 {comparison['reference_intervals']['trunk_to_arm_s']:.2f}s",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_reference_comparison_report(path, comparison):
    lines = [
        "# 학생선수 발전 포인트",
        "",
        "## 핵심 차이",
        "",
    ]
    for insight in comparison["insights"]:
        lines.append(f"- {insight}")
    if not comparison["insights"]:
        lines.append("- 유저와 선수의 주요 속도/타이밍 차이가 크지 않습니다. 세부 구조 지표를 우선 점검하세요.")
    lines.extend([
        "",
        "## 운동 사슬 전달 해석",
        "",
        f"- 몸통/골반 전달비: 유저 {comparison['transfer']['user_trunk_from_pelvis']:.2f} / 선수 {comparison['transfer']['ref_trunk_from_pelvis']:.2f}",
        f"- 팔/몸통 전달비: 유저 {comparison['transfer']['user_arm_from_trunk']:.2f} / 선수 {comparison['transfer']['ref_arm_from_trunk']:.2f}",
    ])
    for note in comparison["chain_notes"]:
        lines.append(f"- {note}")
    if not comparison["chain_notes"]:
        lines.append("- 피크 간 간격은 선수와 크게 다르지 않습니다. 절대 속도와 유지시간을 우선 개선해도 됩니다.")
    lines.extend([
        "",
        "## 우선 훈련 과제",
        "",
    ])
    for priority in comparison["priorities"]:
        lines.append(f"- {priority}")
    lines.extend([
        "",
        "## 코칭 해석",
        "",
        "- 가속 시작이 늦다면 동작 초반에 해당 분절이 늦게 열리는 패턴입니다.",
        "- 피크까지 지속시간이 짧다면 힘을 급하게 쓰고 끝나는 패턴일 수 있습니다.",
        "- 고속 유지시간이 짧다면 릴리스 직전까지 속도를 유지하는 능력을 더 키워야 합니다.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_speed_summary(path, summary):
    labels = {"pelvis": "골반", "trunk": "몸통", "arm": "팔"}
    lines = [
        "# 속도 비교 요약",
        "",
        summary["summary_text"],
        "",
        "| 항목 | 피크 시점(s) | 릴리스 대비 차이(s) |",
        "|---|---:|---:|",
    ]
    for name in ("pelvis", "trunk", "arm"):
        lines.append(f"| {labels[name]} | {summary['peak_times_s'][name]:.2f} | {summary['release_lag_s'][name]:+.2f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_report(stem, metrics, timing, hand, fps, n, stride_pct):
    rel = timing["release"]
    fc = timing["foot_contact"]
    summary = summarize_speed_comparison(metrics, timing, fps)

    lines = [
        f"# 학생용 투구 분석 리포트 - {os.path.basename(stem)}",
        "",
        f"- 프레임 수: {n}, 프레임레이트: {fps:.0f} fps, 길이: {n/fps:.1f}s",
        f"- 던지는 팔(추정): **{'왼손' if hand['throw']=='L' else '오른손'}**, "
        f"리드 다리: **{'왼다리' if hand['lead']=='L' else '오른다리'}**",
        "",
        "## 1) 스피드 데이터",
        "",
        f"- 속도 비교 요약: **{summary['summary_text']}**",
        f"- 피크 순서: **{' -> '.join(summary['peak_sequence'])}**",
        f"- 골반 회전 속도(최대): **{np.nanmax(metrics['pelvis_speed']):.1f} deg/s**",
        f"- 몸통 회전 속도(최대): **{np.nanmax(metrics['trunk_speed']):.1f} deg/s**",
        f"- 팔 속도(최대): **{np.nanmax(metrics['arm_speed']):.1f} deg/s**",
        "",
        "## 2) 구조 데이터",
        "",
        f"- 골반-몸통 분리각(앞발 착지): **{metrics['hip_shoulder_sep'][fc]:.1f} deg**",
        f"- 보폭(키 대비): **{stride_pct:.1f}%**" if not np.isnan(stride_pct) else "- 보폭(키 대비): 측정 불가",
        f"- 앞무릎 각도(착지): **{metrics['lead_knee_flex'][fc]:.1f} deg**",
        f"- 몸통 기울기(릴리스): **{abs(metrics['trunk_tilt'][rel]):.1f} deg**",
        "",
        "## 3) 타이밍 데이터",
        "",
        "| 항목 | 프레임 | 시각(s) |",
        "|---|---:|---:|",
        f"| 다리 들기 최고점 | {timing['leg_lift_peak']} | {timing['leg_lift_peak']/fps:.2f} |",
        f"| 앞발 착지 | {timing['foot_contact']} | {timing['foot_contact']/fps:.2f} |",
        f"| 골반 속도 피크 | {timing['pelvis_peak']} | {timing['pelvis_peak']/fps:.2f} |",
        f"| 몸통 속도 피크 | {timing['trunk_peak']} | {timing['trunk_peak']/fps:.2f} |",
        f"| 팔 속도 피크 | {timing['arm_peak']} | {timing['arm_peak']/fps:.2f} |",
        f"| 릴리스 | {timing['release']} | {timing['release']/fps:.2f} |",
        "",
        "- 이상적인 순서 예: 골반 속도 피크 -> 몸통 속도 피크 -> 팔 속도 피크 -> 릴리스",
        "",
        "## 주의",
        "",
        "- 본 결과는 2D 단일 카메라 기반 추정값이며, 같은 촬영 조건 내 비교에 적합합니다.",
    ]
    return "\n".join(lines)


def plot_compare_kinematics(png_path, series_a, series_b):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    for fam in (
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "AppleGothic",
        "Apple SD Gothic Neo",
        "Arial Unicode MS",
        "DejaVu Sans",
    ):
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = fam
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = [(series_a["label"], "#2B6CB0", "-"), (series_b["label"], "#C05621", "--")]
    peak_key_map = {
        "pelvis_speed": "pelvis_peak",
        "trunk_speed": "trunk_peak",
        "arm_speed": "arm_peak",
    }
    for ax, metric_name, title in zip(axes, ["pelvis_speed", "trunk_speed", "arm_speed"], ["골반 속도", "몸통 속도", "팔 속도"]):
        y_max = 0.0
        for label, color, linestyle in labels:
            series = series_a if label == series_a["label"] else series_b
            values = _smooth(series["metrics"][metric_name])
            n = len(values)
            t = np.linspace(0.0, 1.0, n)
            ax.plot(t, values, color=color, linestyle=linestyle, linewidth=2.0, label=f"{series['label']} {title}")
            y_max = max(y_max, float(np.nanmax(values)))
            peak_idx = int(np.nanargmax(values)) if n > 0 else 0
            peak_x = float(t[peak_idx]) if n > 0 else 0.0
            peak_y = float(values[peak_idx]) if n > 0 else 0.0
            ax.axhline(
                peak_y,
                color=color,
                linestyle=":",
                linewidth=1.8,
                alpha=0.95,
            )
            ax.scatter([peak_x], [peak_y], color=color, s=18, zorder=5)
        ax.set_ylabel("속도 (deg/s)")
        ax.set_title(title)
        ax.text(
            0.99,
            1.02,
            "가로 점선: 피크 위치 / 파란색: 유저 / 주황색: 선수",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            color="#444444",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#cccccc", "alpha": 0.85},
        )
        ax.grid(alpha=0.25)
        ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1.0)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("투구 진행 비율 (0~1)")
    fig.suptitle("유저 vs 선수 비교: 골반/몸통/팔 속도", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def plot_kinematics(png_path, metrics, timing, hand, fps, n, stride_pct):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    # 플랫폼별로 흔한 폰트를 우선 시도해 문자 깨짐을 줄인다.
    for fam in (
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "AppleGothic",
        "Apple SD Gothic Neo",
        "Arial Unicode MS",
        "DejaVu Sans",
    ):
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = fam
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    summary = summarize_speed_comparison(metrics, timing, fps)
    t = np.arange(n) / fps
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5), sharex=True)
    throw_txt = "왼손" if hand["throw"] == "L" else "오른손"

    ll_t = timing["leg_lift_peak"] / fps
    fc_t = timing["foot_contact"] / fps
    rel_t = timing["release"] / fps

    def phase_bg(a):
        a.axvspan(0, ll_t, color="#E8F1FF", alpha=0.5)
        a.axvspan(ll_t, fc_t, color="#EAF8EC", alpha=0.5)
        a.axvspan(fc_t, rel_t, color="#FFF3E6", alpha=0.5)

    def event_lines(a):
        a.axvline(rel_t, color="#D9534F", ls="--", lw=1.2, label="릴리스")

    pelvis_smooth = _smooth(metrics["pelvis_speed"])
    trunk_smooth = _smooth(metrics["trunk_speed"])
    arm_smooth = _smooth(metrics["arm_speed"])

    phase_bg(ax)
    ax.plot(t, pelvis_smooth, color="#2B6CB0", label="골반 회전 속도")
    ax.plot(t, trunk_smooth, color="#2F855A", label="몸통 회전 속도")
    ax.plot(t, arm_smooth, color="#C05621", label="팔 속도")
    event_lines(ax)

    speed_series = {"pelvis": pelvis_smooth, "trunk": trunk_smooth, "arm": arm_smooth}
    for name, color, label in (("pelvis", "#2B6CB0", "골반"), ("trunk", "#2F855A", "몸통"), ("arm", "#C05621", "팔")):
        peak_idx = int(timing[f"{name}_peak"])
        if 0 <= peak_idx < len(t):
            peak_t = summary["peak_times_s"][name]
            peak_y = speed_series[name][peak_idx]
            ax.scatter(peak_t, peak_y, color=color, zorder=5)
            ax.annotate(label, (peak_t, peak_y), xytext=(8, 6), textcoords="offset points", color=color, fontsize=8)

    ax.set_ylabel("속도 (deg/s)")
    ax.set_xlabel("시간 (s)")
    ax.set_title(f"투구 스피드 그래프 (피크 순서: {' -> '.join(summary['peak_sequence'])})")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(np.nanmax(pelvis_smooth), np.nanmax(trunk_smooth), np.nanmax(arm_smooth)) * 1.15)

    fig.suptitle("야구 투구 스피드 (골반/몸통/팔)", fontsize=13, y=0.985)
    fig.text(
        0.5,
        0.935,
        f"운동 데이터: 야구 피칭 | 던지는 팔: {throw_txt}",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
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
    metrics, timing = compute_student_metrics(k, angles, hand, events, fps)

    stem = os.path.splitext(csv_path)[0].replace("_keypoints", "")
    if out_dir:
        stem = os.path.join(out_dir, os.path.basename(stem))
        os.makedirs(out_dir, exist_ok=True)

    # 학생용 3축 CSV.
    _save_angle_csv(stem + "_angles.csv", metrics, timing, times, fps)
    # 학생용 3축 그래프.
    plot_kinematics(stem + "_kinematics.png", metrics, timing, hand, fps, n, stride_pct)
    # 학생용 3축 리포트.
    summary = summarize_speed_comparison(metrics, timing, fps)
    report = build_report(stem, metrics, timing, hand, fps, n, stride_pct)
    with open(stem + "_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    write_speed_summary(stem + "_speed_summary.md", summary)

    print(f"[운동학] {csv_path}")
    print(f"  던지는 팔={hand['throw']} 리드다리={hand['lead']} | "
          f"이벤트: 들기={timing['leg_lift_peak']} 착지={timing['foot_contact']} "
          f"릴리스={timing['release']} (fps={fps:.0f})")
    print(f"  → {stem}_angles.csv / _kinematics.png / _report.md")
    return {"hand": hand, "timing": timing, "metrics": metrics, "fps": fps, "n": n, "stride_pct": stride_pct, "stem": stem}


def _save_angle_csv(path, metrics, timing, times, fps):
    import pandas as pd

    data = {"frame": np.arange(len(times)), "time_s": times}
    data["pelvis_speed_deg_s"] = np.round(metrics["pelvis_speed"], 2)
    data["trunk_speed_deg_s"] = np.round(metrics["trunk_speed"], 2)
    data["arm_speed_deg_s"] = np.round(metrics["arm_speed"], 2)
    data["hip_shoulder_sep_deg"] = np.round(metrics["hip_shoulder_sep"], 2)
    data["lead_knee_flex_deg"] = np.round(metrics["lead_knee_flex"], 2)
    data["trunk_tilt_deg"] = np.round(metrics["trunk_tilt"], 2)
    data["throw_elbow_flex_deg"] = np.round(metrics["throw_elbow_flex"], 2)
    data["leg_lift_peak_frame"] = timing["leg_lift_peak"]
    data["foot_contact_frame"] = timing["foot_contact"]
    data["pelvis_peak_frame"] = timing["pelvis_peak"]
    data["trunk_peak_frame"] = timing["trunk_peak"]
    data["arm_peak_frame"] = timing["arm_peak"]
    data["release_frame"] = timing["release"]
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

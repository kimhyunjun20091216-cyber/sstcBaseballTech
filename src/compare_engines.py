"""자세추정 엔진 정량 비교 — MediaPipe vs YOLO-pose.

동일 영상에 대한 두 엔진의 keypoint CSV를 받아 검출률·신뢰도·시간적 지터·
엔진 간 일치도를 계산하고 out/engine_comparison.md 리포트를 생성한다.

사용:
    python src/compare_engines.py \
        --pairs "측면:out/영상1_측면_keypoints.csv:out/영상1_측면_yolo_keypoints.csv" \
                "정면:out/영상2_정면_keypoints.csv:out/영상2_정면_yolo_keypoints.csv"

지표 정의(2D 픽셀 기준):
  - 검출률: 유효 관절이 하나라도 있는 프레임 비율.
  - 평균 신뢰도: 관절별 visibility/confidence 채널 평균.
  - 시간적 지터(proxy): 관절 좌표의 프레임간 이동거리 평균/중앙값(px).
    ※ 실제 빠른 동작과 노이즈를 분리하지 않은 원시값 → 중앙값이 더 강건.
  - 엔진 간 일치도: 두 엔진이 모두 검출한 프레임에서 관절별 유클리드 거리 평균,
    몸통 길이(어깨중점-골반중점)로 정규화한 값도 병기.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

COCO = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def load(csv_path):
    """CSV → (xy (N,17,2), vis (N,17))."""
    df = pd.read_csv(csv_path)
    n = len(df)
    xy = np.full((n, 17, 2), np.nan)
    vis = np.full((n, 17), np.nan)
    for j, nm in enumerate(COCO):
        if f"{nm}_x" in df:
            xy[:, j, 0] = df[f"{nm}_x"]
            xy[:, j, 1] = df[f"{nm}_y"]
            vis[:, j] = df[f"{nm}_v"]
    return xy, vis


def detection_rate(xy):
    valid = ~np.isnan(xy[:, :, 0])
    return float(valid.any(axis=1).mean())


def mean_conf(vis):
    return float(np.nanmean(vis))


def jitter(xy):
    """프레임간 관절 이동거리(px)의 평균/중앙값(전 관절 통합)."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=2)  # (N-1, 17)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan
    return float(np.mean(d)), float(np.median(d))


def torso_len(xy):
    sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    d = np.linalg.norm(sh - hip, axis=1)
    return np.nanmedian(d)


def agreement(xy_a, xy_b):
    """두 엔진 관절별 유클리드 거리 평균(px) + 몸통정규화값."""
    n = min(len(xy_a), len(xy_b))
    a, b = xy_a[:n], xy_b[:n]
    both = (~np.isnan(a[:, :, 0])) & (~np.isnan(b[:, :, 0]))
    d = np.linalg.norm(a - b, axis=2)
    d = d[both]
    if d.size == 0:
        return np.nan, np.nan
    px = float(np.mean(d))
    tl = np.nanmean([torso_len(a), torso_len(b)])
    return px, (px / tl if tl and tl > 0 else np.nan)


def compare_pair(label, mp_csv, yolo_csv):
    mp_xy, mp_vis = load(mp_csv)
    yo_xy, yo_vis = load(yolo_csv)
    mp_jm, mp_jmed = jitter(mp_xy)
    yo_jm, yo_jmed = jitter(yo_xy)
    agr_px, agr_norm = agreement(mp_xy, yo_xy)
    return {
        "label": label,
        "mp_det": detection_rate(mp_xy),
        "yo_det": detection_rate(yo_xy),
        "mp_conf": mean_conf(mp_vis),
        "yo_conf": mean_conf(yo_vis),
        "mp_jit_mean": mp_jm, "mp_jit_med": mp_jmed,
        "yo_jit_mean": yo_jm, "yo_jit_med": yo_jmed,
        "agr_px": agr_px, "agr_norm": agr_norm,
        "torso_px": np.nanmean([torso_len(mp_xy), torso_len(yo_xy)]),
    }


def build_report(rows):
    L = ["# 자세추정 엔진 비교 — MediaPipe vs YOLO11x-pose", ""]
    L += [
        "MediaPipe(BlazePose Heavy, Tasks API) 대 YOLO11x-pose를 동일 투구 영상에서",
        "정량 비교. 값은 스무딩 **적용된** keypoint CSV 기준(파이프라인 최종 출력).",
        "",
        "## 검출률 · 평균 신뢰도",
        "",
        "| 영상 | MP 검출률 | YOLO 검출률 | MP 신뢰도 | YOLO 신뢰도 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['label']} | {r['mp_det']*100:.0f}% | {r['yo_det']*100:.0f}% | "
                 f"{r['mp_conf']:.2f} | {r['yo_conf']:.2f} |")
    L += [
        "",
        "## 시간적 지터 (프레임간 이동거리, px — 낮을수록 안정)",
        "",
        "| 영상 | MP 평균 | MP 중앙값 | YOLO 평균 | YOLO 중앙값 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['label']} | {r['mp_jit_mean']:.1f} | {r['mp_jit_med']:.1f} | "
                 f"{r['yo_jit_mean']:.1f} | {r['yo_jit_med']:.1f} |")
    L += [
        "",
        "*중앙값이 평균보다 노이즈에 강건(빠른 릴리스 프레임의 큰 이동을 완화). "
        "두 엔진의 지터 차이가 곧 스무딩 후 안정성 차이.*",
        "",
        "## 엔진 간 일치도 (두 엔진 모두 검출한 프레임)",
        "",
        "| 영상 | 평균 관절거리(px) | 몸통 대비(%) | 몸통길이(px) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        norm = f"{r['agr_norm']*100:.0f}%" if np.isfinite(r['agr_norm']) else "–"
        L.append(f"| {r['label']} | {r['agr_px']:.1f} | {norm} | {r['torso_px']:.0f} |")
    L += [
        "",
        "*몸통길이(어깨-골반)로 정규화 → 피사체 크기 차이를 보정. 값이 작을수록 두 "
        "엔진이 동일 지점을 가리킴(어느 쪽이 정답인지는 참값 대조(Phase 4) 필요).*",
        "",
        "## 결론 / 권고",
        "",
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="MediaPipe vs YOLO 엔진 비교")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help='"라벨:mediapipe_csv:yolo_csv" 형식(콜론 구분)')
    ap.add_argument("--out", default="out/engine_comparison.md")
    args = ap.parse_args()

    rows = []
    for spec in args.pairs:
        label, mp_csv, yo_csv = spec.split(":")
        rows.append(compare_pair(label, mp_csv, yo_csv))
        r = rows[-1]
        print(f"[{label}] MP검출 {r['mp_det']*100:.0f}% / YOLO검출 {r['yo_det']*100:.0f}% | "
              f"지터(중앙) MP {r['mp_jit_med']:.1f} vs YOLO {r['yo_jit_med']:.1f} px | "
              f"일치 {r['agr_px']:.1f}px ({r['agr_norm']*100:.0f}% 몸통)")

    report = build_report(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"→ {args.out} (결론 단락은 수동/후속 보강)")
    return rows


if __name__ == "__main__":
    main()

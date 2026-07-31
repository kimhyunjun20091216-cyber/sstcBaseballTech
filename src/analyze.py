"""투구 동작 분석 파이프라인 — 스켈레톤("졸라맨") 오버레이 생성.

영상 → 자세추정 → 스무딩 → 스켈레톤 오버레이 mp4 + 관절좌표 CSV.

사용 예:
    python src/analyze.py rec/영상.mp4 -o out/영상_pose.mp4 --engine mediapipe

Phase 1(핵심 요구사항: 두 각도 각각 졸라맨 생성)의 산출물을 만든다.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_engine import build_engine          # noqa: E402
from smoothing import smooth_sequence          # noqa: E402


# 좌/우를 색으로 구분(측/정면 판독 편의). BGR.
_LEFT_COLOR = (0, 180, 255)    # 주황
_RIGHT_COLOR = (255, 180, 0)   # 하늘
_LEFT_IDX = {5, 7, 9, 11, 13, 15}


def _edge_color(a, b):
    if a in _LEFT_IDX and b in _LEFT_IDX:
        return _LEFT_COLOR
    if a not in _LEFT_IDX and b not in _LEFT_IDX and a != 0 and b != 0:
        return _RIGHT_COLOR
    return (230, 230, 230)      # 중앙(몸통/머리)


def draw_skeleton(frame, kpts, edges, vis_thresh=0.3):
    """kpts (K,3) 오버레이. visibility 낮은 점/선은 생략."""
    if kpts is None:
        return frame
    h, w = frame.shape[:2]

    def ok(i):
        x, y, v = kpts[i]
        return not (np.isnan(x) or np.isnan(y)) and v >= vis_thresh and 0 <= x < w and 0 <= y < h

    for a, b in edges:
        if ok(a) and ok(b):
            pa = (int(kpts[a, 0]), int(kpts[a, 1]))
            pb = (int(kpts[b, 0]), int(kpts[b, 1]))
            cv2.line(frame, pa, pb, _edge_color(a, b), 3, cv2.LINE_AA)
    for i in range(kpts.shape[0]):
        if ok(i):
            c = _LEFT_COLOR if i in _LEFT_IDX else (_RIGHT_COLOR if i != 0 else (255, 255, 255))
            cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 4, c, -1, cv2.LINE_AA)
    return frame


def analyze_video(inp, outp, engine_name="mediapipe", csv_path=None,
                  smooth=True, preprocess=False, progress=True, **engine_kw):
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    engine = build_engine(engine_name, **engine_kw)

    # 1차 패스: 프레임 수집 + 자세추정.
    frames, raw_kpts = [], []
    t0 = time.time()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        proc = _enhance(frame) if preprocess else frame
        kpts = engine.estimate(proc)
        frames.append(frame)
        raw_kpts.append(kpts)
        idx += 1
        if progress and idx % 20 == 0:
            print(f"  추정 {idx}/{n_total} 프레임...", flush=True)
    cap.release()
    engine.close()

    detected = sum(k is not None for k in raw_kpts)
    print(f"  검출 성공: {detected}/{len(frames)} 프레임 "
          f"({100*detected/max(1,len(frames)):.0f}%)")

    # 스무딩.
    kpts_seq = smooth_sequence(raw_kpts, fps) if smooth else raw_kpts

    # 2차 패스: 렌더 + 저장.
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(outp, fourcc, fps, (w, h))
    for frame, kpts in zip(frames, kpts_seq):
        writer.write(draw_skeleton(frame, kpts, engine.edges))
    writer.release()

    if csv_path:
        _write_csv(csv_path, kpts_seq, engine.keypoint_names, fps)

    dt = time.time() - t0
    print(f"  완료: {outp}  ({dt:.1f}s, {len(frames)/max(dt,1e-6):.1f} fps 처리)")
    return {"frames": len(frames), "detected": detected, "fps": fps}


def _enhance(frame):
    """역광·저조도 보정: LAB 공간 CLAHE(명암 대비 향상)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _write_csv(path, kpts_seq, names, fps):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        header = ["frame", "time_s"]
        for nm in names:
            header += [f"{nm}_x", f"{nm}_y", f"{nm}_v"]
        wr.writerow(header)
        for i, kpts in enumerate(kpts_seq):
            row = [i, round(i / fps, 4)]
            if kpts is None:
                row += [""] * (3 * len(names))
            else:
                for k in range(len(names)):
                    row += [round(float(kpts[k, 0]), 2), round(float(kpts[k, 1]), 2),
                            round(float(kpts[k, 2]), 3)]
            wr.writerow(row)
    print(f"  좌표 CSV: {path}")


def main():
    ap = argparse.ArgumentParser(description="투구 스켈레톤 오버레이 생성")
    ap.add_argument("input", help="입력 영상 경로")
    ap.add_argument("-o", "--output", help="출력 mp4 경로")
    ap.add_argument("--engine", default="mediapipe", choices=["mediapipe", "yolo"])
    ap.add_argument("--csv", help="관절좌표 CSV 저장 경로")
    ap.add_argument("--no-smooth", action="store_true", help="스무딩 비활성")
    ap.add_argument("--preprocess", action="store_true", help="역광/저조도 보정(CLAHE)")
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + f"_{args.engine}.mp4"
    print(f"[분석] {args.input}  (engine={args.engine})")
    analyze_video(args.input, out, engine_name=args.engine, csv_path=args.csv,
                  smooth=not args.no_smooth, preprocess=args.preprocess)


if __name__ == "__main__":
    main()

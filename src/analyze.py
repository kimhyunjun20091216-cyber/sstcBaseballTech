"""투구 동작 분석 파이프라인 — 스켈레톤("졸라맨") 오버레이 생성.

영상 → 자세추정 → 스무딩 → 스켈레톤 오버레이 mp4 + 관절좌표 CSV.

사용 예:
    python src/analyze.py rec/영상.mp4 -o out/영상_pose.mp4 --engine yolo

Phase 1(핵심 요구사항: 두 각도 각각 졸라맨 생성)의 산출물을 만든다.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import importlib.util
import time
from html import escape
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _write_video_with_fallback(outp, frames, fps, width, height):
    """브라우저 재생이 가능한 MP4/WEBM 생성 시도.

    OpenCV가 사용할 수 있는 코덱 조합을 순차적으로 시도하고, 첫 번째로
    정상적으로 열고 프레임을 쓸 수 있는 파일을 반환한다. ffmpeg가 없어도
    동작할 수 있도록 mp4v/avi/webm 폴백을 사용한다.
    """
    outp = os.fspath(outp)
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)

    candidates = []
    if outp.lower().endswith(".mp4"):
        candidates.extend([
            ("mp4v", "mp4", "mp4v"),
            ("avc1", "mp4", "avc1"),
            ("H264", "mp4", "H264"),
            ("MJPG", "avi", "MJPG"),
        ])
    elif outp.lower().endswith(".webm"):
        candidates.extend([
            ("VP80", "webm", "VP80"),
            ("VP90", "webm", "VP90"),
        ])
    else:
        candidates.extend([("mp4v", "mp4", "mp4v")])

    for codec, ext, tag in candidates:
        path = outp
        if ext and not path.lower().endswith(f".{ext}"):
            path = os.path.splitext(outp)[0] + f".{ext}"
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        except Exception:
            continue
        if not writer.isOpened():
            writer.release()
            continue
        ok = True
        for frame in frames:
            if frame is None:
                continue
            if not writer.write(frame):
                ok = False
                break
        writer.release()
        if ok and os.path.exists(path) and os.path.getsize(path) > 0:
            if path != outp and os.path.exists(outp):
                try:
                    os.remove(outp)
                except OSError:
                    pass
            return path

    return None

from pose_engine import COCO_EDGES, COCO_KEYPOINTS, build_engine          # noqa: E402
from smoothing import smooth_sequence          # noqa: E402


# 좌/우를 색으로 구분(측/정면 판독 편의). BGR.
_LEFT_COLOR = (0, 180, 255)    # 주황
_RIGHT_COLOR = (255, 180, 0)   # 하늘
_LEFT_IDX = {5, 7, 9, 11, 13, 15}
_SKELETON_BG_CACHE = {}

COCO_BONES_17 = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# PoseFormer/H36M 17-joint -> COCO 17-joint mapping.
# Each COCO index points to its source index in H36M layout.
H36M_TO_COCO_17 = np.array(
    [
        9,   # nose <- neck/nose
        9,   # left_eye (unavailable): fallback to nose
        9,   # right_eye (unavailable): fallback to nose
        9,   # left_ear (unavailable): fallback to nose
        9,   # right_ear (unavailable): fallback to nose
        11,  # left_shoulder
        14,  # right_shoulder
        12,  # left_elbow
        15,  # right_elbow
        13,  # left_wrist
        16,  # right_wrist
        4,   # left_hip
        1,   # right_hip
        5,   # left_knee
        2,   # right_knee
        6,   # left_ankle
        3,   # right_ankle
    ],
    dtype=np.int64,
)


def _skeleton_space_background(height, width):
    """스켈레톤 전용 보기용 3D 공간 느낌 배경을 생성한다."""
    key = (height, width)
    cached = _SKELETON_BG_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    bg = np.zeros((height, width, 3), dtype=np.uint8)

    # 위는 짙고 아래는 살짝 밝은 그라디언트로 깊이감을 만든다.
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    bg[..., 0] = (16 + 28 * y).astype(np.uint8)
    bg[..., 1] = (8 + 18 * y).astype(np.uint8)
    bg[..., 2] = (18 + 42 * y).astype(np.uint8)

    horizon = int(height * 0.38)
    vanish_x = width // 2

    # 원근감 있는 바닥 격자.
    n_vertical = 18
    for i in range(n_vertical + 1):
        x0 = int(i * width / n_vertical)
        cv2.line(bg, (x0, height - 1), (vanish_x, horizon), (58, 82, 122), 1, cv2.LINE_AA)

    n_h = 15
    for j in range(1, n_h + 1):
        t = j / n_h
        yy = int(horizon + (height - horizon - 1) * (t ** 2.2))
        cv2.line(bg, (0, yy), (width - 1, yy), (42, 64, 98), 1, cv2.LINE_AA)

    # 별 포인트를 고정 시드로 찍어 프레임 간 깜빡임을 없앤다.
    rng = np.random.default_rng(1234)
    n_stars = max(90, (width * height) // 16000)
    xs = rng.integers(0, width, size=n_stars)
    ys = rng.integers(0, max(1, horizon), size=n_stars)
    for sx, sy in zip(xs, ys):
        c = int(rng.integers(170, 255))
        cv2.circle(bg, (int(sx), int(sy)), 1, (c, c, c), -1, cv2.LINE_AA)

    # 수평선 글로우.
    cv2.line(bg, (0, horizon), (width - 1, horizon), (86, 132, 220), 2, cv2.LINE_AA)

    _SKELETON_BG_CACHE[key] = bg
    return bg.copy()


def _edge_color(a, b):
    if a in _LEFT_IDX and b in _LEFT_IDX:
        return _LEFT_COLOR
    if a not in _LEFT_IDX and b not in _LEFT_IDX and a != 0 and b != 0:
        return _RIGHT_COLOR
    return (230, 230, 230)      # 중앙(몸통/머리)


def build_dynamic_viewer(frames, out_html, fps=30.0, title="투구 포즈 시각화"):
    """프레임 시퀀스를 브라우저에서 재생할 수 있는 standalone HTML로 저장한다."""
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)

    frame_sources = []
    for frame in frames:
        if frame is None:
            continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frame_sources.append(f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}")

    width = int(frames[0].shape[1]) if frames else 640
    height = int(frames[0].shape[0]) if frames else 360
    safe_title = escape(title)
    safe_fps = float(fps)
    frame_json = json.dumps(frame_sources)

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; background: #111; color: #fff; }}
        .wrap {{ max-width: 100%; margin: 0 auto; padding: 0; }}
        .panel {{ background: #1b1b1b; border-radius: 8px; padding: 2px; }}
        h2 {{ margin: 0 0 2px; font-size: 12px; line-height: 1.1; }}
        .viewport {{
            width: 100%;
            max-height: min(82vh, 860px);
            aspect-ratio: {width} / {height};
            object-fit: contain;
            display: block;
            margin: 0 auto;
            background: #000;
            border-radius: 8px;
        }}
        .controls {{
            display: flex;
            gap: 6px;
            align-items: center;
            margin-top: 4px;
            flex-wrap: nowrap;
            overflow: hidden;
        }}
        button {{ border: 0; border-radius: 6px; padding: 3px 7px; cursor: pointer; font-size: 11px; }}
        .badge {{ font-size: 10px; color: #aaa; white-space: nowrap; }}
    </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h2>{safe_title}</h2>
            <img id="poseFrame" class="viewport" alt="pose frame" />
      <div class="controls">
                <button id="togglePlayBtn">정지</button>
        <button id="stepBackBtn"><</button>
        <button id="stepFwdBtn">></button>
        <span class="badge" id="statusLabel">0 / {len(frame_sources)} 프레임</span>
      </div>
    </div>
  </div>
  <script>
    const frameSources = {frame_json};
    const fps = {safe_fps};
    const poseFrame = document.getElementById('poseFrame');
    const statusLabel = document.getElementById('statusLabel');
        const togglePlayBtn = document.getElementById('togglePlayBtn');
    const stepBackBtn = document.getElementById('stepBackBtn');
    const stepFwdBtn = document.getElementById('stepFwdBtn');
    const images = frameSources;
    let index = 0;
    let timer = null;

        function isPlaying() {{
            return timer !== null;
        }}

        function syncToggleLabel() {{
            togglePlayBtn.textContent = isPlaying() ? '정지' : '재생';
        }}

    function renderFrame() {{
      if (!images.length) return;
            poseFrame.src = images[index];
      statusLabel.textContent = `${{index + 1}} / ${{images.length}} 프레임`;
    }}

    function play() {{
      if (!images.length) return;
      if (timer) clearInterval(timer);
      timer = setInterval(() => {{
        index = (index + 1) % images.length;
        renderFrame();
      }}, 1000 / fps);
            syncToggleLabel();
    }}

    function pause() {{
      if (timer) clearInterval(timer);
      timer = null;
            syncToggleLabel();
    }}

    function step(delta) {{
      if (!images.length) return;
      index = (index + delta + images.length) % images.length;
      renderFrame();
    }}

        togglePlayBtn.addEventListener('click', () => {{
            if (isPlaying()) pause();
            else play();
        }});
    stepBackBtn.addEventListener('click', () => step(-1));
    stepFwdBtn.addEventListener('click', () => step(1));

    renderFrame();
    play();
  </script>
</body>
</html>''';

    with open(out_html, 'w', encoding='utf-8') as fh:
        fh.write(html)
    return out_html


def draw_skeleton(frame, kpts, edges, vis_thresh=0.3, skeleton_only=False):
    """kpts (K,3) 오버레이. visibility 낮은 점/선은 생략."""
    if kpts is None:
        return frame
    if skeleton_only:
        frame = _skeleton_space_background(frame.shape[0], frame.shape[1])
        vis_thresh = min(vis_thresh, 0.2)
    h, w = frame.shape[:2]

    def ok(i):
        x, y, v = kpts[i]
        return not (np.isnan(x) or np.isnan(y)) and v >= vis_thresh and 0 <= x < w and 0 <= y < h

    for a, b in edges:
        if ok(a) and ok(b):
            pa = (int(kpts[a, 0]), int(kpts[a, 1]))
            pb = (int(kpts[b, 0]), int(kpts[b, 1]))
            color = _edge_color(a, b)
            thickness = 5 if skeleton_only and color == _RIGHT_COLOR else 3
            cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)
    for i in range(kpts.shape[0]):
        if ok(i):
            c = _LEFT_COLOR if i in _LEFT_IDX else (_RIGHT_COLOR if i != 0 else (255, 255, 255))
            cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 4, c, -1, cv2.LINE_AA)
    return frame


def validate_analysis_inputs(inp, engine_name="yolo", **engine_kw):
    """영상 파일과 엔진 준비 상태를 사전에 검증한다.

    현재 파이프라인 안정화 단계에서 가장 자주 발생하는 문제는
    - 파일 경로 문제
    - 비디오 열기 실패
    - 엔진 모델 누락
    을 추론 시작 후에야 발견하는 것이다. 따라서 추론 전 미리 확인해
    오류를 명확하게 전달하도록 한다.
    """
    if not inp or not os.path.exists(inp):
        raise FileNotFoundError(f"영상 파일이 없습니다: {inp}")

    cap = cv2.VideoCapture(inp)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {inp}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_total <= 0:
            raise RuntimeError(f"프레임을 읽지 못했습니다. 영상이 손상되었거나 코덱이 지원되지 않을 수 있습니다: {inp}")
        if src_w <= 0 or src_h <= 0:
            raise RuntimeError(f"영상 크기를 읽지 못했습니다: {inp}")
        if fps <= 0:
            raise RuntimeError(f"영상 FPS를 읽지 못했습니다: {inp}")
    finally:
        cap.release()

    if engine_name == "yolo":
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights = engine_kw.get("weights", "yolo11x-pose.pt")
        candidates = [weights]
        candidates.append(os.path.join(root_dir, weights))
        if not any(os.path.exists(path) for path in candidates if path):
            raise FileNotFoundError(
                "YOLO pose weights가 없습니다. 프로젝트 루트에 yolo11x-pose.pt 를 두거나 "
                "weights 인자를 지정하세요."
            )
    elif engine_name == "rtmpose":
        repo_dir = engine_kw.get("repo_dir") or default_rtmpose_repo_dir()
        config_or_alias = engine_kw.get("config") or default_rtmpose_config_path(repo_dir)
        checkpoint = engine_kw.get("checkpoint")

        if repo_dir and not os.path.isdir(repo_dir):
            raise FileNotFoundError(f"RTMPose repo not found: {repo_dir}")
        if config_or_alias and os.path.sep in str(config_or_alias) and not os.path.exists(config_or_alias):
            raise FileNotFoundError(f"RTMPose config not found: {config_or_alias}")
        if not repo_dir and importlib.util.find_spec("mmpose") is None:
            raise FileNotFoundError(
                "RTMPose를 실행하려면 mmpose 패키지 또는 로컬 mmpose repo가 필요합니다."
            )
        if checkpoint and not os.path.exists(checkpoint):
            raise FileNotFoundError(f"RTMPose checkpoint not found: {checkpoint}")
    else:
        raise ValueError(f"unsupported engine: {engine_name!r}")

    return {
        "fps": float(fps),
        "width": int(src_w),
        "height": int(src_h),
        "frame_count": int(n_total),
    }


def default_poseformer_repo_dir():
    """프로젝트 내 기본 PoseFormerV2 저장소 경로를 찾는다."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(root_dir, "PoseFormerV2-main"),
        os.path.join(root_dir, "PoseFormerV2-main", "PoseFormerV2-main"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "run_poseformer.py")):
            return candidate
    return None


def default_poseformer_checkpoint_path(repo_dir=None):
    """프로젝트 내 기본 PoseFormerV2 체크포인트 경로를 찾는다."""
    roots = []
    if repo_dir:
        roots.append(repo_dir)
    default_repo = default_poseformer_repo_dir()
    if default_repo:
        roots.append(default_repo)
    candidates = []
    for root in roots:
        candidates.extend([
            os.path.join(root, "checkpoint", "poseformerv2_27f_3k_3c.bin"),
            os.path.join(root, "checkpoint", "27_3_3.bin"),
            os.path.join(root, "checkpoint", "best_epoch.bin"),
        ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def default_rtmpose_repo_dir():
    """RTMPose/MMPose 저장소 경로를 찾는다."""
    candidates = [
        os.path.join(os.path.expanduser("~"), "OneDrive", "바탕 화면", "mmpose-main", "mmpose-main"),
        os.path.join(os.path.expanduser("~"), "OneDrive", "바탕 화면", "mmpose-main"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mmpose-main", "mmpose-main"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mmpose-main"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "mmpose")):
            return candidate
    return ""


def default_rtmpose_config_path(repo_dir=None):
    """RTMPose 기본 config 파일 경로 또는 alias를 반환한다."""
    if repo_dir:
        candidates = [
            os.path.join(repo_dir, "configs", "body_2d_keypoint", "rtmpose", "coco", "rtmpose-m_8xb256-420e_coco-256x192.py"),
            os.path.join(repo_dir, "configs", "body_2d_keypoint", "rtmpose", "coco", "rtmpose-s_8xb256-420e_coco-256x192.py"),
            os.path.join(repo_dir, "configs", "body_2d_keypoint", "rtmpose", "coco", "rtmpose-t_8xb256-420e_coco-256x192.py"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return "rtmpose-m_8xb256-420e_coco-256x192"


def _load_keypoints_csv(csv_path, *, expected_names=None):
    """COCO keypoint CSV를 프레임별 (K,3) 시퀀스로 읽는다."""
    expected_names = list(expected_names or COCO_KEYPOINTS)
    required_columns = [f"{name}_x" for name in expected_names] + [f"{name}_y" for name in expected_names] + [f"{name}_v" for name in expected_names]

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"keypoints CSV has no header: {csv_path}")
        missing = [col for col in required_columns if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"keypoints CSV is missing columns: {missing}")

        kpts_seq = []
        for row in reader:
            joints = []
            for name in expected_names:
                x = row.get(f"{name}_x", "")
                y = row.get(f"{name}_y", "")
                v = row.get(f"{name}_v", "")
                if x == "" or y == "" or v == "":
                    joints.append([np.nan, np.nan, 0.0])
                else:
                    joints.append([float(x), float(y), float(v)])
            kpts_seq.append(np.asarray(joints, dtype=np.float32))

    if not kpts_seq:
        raise ValueError(f"keypoints CSV is empty: {csv_path}")
    return kpts_seq


def evaluate_2d_stability(kpts_seq):
    """2D 키포인트 시퀀스의 안정성을 숫자로 요약한다.

    목표는 PoseFormerV2를 붙이기 전에, 입력 시퀀스가 충분히 안정적인지
    정량적으로 판단하는 기준을 제공하는 것이다.
    """
    if not kpts_seq:
        raise ValueError("kpts_seq must not be empty")

    frames = len(kpts_seq)
    valid_frame_count = sum(1 for k in kpts_seq if k is not None)
    detection_rate = float(valid_frame_count / frames)

    valid_joint_count = 0
    total_joint_count = 0
    conf_values = []
    jitter_values = []

    for i, kpts in enumerate(kpts_seq):
        if kpts is None:
            continue
        total_joint_count += kpts.shape[0]
        valid_mask = np.isfinite(kpts[:, 0]) & np.isfinite(kpts[:, 1]) & (kpts[:, 2] > 0)
        valid_joint_count += int(valid_mask.sum())
        conf_values.extend(kpts[valid_mask, 2].tolist())

        if i == 0:
            continue
        prev = kpts_seq[i - 1]
        if prev is None:
            continue
        for a in range(min(len(kpts), len(prev))):
            if not (np.isfinite(kpts[a, 0]) and np.isfinite(kpts[a, 1]) and
                    np.isfinite(prev[a, 0]) and np.isfinite(prev[a, 1])):
                continue
            delta = np.linalg.norm(kpts[a, :2] - prev[a, :2])
            jitter_values.append(float(delta))

    mean_joint_confidence = float(np.mean(conf_values)) if conf_values else 0.0
    temporal_jitter_px = float(np.mean(jitter_values)) if jitter_values else 0.0
    missing_joint_ratio = 1.0 - (valid_joint_count / max(1, total_joint_count))

    return {
        "frames": frames,
        "valid_frames": valid_frame_count,
        "detection_rate": detection_rate,
        "mean_joint_confidence": mean_joint_confidence,
        "missing_joint_ratio": missing_joint_ratio,
        "temporal_jitter_px": temporal_jitter_px,
        "stability_score": float(
            0.55 * detection_rate +
            0.30 * min(1.0, mean_joint_confidence) +
            0.15 * max(0.0, 1.0 - min(1.0, temporal_jitter_px / 50.0))
        ),
    }


def assess_poseformer_readiness(kpts_seq, detection_threshold=0.9,
                               confidence_threshold=0.7, jitter_threshold=10.0,
                               missing_joint_threshold=0.1):
    """PoseFormerV2 연동 전 2D 입력 품질을 확인한다.

    readiness = True when all core conditions are met.
    """
    metrics = evaluate_2d_stability(kpts_seq)
    checks = {
        "detection_rate_ok": metrics["detection_rate"] >= detection_threshold,
        "confidence_ok": metrics["mean_joint_confidence"] >= confidence_threshold,
        "jitter_ok": metrics["temporal_jitter_px"] <= jitter_threshold,
        "missing_joint_ok": metrics["missing_joint_ratio"] <= missing_joint_threshold,
    }
    issues = [name for name, ok in checks.items() if not ok]
    return {
        "ready": all(checks.values()),
        "metrics": metrics,
        "checks": checks,
        "issues": issues,
    }


def _interpolate_nan_1d(values):
    arr = np.asarray(values, dtype=np.float32).copy()
    idx = np.arange(arr.shape[0], dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.full_like(arr, np.nan)
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid]).astype(np.float32)
    return arr


def stabilize_keypoints_single_camera(
    kpts_seq,
    min_confidence=0.2,
    spike_px=45.0,
    alpha_low_conf=0.25,
    alpha_high_conf=0.8,
):
    """단일 카메라용 2D 키포인트 안정화.

    - 저신뢰/결측 구간 보간
    - 프레임 간 스파이크 억제
    - confidence-aware EMA 스무딩
    """
    if not kpts_seq:
        return kpts_seq

    frames = []
    for k in kpts_seq:
        if k is None:
            frames.append(np.full((17, 3), np.nan, dtype=np.float32))
        else:
            arr = np.asarray(k, dtype=np.float32)
            if arr.shape != (17, 3):
                frames.append(np.full((17, 3), np.nan, dtype=np.float32))
            else:
                frames.append(arr.copy())

    seq = np.stack(frames, axis=0)
    T, J, _ = seq.shape

    for j in range(J):
        conf = seq[:, j, 2]
        x = seq[:, j, 0]
        y = seq[:, j, 1]

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(conf) & (conf >= float(min_confidence))
        x[~valid] = np.nan
        y[~valid] = np.nan

        x = _interpolate_nan_1d(x)
        y = _interpolate_nan_1d(y)

        if T >= 3:
            for t in range(1, T - 1):
                prev_xy = np.array([x[t - 1], y[t - 1]], dtype=np.float32)
                cur_xy = np.array([x[t], y[t]], dtype=np.float32)
                next_xy = np.array([x[t + 1], y[t + 1]], dtype=np.float32)
                med_xy = 0.5 * (prev_xy + next_xy)
                jump = float(np.linalg.norm(cur_xy - med_xy))
                if jump > float(spike_px):
                    c = float(conf[t]) if np.isfinite(conf[t]) else 0.0
                    w = np.clip((0.5 - c) / 0.5, 0.0, 1.0)
                    x[t] = (1.0 - w) * x[t] + w * med_xy[0]
                    y[t] = (1.0 - w) * y[t] + w * med_xy[1]

        x_s = x.copy()
        y_s = y.copy()
        for t in range(1, T):
            c = float(conf[t]) if np.isfinite(conf[t]) else 0.0
            c = float(np.clip(c, 0.0, 1.0))
            alpha = float(alpha_low_conf + (alpha_high_conf - alpha_low_conf) * c)
            x_s[t] = alpha * x[t] + (1.0 - alpha) * x_s[t - 1]
            y_s[t] = alpha * y[t] + (1.0 - alpha) * y_s[t - 1]

        seq[:, j, 0] = x_s
        seq[:, j, 1] = y_s
        seq[:, j, 2] = np.nan_to_num(conf, nan=0.0)

    out = []
    for t in range(T):
        frame = seq[t].astype(np.float32)
        if not np.isfinite(frame[:, :2]).any():
            out.append(None)
        else:
            out.append(frame)
    return out


def evaluate_input_joint_mapping_consistency_2d(kpts_seq):
    """2D 입력의 좌우 관절 매핑/비율 일관성을 점검한다."""
    if not kpts_seq:
        return {"valid": False, "reason": "empty_sequence"}

    frames = []
    for k in kpts_seq:
        if k is None:
            continue
        arr = np.asarray(k, dtype=np.float32)
        if arr.shape != (17, 3):
            continue
        frames.append(arr)
    if not frames:
        return {"valid": False, "reason": "no_valid_frames"}

    arr = np.stack(frames, axis=0)
    # 좌우 어깨/골반 X 방향 일관성(카메라 미러 여부 포함)
    sh_dx = arr[:, 6, 0] - arr[:, 5, 0]
    hip_dx = arr[:, 12, 0] - arr[:, 11, 0]
    sh_sign = np.sign(np.nanmedian(sh_dx))
    hip_sign = np.sign(np.nanmedian(hip_dx))
    lr_consistent = bool(sh_sign != 0 and hip_sign != 0 and sh_sign == hip_sign)

    # 좌우 팔 길이 대칭성 점검
    l_up = np.linalg.norm(arr[:, 7, :2] - arr[:, 5, :2], axis=1)
    r_up = np.linalg.norm(arr[:, 8, :2] - arr[:, 6, :2], axis=1)
    l_lo = np.linalg.norm(arr[:, 9, :2] - arr[:, 7, :2], axis=1)
    r_lo = np.linalg.norm(arr[:, 10, :2] - arr[:, 8, :2], axis=1)

    def _sym_err(a, b):
        av = a[np.isfinite(a) & (a > 1e-6)]
        bv = b[np.isfinite(b) & (b > 1e-6)]
        if av.size == 0 or bv.size == 0:
            return np.nan
        am = float(np.median(av))
        bm = float(np.median(bv))
        denom = max(1e-6, 0.5 * (am + bm))
        return abs(am - bm) / denom

    upper_sym = _sym_err(l_up, r_up)
    lower_sym = _sym_err(l_lo, r_lo)

    return {
        "valid": True,
        "left_right_consistent": lr_consistent,
        "left_right_swap_or_mirror_suspected": not lr_consistent,
        "upper_arm_symmetry_error_2d": upper_sym,
        "forearm_symmetry_error_2d": lower_sym,
        "pass_upper_arm_symmetry_2d": bool(np.isfinite(upper_sym) and upper_sym <= 0.20),
        "pass_forearm_symmetry_2d": bool(np.isfinite(lower_sym) and lower_sym <= 0.20),
    }


def validate_poseformer_output(pose_3d, min_depth_span=1e-4):
    """3D 출력이 후속 분석에 쓸 수 있는 최소 조건을 만족하는지 확인한다."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"pose_3d must have shape (T, J, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("PoseFormerV2 produced non-finite 3D coordinates")

    root = arr[:, 0, :]
    if not np.isfinite(root).all():
        raise ValueError("PoseFormerV2 root joint contains non-finite values")

    depth_span = float(np.max(arr[..., 2]) - np.min(arr[..., 2]))
    if depth_span <= float(min_depth_span):
        raise ValueError(
            f"PoseFormerV2 3D depth span is too small for reliable output: {depth_span:.6f}"
        )

    return {
        "frames": int(arr.shape[0]),
        "num_joints": int(arr.shape[1]),
        "depth_span": depth_span,
    }


def enforce_bone_length_consistency_3d(pose_3d, bones=None, strength=0.35, iterations=2):
    """프레임 간 뼈 길이 변동을 완화해 3D 시퀀스의 물리적 일관성을 높인다."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr

    out = arr.copy()
    bones = list(bones or COCO_BONES_17)
    if not bones:
        return out

    target_lengths = {}
    for a, b in bones:
        seg = out[:, b, :] - out[:, a, :]
        lens = np.linalg.norm(seg, axis=1)
        valid = np.isfinite(lens) & (lens > 1e-6)
        if np.any(valid):
            target_lengths[(a, b)] = float(np.median(lens[valid]))

    if not target_lengths:
        return out

    s = float(np.clip(strength, 0.0, 1.0))
    n_iter = max(1, int(iterations))
    for _ in range(n_iter):
        for t in range(out.shape[0]):
            frame = out[t]
            for (a, b), target_len in target_lengths.items():
                pa = frame[a]
                pb = frame[b]
                if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                    continue
                vec = pb - pa
                cur_len = float(np.linalg.norm(vec))
                if cur_len <= 1e-6:
                    continue
                desired_pb = pa + vec * (target_len / cur_len)
                frame[b] = (1.0 - s) * pb + s * desired_pb
            out[t] = frame

    return out


def canonicalize_vertical_axis_3d(pose_3d, hip_indices=(11, 12), ankle_indices=(15, 16)):
    """발목이 골반보다 아래에 오도록 상하축 방향을 정규화한다.

    좌표계의 Y축 방향이 뒤집힌 경우(발목 Y가 골반 Y보다 큰 방향이 주가 되는 경우)
    Y축 부호를 반전해 인체 상하 관계를 맞춘다.
    """
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr

    out = arr.copy()
    if out.shape[1] <= max(max(hip_indices), max(ankle_indices)):
        return out

    hip_y = np.nanmean(out[:, list(hip_indices), 1], axis=1)
    ankle_y = np.nanmean(out[:, list(ankle_indices), 1], axis=1)
    delta = ankle_y - hip_y
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return out

    # 프로젝트 기준에서 발은 Y가 더 작아야 아래로 해석된다.
    if float(np.median(delta)) > 0.0:
        out[..., 1] *= -1.0
    return out


def enforce_human_vertical_hierarchy_3d(
    pose_3d,
    chains=None,
    strength=0.6,
    iterations=2,
):
    """프레임별 수직 위계를 강제해 '졸라맨 형태'를 안정화한다.

    기본 체인은 (어깨, 골반, 무릎, 발목)이며,
    발목 <= 무릎 <= 골반 <= 어깨 관계를 Y축에서 보정한다.
    """
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr

    out = arr.copy()
    default_chains = [
        (5, 11, 13, 15),   # left
        (6, 12, 14, 16),   # right
    ]
    chains = list(chains or default_chains)
    if not chains:
        return out

    s = float(np.clip(strength, 0.0, 1.0))
    n_iter = max(1, int(iterations))
    n_joints = out.shape[1]

    for _ in range(n_iter):
        for t in range(out.shape[0]):
            frame = out[t]
            for shoulder, hip, knee, ankle in chains:
                if max(shoulder, hip, knee, ankle) >= n_joints:
                    continue
                idxs = [ankle, knee, hip, shoulder]
                y = frame[idxs, 1]
                if not np.isfinite(y).all():
                    continue

                target_y = np.maximum.accumulate(y)
                frame[idxs, 1] = (1.0 - s) * y + s * target_y
            out[t] = frame

    return out


def canonicalize_body_frame_3d(pose_3d):
    """골반-어깨 축 기준으로 3D 좌표계를 정규화한다.

    - Y: 골반->어깨(상하)
    - X: 좌어깨->우어깨(좌우)
    - Z: X x Y (전후)
    """
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.shape[1] <= 12:
        return arr

    out = arr.copy()
    y_axes = []
    x_axes = []
    pelvis_seq = []
    for t in range(out.shape[0]):
        frame = out[t]
        if not (
            np.isfinite(frame[5]).all()
            and np.isfinite(frame[6]).all()
            and np.isfinite(frame[11]).all()
            and np.isfinite(frame[12]).all()
        ):
            continue
        pelvis = (frame[11] + frame[12]) * 0.5
        shoulder = (frame[5] + frame[6]) * 0.5
        y = shoulder - pelvis
        y_norm = float(np.linalg.norm(y))
        if y_norm <= 1e-6:
            continue
        y = y / y_norm

        x = frame[6] - frame[5]
        x = x - np.dot(x, y) * y
        x_norm = float(np.linalg.norm(x))
        if x_norm <= 1e-6:
            continue
        x = x / x_norm

        y_axes.append(y)
        x_axes.append(x)
        pelvis_seq.append(pelvis)

    if not y_axes or not x_axes:
        return out

    y_axis = np.nanmedian(np.stack(y_axes, axis=0), axis=0)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-6 or not np.isfinite(y_norm):
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        y_axis = y_axis / y_norm

    x_axis = np.nanmedian(np.stack(x_axes, axis=0), axis=0)
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6 or not np.isfinite(x_norm):
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm <= 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x_norm = 1.0
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6 or not np.isfinite(z_norm):
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        z_axis = z_axis / z_norm

    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    for t in range(out.shape[0]):
        frame = out[t]
        if np.isfinite(frame[11]).all() and np.isfinite(frame[12]).all():
            pelvis = (frame[11] + frame[12]) * 0.5
        else:
            pelvis = np.nanmedian(frame, axis=0)
        out[t] = (frame - pelvis[None, :]) @ basis

    return out


def _rodrigues_rotate(v, axis, theta):
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return v
    axis = axis / norm
    v = np.asarray(v, dtype=np.float32)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def enforce_two_bone_ik_consistency_3d(pose_3d, chains=None, strength=0.55, iterations=2):
    """2-본 체인(어깨-팔꿈치-손목, 골반-무릎-발목)에 IK를 적용한다."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr

    out = arr.copy()
    chains = list(chains or [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)])
    s = float(np.clip(strength, 0.0, 1.0))
    n_iter = max(1, int(iterations))

    target_lengths = {}
    for a, b, c in chains:
        if max(a, b, c) >= out.shape[1]:
            continue
        l1 = np.linalg.norm(out[:, b, :] - out[:, a, :], axis=1)
        l2 = np.linalg.norm(out[:, c, :] - out[:, b, :], axis=1)
        v1 = l1[np.isfinite(l1) & (l1 > 1e-6)]
        v2 = l2[np.isfinite(l2) & (l2 > 1e-6)]
        if v1.size == 0 or v2.size == 0:
            continue
        target_lengths[(a, b, c)] = (float(np.median(v1)), float(np.median(v2)))

    if not target_lengths:
        return out

    for _ in range(n_iter):
        for t in range(out.shape[0]):
            frame = out[t]
            for (a, b, c), (l1, l2) in target_lengths.items():
                pa = frame[a]
                pb = frame[b]
                pc = frame[c]
                if not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(pc).all()):
                    continue

                dvec = pc - pa
                d = float(np.linalg.norm(dvec))
                if d <= 1e-6:
                    continue
                n = dvec / d
                d_clamped = float(np.clip(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
                target_end = pa + n * d_clamped

                bend = np.cross(pb - pa, pc - pb)
                bend_norm = float(np.linalg.norm(bend))
                if bend_norm <= 1e-6:
                    bend = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    bend = bend / bend_norm

                a_proj = (l1 * l1 - l2 * l2 + d_clamped * d_clamped) / (2.0 * d_clamped)
                h2 = max(l1 * l1 - a_proj * a_proj, 0.0)
                h = float(np.sqrt(h2))
                perp = np.cross(bend, n)
                p_mid = pa + n * a_proj
                cand1 = p_mid + perp * h
                cand2 = p_mid - perp * h
                target_mid = cand1 if np.linalg.norm(cand1 - pb) <= np.linalg.norm(cand2 - pb) else cand2

                frame[b] = (1.0 - s) * pb + s * target_mid
                frame[c] = (1.0 - s) * pc + s * target_end

            out[t] = frame

    return out


def enforce_joint_rom_constraints_3d(pose_3d, strength=0.45):
    """팔꿈치/무릎 관절각을 인체 가동범위로 제한한다."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr

    out = arr.copy()
    s = float(np.clip(strength, 0.0, 1.0))
    # (proximal, joint, distal, min_deg, max_deg)
    hinges = [
        (5, 7, 9, 15.0, 175.0),
        (6, 8, 10, 15.0, 175.0),
        (11, 13, 15, 10.0, 175.0),
        (12, 14, 16, 10.0, 175.0),
    ]

    for t in range(out.shape[0]):
        frame = out[t]
        for a, b, c, min_deg, max_deg in hinges:
            if max(a, b, c) >= frame.shape[0]:
                continue
            pa = frame[a]
            pb = frame[b]
            pc = frame[c]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(pc).all()):
                continue

            u = pa - pb
            v = pc - pb
            lu = float(np.linalg.norm(u))
            lv = float(np.linalg.norm(v))
            if lu <= 1e-6 or lv <= 1e-6:
                continue
            u = u / lu
            v = v / lv
            dot_uv = float(np.clip(np.dot(u, v), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot_uv)))
            clamped = float(np.clip(angle, min_deg, max_deg))
            if abs(clamped - angle) <= 1e-4:
                continue

            axis = np.cross(u, v)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-6:
                continue
            axis = axis / axis_norm

            theta = np.deg2rad(clamped)
            cand1 = _rodrigues_rotate(u, axis, theta)
            cand2 = _rodrigues_rotate(u, axis, -theta)
            v_des = cand1 if np.dot(cand1, v) >= np.dot(cand2, v) else cand2
            target_c = pb + v_des * lv
            frame[c] = (1.0 - s) * pc + s * target_c

        out[t] = frame

    return out


def smooth_pose3d_temporal_velocity_aware(pose_3d, alpha_slow=0.25, alpha_fast=0.85, velocity_scale=0.08):
    """속도 기반 3D 시간 스무딩(빠른 동작 구간은 덜 스무딩)."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.shape[0] <= 1:
        return arr

    out = arr.copy()
    a0 = float(np.clip(alpha_slow, 0.0, 1.0))
    a1 = float(np.clip(alpha_fast, 0.0, 1.0))
    if a1 < a0:
        a0, a1 = a1, a0
    v_scale = max(1e-6, float(velocity_scale))

    for t in range(1, out.shape[0]):
        prev = out[t - 1]
        cur = out[t]
        vel = np.linalg.norm(cur - prev, axis=1)
        for j in range(out.shape[1]):
            if not (np.isfinite(prev[j]).all() and np.isfinite(cur[j]).all()):
                continue
            ratio = float(np.clip(vel[j] / v_scale, 0.0, 1.0))
            alpha = a0 + (a1 - a0) * ratio
            out[t, j, :] = alpha * cur[j] + (1.0 - alpha) * prev[j]

    return out


def smooth_pose3d_temporal_acceleration(pose_3d, beta=0.2, iterations=1):
    """2차 차분(가속도) 기반 시간 스무딩."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.shape[0] <= 2:
        return arr

    out = arr.copy()
    b = float(np.clip(beta, 0.0, 1.0))
    n_iter = max(1, int(iterations))
    for _ in range(n_iter):
        prev = out[:-2]
        cur = out[1:-1]
        nxt = out[2:]
        target = 0.5 * (prev + nxt)
        mask = np.isfinite(cur).all(axis=-1) & np.isfinite(target).all(axis=-1)
        upd = cur.copy()
        upd[mask] = (1.0 - b) * cur[mask] + b * target[mask]
        out[1:-1] = upd
    return out


def enforce_ground_contact_stability_3d(pose_3d, ankle_indices=(15, 16), velocity_threshold=0.03, strength=0.35):
    """발목 저속 구간에서 지면 높이를 안정화한다."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.shape[0] <= 1:
        return arr

    out = arr.copy()
    s = float(np.clip(strength, 0.0, 1.0))
    for j in ankle_indices:
        if j >= out.shape[1]:
            continue
        ankle = out[:, j, :]
        if not np.isfinite(ankle).all():
            continue
        vel = np.linalg.norm(np.diff(ankle, axis=0), axis=1)
        vel = np.concatenate([[vel[0]], vel], axis=0)
        contact = vel <= float(velocity_threshold)
        y = ankle[:, 1]
        if not np.any(contact):
            continue
        ground_y = float(np.median(y[contact]))
        y_new = y.copy()
        y_new[contact] = (1.0 - s) * y[contact] + s * ground_y
        out[:, j, 1] = y_new
    return out


def _weak_perspective_fit(xy3d, xy2d):
    xy3d = np.asarray(xy3d, dtype=np.float32)
    xy2d = np.asarray(xy2d, dtype=np.float32)
    mu3 = np.mean(xy3d, axis=0)
    mu2 = np.mean(xy2d, axis=0)
    x3 = xy3d - mu3
    x2 = xy2d - mu2
    denom = float(np.sum(x3 * x3))
    if denom <= 1e-8:
        return 1.0, mu2 - mu3
    s = float(np.sum(x3 * x2) / denom)
    t = mu2 - s * mu3
    return s, t


def _reprojection_rmse_norm(pose_3d, target_2d, valid_mask=None):
    arr3 = np.asarray(pose_3d, dtype=np.float32)
    arr2 = np.asarray(target_2d, dtype=np.float32)
    if arr3.ndim != 3 or arr2.ndim != 3 or arr3.shape[0] != arr2.shape[0]:
        return np.nan
    mask = np.ones(arr2.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    sq = []
    for t in range(arr3.shape[0]):
        valid = mask[t]
        if not np.any(valid):
            continue
        xy = arr3[t, valid, :2]
        tgt = arr2[t, valid, :]
        if xy.shape[0] < 2:
            continue
        s, tr = _weak_perspective_fit(xy, tgt)
        proj = s * xy + tr
        err = proj - tgt
        sq.append(np.mean(np.sum(err * err, axis=1)))
    if not sq:
        return np.nan
    return float(np.sqrt(np.mean(sq)))


def refine_pose3d_with_reprojection(
    pose_3d,
    target_2d,
    valid_mask=None,
    iterations=2,
    step=0.35,
    depth_smooth=0.12,
):
    """약한 원근 투영 오차를 줄이도록 3D X/Y를 정합하고 Z를 시간축으로 안정화한다."""
    arr3 = np.asarray(pose_3d, dtype=np.float32)
    arr2 = np.asarray(target_2d, dtype=np.float32)
    if arr3.ndim != 3 or arr3.shape[-1] != 3 or arr2.ndim != 3 or arr2.shape[-1] != 2:
        return arr3, {"applied": False, "reason": "invalid_shape"}
    if arr3.shape[0] != arr2.shape[0] or arr3.shape[1] != arr2.shape[1]:
        return arr3, {"applied": False, "reason": "shape_mismatch"}

    out = arr3.copy()
    mask = np.ones(arr2.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    rmse_before = _reprojection_rmse_norm(out, arr2, valid_mask=mask)

    n_iter = max(1, int(iterations))
    lr = float(np.clip(step, 0.0, 1.0))
    for _ in range(n_iter):
        for t in range(out.shape[0]):
            valid = mask[t]
            if not np.any(valid):
                continue
            xy = out[t, valid, :2]
            tgt = arr2[t, valid, :]
            if xy.shape[0] < 2:
                continue
            s, tr = _weak_perspective_fit(xy, tgt)
            proj = s * xy + tr
            resid = tgt - proj
            denom = max(abs(s), 1e-5)
            corr = (lr / denom) * resid
            out[t, valid, :2] = xy + corr

        d = float(np.clip(depth_smooth, 0.0, 1.0))
        if d > 0.0 and out.shape[0] > 2:
            z = out[:, :, 2]
            z[1:-1] = (1.0 - d) * z[1:-1] + d * 0.5 * (z[:-2] + z[2:])
            out[:, :, 2] = z

    rmse_after = _reprojection_rmse_norm(out, arr2, valid_mask=mask)
    return out, {
        "applied": True,
        "iterations": n_iter,
        "step": lr,
        "depth_smooth": float(np.clip(depth_smooth, 0.0, 1.0)),
        "reproj_rmse_before": rmse_before,
        "reproj_rmse_after": rmse_after,
    }


def evaluate_poseformer_human_shape_quality_3d(pose_3d):
    """3D 인체 형태 품질 지표와 합격/불합격 판정을 계산한다."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return {"valid": False, "reason": "invalid_shape"}

    def _ratio_len(a, b):
        d = np.linalg.norm(arr[:, b] - arr[:, a], axis=1)
        v = d[np.isfinite(d) & (d > 1e-8)]
        return v

    sh = _ratio_len(5, 6)
    hp = _ratio_len(11, 12)
    lu = _ratio_len(5, 7)
    lf = _ratio_len(7, 9)
    ru = _ratio_len(6, 8)
    rf = _ratio_len(8, 10)

    hip_y = np.nanmean(arr[:, [11, 12], 1], axis=1)
    ank_y = np.nanmean(arr[:, [15, 16], 1], axis=1)
    ankle_below_ratio = float(np.nanmean((ank_y - hip_y) < 0.0))

    left_chain_ok = (arr[:, 15, 1] <= arr[:, 13, 1]) & (arr[:, 13, 1] <= arr[:, 11, 1]) & (arr[:, 11, 1] <= arr[:, 5, 1])
    right_chain_ok = (arr[:, 16, 1] <= arr[:, 14, 1]) & (arr[:, 14, 1] <= arr[:, 12, 1]) & (arr[:, 12, 1] <= arr[:, 6, 1])
    chain_ok_ratio = float(np.nanmean(left_chain_ok & right_chain_ok))

    def _cv_from_bones(bones):
        cvs = []
        for a, b in bones:
            d = np.linalg.norm(arr[:, b] - arr[:, a], axis=1)
            v = d[np.isfinite(d) & (d > 1e-8)]
            if v.size >= 2:
                m = float(np.mean(v))
                if m > 1e-8:
                    cvs.append(float(np.std(v) / m))
        return float(np.mean(cvs)) if cvs else np.nan

    bone_cv = _cv_from_bones(COCO_BONES_17)

    def _rel_diff(a_vals, b_vals):
        if a_vals.size == 0 or b_vals.size == 0:
            return np.nan
        a_med = float(np.median(a_vals))
        b_med = float(np.median(b_vals))
        denom = max((a_med + b_med) * 0.5, 1e-8)
        return float(abs(a_med - b_med) / denom)

    upper_sym = _rel_diff(lu, ru)
    lower_sym = _rel_diff(lf, rf)

    metrics = {
        "ankle_below_hip_ratio": ankle_below_ratio,
        "vertical_chain_ok_ratio": chain_ok_ratio,
        "bone_length_cv": bone_cv,
        "upper_arm_symmetry_error": upper_sym,
        "forearm_symmetry_error": lower_sym,
        "pass_ankle_below_hip": ankle_below_ratio >= 0.99,
        "pass_vertical_chain": chain_ok_ratio >= 0.95,
        "pass_bone_cv": bool(np.isfinite(bone_cv) and bone_cv <= 0.10),
        "pass_symmetry": bool(
            np.isfinite(upper_sym) and np.isfinite(lower_sym)
            and upper_sym <= 0.15 and lower_sym <= 0.15
        ),
    }
    metrics["pass_all"] = bool(
        metrics["pass_ankle_below_hip"]
        and metrics["pass_vertical_chain"]
        and metrics["pass_bone_cv"]
        and metrics["pass_symmetry"]
    )
    return metrics


def remap_pose3d_joint_layout_to_coco(pose_3d, source_layout="coco"):
    """3D 관절 레이아웃을 COCO 17 순서로 정규화한다."""
    arr = np.asarray(pose_3d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr
    if arr.shape[1] < 17:
        return arr

    layout = str(source_layout).lower().strip()
    if layout == "coco":
        return arr.copy()
    if layout == "h36m":
        return arr[:, H36M_TO_COCO_17, :].astype(np.float32, copy=False)
    raise ValueError(f"unknown source_layout: {source_layout!r} (expected: coco, h36m, auto)")


def _pose3d_layout_plausibility_score(pose_3d_coco):
    """COCO 기준 인체 형태 점수(클수록 더 plausible)."""
    normalized = canonicalize_vertical_axis_3d(pose_3d_coco)
    q = evaluate_poseformer_human_shape_quality_3d(normalized)
    if not q.get("valid", True):
        return -1e9, q

    score = 0.0
    score += 3.0 * float(q.get("ankle_below_hip_ratio", 0.0))
    score += 3.0 * float(q.get("vertical_chain_ok_ratio", 0.0))

    bone_cv = float(q.get("bone_length_cv", np.inf))
    if np.isfinite(bone_cv):
        score += max(0.0, 1.0 - min(1.0, bone_cv / 0.6))

    upper_sym = float(q.get("upper_arm_symmetry_error", np.inf))
    if np.isfinite(upper_sym):
        score += max(0.0, 1.0 - min(1.0, upper_sym / 1.0))
    lower_sym = float(q.get("forearm_symmetry_error", np.inf))
    if np.isfinite(lower_sym):
        score += max(0.0, 1.0 - min(1.0, lower_sym / 1.0))

    score += 0.5 * float(bool(q.get("pass_ankle_below_hip", False)))
    score += 0.5 * float(bool(q.get("pass_vertical_chain", False)))
    score += 0.5 * float(bool(q.get("pass_bone_cv", False)))
    score += 0.5 * float(bool(q.get("pass_symmetry", False)))
    return float(score), q


def resolve_pose3d_layout_to_coco(pose_3d, source_layout="auto"):
    """PoseFormer 출력 레이아웃을 판별/변환해 COCO 순서 3D를 반환한다."""
    layout = str(source_layout).lower().strip()
    arr = np.asarray(pose_3d, dtype=np.float32)

    if layout in {"coco", "h36m"}:
        remapped = remap_pose3d_joint_layout_to_coco(arr, source_layout=layout)
        score, quality = _pose3d_layout_plausibility_score(remapped)
        return remapped, {
            "requested_layout": layout,
            "selected_layout": layout,
            "scores": {layout: score},
            "qualities": {layout: quality},
        }

    if layout != "auto":
        raise ValueError("source_layout must be one of: auto, coco, h36m")

    candidates = {
        "coco": remap_pose3d_joint_layout_to_coco(arr, source_layout="coco"),
        "h36m": remap_pose3d_joint_layout_to_coco(arr, source_layout="h36m"),
    }
    scores = {}
    qualities = {}
    for name, cand in candidates.items():
        score, quality = _pose3d_layout_plausibility_score(cand)
        scores[name] = score
        qualities[name] = quality

    selected = max(scores.keys(), key=lambda k: scores[k])
    # 동점/근접 시 불필요한 변환을 피하기 위해 coco 우선.
    if abs(scores["coco"] - scores["h36m"]) <= 1e-6:
        selected = "coco"

    return candidates[selected], {
        "requested_layout": "auto",
        "selected_layout": selected,
        "scores": scores,
        "qualities": qualities,
    }


def write_poseformer_csv(path, pose_3d_seq, names, fps):
    """프레임별 3D 관절 좌표를 CSV로 저장한다."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        header = ["frame", "time_s"]
        for name in names:
            header += [f"{name}_x", f"{name}_y", f"{name}_z"]
        wr.writerow(header)

        for idx, joints in enumerate(np.asarray(pose_3d_seq, dtype=np.float32)):
            row = [idx, round(idx / fps, 4)]
            for joint in joints:
                row += [round(float(joint[0]), 5), round(float(joint[1]), 5), round(float(joint[2]), 5)]
            wr.writerow(row)


def run_poseformer_3d_inference(
    kpts_seq,
    frame_width,
    frame_height,
    *,
    repo_dir,
    checkpoint_path,
    fps,
    output_path=None,
    csv_path=None,
    runner_cls=None,
    runner_kwargs=None,
    joint_names=None,
    min_input_confidence=0.35,
    readiness_thresholds=None,
    enforce_readiness=True,
    physics_constraints=True,
    physics_strength=0.35,
    physics_iterations=2,
    human_shape_constraints=True,
    human_shape_strength=0.6,
    human_shape_iterations=2,
    ik_constraints=True,
    ik_strength=0.55,
    ik_iterations=2,
    rom_constraints=True,
    rom_strength=0.45,
    temporal_smoothing=True,
    temporal_alpha_slow=0.25,
    temporal_alpha_fast=0.85,
    temporal_velocity_scale=0.08,
    acceleration_smoothing=True,
    acceleration_beta=0.18,
    acceleration_iterations=1,
    ground_contact_constraints=True,
    ground_contact_strength=0.38,
    ground_contact_velocity_threshold=0.03,
    reprojection_refinement=True,
    reprojection_iterations=2,
    reprojection_step=0.4,
    reprojection_depth_smooth=0.1,
    poseformer_source_layout="auto",
    pose3d_model_name="poseformer",
):
    """2D 키포인트 시퀀스에서 PoseFormerV2 기반 3D 시퀀스를 생성한다."""
    if not kpts_seq:
        raise ValueError("kpts_seq must not be empty")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    from poseformer_adapter import (
        COCO_KEYPOINT_NAMES,
        PoseFormerV2Runner,
        prepare_poseformer_inputs,
        validate_poseformer_joint_contract,
    )
    from mixste_adapter import MixSTERunner

    names = list(joint_names or COCO_KEYPOINT_NAMES)
    validate_poseformer_joint_contract(names)

    readiness_kwargs = dict(readiness_thresholds or {})
    readiness = assess_poseformer_readiness(kpts_seq, **readiness_kwargs)
    input_mapping = evaluate_input_joint_mapping_consistency_2d(kpts_seq)
    if enforce_readiness and not readiness["ready"]:
        issues = ", ".join(readiness["issues"]) or "unknown"
        raise ValueError(f"PoseFormerV2 input sequence is not ready: {issues}")

    prepared = prepare_poseformer_inputs(
        kpts_seq,
        width=frame_width,
        height=frame_height,
        min_confidence=min_input_confidence,
        joint_names=names,
    )
    model_name = str(pose3d_model_name).lower().strip()
    if runner_cls is not None:
        runner_type = runner_cls
    elif model_name in {"mixste", "mixste2"}:
        runner_type = MixSTERunner
    else:
        runner_type = PoseFormerV2Runner
    runner = runner_type(repo_dir, checkpoint_path, **(runner_kwargs or {}))
    pose_3d = runner.predict_sequence(prepared["normalized_sequence"])
    pose_3d, layout_info = resolve_pose3d_layout_to_coco(
        pose_3d,
        source_layout=poseformer_source_layout,
    )
    reprojection_info = {"applied": False}
    if reprojection_refinement:
        pose_3d, reprojection_info = refine_pose3d_with_reprojection(
            pose_3d,
            prepared["normalized_sequence"],
            valid_mask=prepared.get("valid_mask"),
            iterations=reprojection_iterations,
            step=reprojection_step,
            depth_smooth=reprojection_depth_smooth,
        )
    pose_3d = canonicalize_vertical_axis_3d(pose_3d)
    pose_3d = canonicalize_body_frame_3d(pose_3d)
    if physics_constraints:
        pose_3d = enforce_bone_length_consistency_3d(
            pose_3d,
            bones=COCO_BONES_17,
            strength=physics_strength,
            iterations=physics_iterations,
        )
    if human_shape_constraints:
        pose_3d = enforce_human_vertical_hierarchy_3d(
            pose_3d,
            strength=human_shape_strength,
            iterations=human_shape_iterations,
        )
    if ik_constraints:
        pose_3d = enforce_two_bone_ik_consistency_3d(
            pose_3d,
            strength=ik_strength,
            iterations=ik_iterations,
        )
    if rom_constraints:
        pose_3d = enforce_joint_rom_constraints_3d(
            pose_3d,
            strength=rom_strength,
        )
    if temporal_smoothing:
        pose_3d = smooth_pose3d_temporal_velocity_aware(
            pose_3d,
            alpha_slow=temporal_alpha_slow,
            alpha_fast=temporal_alpha_fast,
            velocity_scale=temporal_velocity_scale,
        )
    if acceleration_smoothing:
        pose_3d = smooth_pose3d_temporal_acceleration(
            pose_3d,
            beta=acceleration_beta,
            iterations=acceleration_iterations,
        )
    if ground_contact_constraints:
        pose_3d = enforce_ground_contact_stability_3d(
            pose_3d,
            velocity_threshold=ground_contact_velocity_threshold,
            strength=ground_contact_strength,
        )
    # 마지막 단계에서 상하축/수직 위계를 다시 고정해 렌더와 지표의 일관성을 높인다.
    pose_3d = canonicalize_vertical_axis_3d(pose_3d)
    if human_shape_constraints:
        pose_3d = enforce_human_vertical_hierarchy_3d(
            pose_3d,
            strength=max(float(human_shape_strength), 0.85),
            iterations=1,
        )
    sanity = validate_poseformer_output(pose_3d)
    quality = evaluate_poseformer_human_shape_quality_3d(pose_3d)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.save(output_path, pose_3d)

    if csv_path:
        write_poseformer_csv(csv_path, pose_3d, names, fps)

    return {
        "pose_3d": pose_3d,
        "prepared": prepared,
        "readiness": readiness,
        "input_mapping": input_mapping,
        "sanity": sanity,
        "physics": {
            "enabled": bool(physics_constraints),
            "strength": float(physics_strength),
            "iterations": int(physics_iterations),
        },
        "human_shape": {
            "enabled": bool(human_shape_constraints),
            "strength": float(human_shape_strength),
            "iterations": int(human_shape_iterations),
        },
        "ik": {
            "enabled": bool(ik_constraints),
            "strength": float(ik_strength),
            "iterations": int(ik_iterations),
        },
        "rom": {
            "enabled": bool(rom_constraints),
            "strength": float(rom_strength),
        },
        "temporal": {
            "enabled": bool(temporal_smoothing),
            "alpha_slow": float(temporal_alpha_slow),
            "alpha_fast": float(temporal_alpha_fast),
            "velocity_scale": float(temporal_velocity_scale),
        },
        "acceleration": {
            "enabled": bool(acceleration_smoothing),
            "beta": float(acceleration_beta),
            "iterations": int(acceleration_iterations),
        },
        "ground_contact": {
            "enabled": bool(ground_contact_constraints),
            "strength": float(ground_contact_strength),
            "velocity_threshold": float(ground_contact_velocity_threshold),
        },
        "reprojection": reprojection_info,
        "quality": quality,
        "layout": layout_info,
        "output_path": output_path,
        "csv_path": csv_path,
        "pose3d_model_name": model_name,
    }


def compare_preprocessing_modes(inp, engine_name="yolo", sample_limit=None, **engine_kw):
    """영상에서 CLAHE 전후의 검출률을 비교해 보정 효과를 정량화한다.

    Returns
    -------
    dict
        {
            "raw": {"frames": int, "detected": int, "detection_rate": float},
            "clahe": {"frames": int, "detected": int, "detection_rate": float},
            "delta": float,
        }
    """
    validate_analysis_inputs(inp, engine_name, **engine_kw)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {inp}")

    frames = []
    limit = sample_limit or 30
    while len(frames) < limit:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"샘플 프레임을 읽지 못했습니다: {inp}")

    results = {}
    for mode_name, should_enhance in (("raw", False), ("clahe", True)):
        engine = build_engine(engine_name, **engine_kw)
        try:
            detected = 0
            for frame in frames:
                proc = _enhance(frame) if should_enhance else frame
                kpts = engine.estimate(proc)
                if kpts is not None:
                    detected += 1
            rate = detected / len(frames) if frames else 0.0
            results[mode_name] = {
                "frames": len(frames),
                "detected": detected,
                "detection_rate": float(rate),
            }
        finally:
            engine.close()

    raw_rate = results["raw"]["detection_rate"]
    clahe_rate = results["clahe"]["detection_rate"]
    results["delta"] = float(clahe_rate - raw_rate)
    return results


def analyze_video(inp, outp, engine_name="yolo", csv_path=None,
                  smooth=True, preprocess=False, progress=True,
                  kpts_seq_override=None,
                  keypoint_names=None,
                  edges=None,
                  single_camera_2d_stabilization=True,
                  single_camera_2d_min_confidence=0.25,
                  single_camera_2d_spike_px=35.0,
                  single_camera_2d_alpha_low_conf=0.25,
                  single_camera_2d_alpha_high_conf=0.8,
                  skeleton_only_outp=None, poseformer_repo_dir=None,
                  poseformer_checkpoint_path=None, mixste_repo_dir=None,
                  mixste_checkpoint_path=None, poseformer_output_path=None,
                  poseformer_csv_path=None, poseformer_runner_kwargs=None,
                  pose3d_model_name="poseformer",
                  pose3d_model_names=None,
                  poseformer_physics_constraints=True,
                  poseformer_physics_strength=0.35,
                  poseformer_physics_iterations=2,
                  poseformer_human_shape_constraints=True,
                  poseformer_human_shape_strength=0.6,
                  poseformer_human_shape_iterations=2,
                  poseformer_ik_constraints=True,
                  poseformer_ik_strength=0.55,
                  poseformer_ik_iterations=2,
                  poseformer_rom_constraints=True,
                  poseformer_rom_strength=0.45,
                  poseformer_temporal_smoothing=True,
                  poseformer_temporal_alpha_slow=0.25,
                  poseformer_temporal_alpha_fast=0.85,
                  poseformer_temporal_velocity_scale=0.08,
                  poseformer_enforce_readiness=True,
                  poseformer_source_layout="auto",
                  **engine_kw):
    engine_kwargs = dict(engine_kw)
    engine_kwargs.pop("skeleton_only_outp", None)
    if kpts_seq_override is None:
        validate_analysis_inputs(inp, engine_name, **engine_kwargs)

    if poseformer_checkpoint_path and not poseformer_repo_dir:
        poseformer_repo_dir = default_poseformer_repo_dir()

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    engine_keypoint_names = list(keypoint_names or COCO_KEYPOINTS)
    engine_edges = list(edges or COCO_EDGES)
    if kpts_seq_override is None:
        engine = None if kpts_seq_override is not None else build_engine(engine_name, **engine_kwargs)
    else:
        engine = SimpleNamespace(keypoint_names=engine_keypoint_names, edges=engine_edges, close=lambda: None)

    # 1차 패스: 프레임 수집.
    frames = []
    t0 = time.time()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1
        if progress and idx % 20 == 0:
            print(f"  프레임 적재 {idx}/{n_total}...", flush=True)
    cap.release()

    if not frames:
        engine.close()
        raise RuntimeError("프레임을 읽지 못했습니다. 영상 코덱 또는 파일을 확인하세요.")

    # 일부 휴대폰 영상은 메타데이터 회전으로 CAP_PROP 크기와 실제 디코딩 크기가 다르다.
    # 렌더/인코딩은 실제 프레임 크기를 기준으로 처리해 화면 잘림을 방지한다.
    h, w = frames[0].shape[:2]
    if (w, h) != (src_w, src_h):
        print(f"  해상도 보정: 메타데이터 {src_w}x{src_h} -> 디코딩 프레임 {w}x{h}")

    # 1.5차 패스: 자세추정 또는 외부 키포인트 입력.
    if kpts_seq_override is None:
        processed_frames = [_enhance(frame) if preprocess else frame for frame in frames]
        if hasattr(engine, "estimate_batch"):
            raw_kpts = engine.estimate_batch(processed_frames)
        else:
            raw_kpts = []
            for idx, proc in enumerate(processed_frames, start=1):
                raw_kpts.append(engine.estimate(proc))
                if progress and idx % 20 == 0:
                    print(f"  추정 {idx}/{len(processed_frames)} 프레임...", flush=True)
    else:
        raw_kpts = list(kpts_seq_override)
        if len(raw_kpts) != len(frames):
            raise ValueError(
                f"keypoints CSV frame count mismatch: video={len(frames)} csv={len(raw_kpts)}"
            )
    if engine is not None:
        engine.close()

    detected = sum(k is not None for k in raw_kpts)
    if kpts_seq_override is None:
        print(f"  검출 성공: {detected}/{len(frames)} 프레임 "
              f"({100*detected/max(1,len(frames)):.0f}%)")
    else:
        print(f"  입력 키포인트 사용: {detected}/{len(frames)} 프레임 "
              f"({100*detected/max(1,len(frames)):.0f}%)")

    # 스무딩.
    kpts_seq = smooth_sequence(raw_kpts, fps, keypoint_names=engine.keypoint_names) if smooth else raw_kpts
    if single_camera_2d_stabilization:
        kpts_seq = stabilize_keypoints_single_camera(
            kpts_seq,
            min_confidence=single_camera_2d_min_confidence,
            spike_px=single_camera_2d_spike_px,
            alpha_low_conf=single_camera_2d_alpha_low_conf,
            alpha_high_conf=single_camera_2d_alpha_high_conf,
        )

    backend_names = list(pose3d_model_names or [pose3d_model_name])
    pose3d_results = {}
    for backend_name in backend_names:
        model_key = str(backend_name).lower().strip()
        if model_key in {"mixste", "mixste2"}:
            backend_repo_dir = mixste_repo_dir or poseformer_repo_dir
            backend_checkpoint_path = mixste_checkpoint_path or poseformer_checkpoint_path
        else:
            backend_repo_dir = poseformer_repo_dir
            backend_checkpoint_path = poseformer_checkpoint_path

        if not backend_checkpoint_path:
            raise ValueError(f"{model_key} backend requires a checkpoint path or discoverable checkpoint")
        base_output = poseformer_output_path
        base_csv = poseformer_csv_path
        if len(backend_names) > 1:
            if base_output:
                root, ext = os.path.splitext(base_output)
                base_output = f"{root}_{model_key}{ext or '.npy'}"
            if base_csv:
                root, ext = os.path.splitext(base_csv)
                base_csv = f"{root}_{model_key}{ext or '.csv'}"
        pose3d_results[model_key] = run_poseformer_3d_inference(
            kpts_seq,
            frame_width=w,
            frame_height=h,
            repo_dir=backend_repo_dir,
            checkpoint_path=backend_checkpoint_path,
            fps=fps,
            output_path=base_output,
            csv_path=base_csv,
            runner_kwargs=poseformer_runner_kwargs,
            joint_names=engine.keypoint_names,
            physics_constraints=poseformer_physics_constraints,
            physics_strength=poseformer_physics_strength,
            physics_iterations=poseformer_physics_iterations,
            human_shape_constraints=poseformer_human_shape_constraints,
            human_shape_strength=poseformer_human_shape_strength,
            human_shape_iterations=poseformer_human_shape_iterations,
            ik_constraints=poseformer_ik_constraints,
            ik_strength=poseformer_ik_strength,
            ik_iterations=poseformer_ik_iterations,
            rom_constraints=poseformer_rom_constraints,
            rom_strength=poseformer_rom_strength,
            temporal_smoothing=poseformer_temporal_smoothing,
            temporal_alpha_slow=poseformer_temporal_alpha_slow,
            temporal_alpha_fast=poseformer_temporal_alpha_fast,
            temporal_velocity_scale=poseformer_temporal_velocity_scale,
            enforce_readiness=poseformer_enforce_readiness,
            poseformer_source_layout=poseformer_source_layout,
            pose3d_model_name=model_key,
        )
    if backend_names and len(pose3d_results) != len(backend_names):
        missing = [name for name in backend_names if str(name).lower().strip() not in pose3d_results]
        raise RuntimeError(f"Missing 3D backend outputs: {missing}")
    poseformer_result = pose3d_results.get(str(backend_names[0]).lower().strip()) if backend_names else None

    # 2차 패스: 렌더 + 저장.
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    if skeleton_only_outp is None:
        skeleton_only_outp = os.path.splitext(outp)[0] + "_skeleton_only.mp4"

    overlay_frames = []
    overlay_rendered = []
    skeleton_rendered = []
    for frame, kpts in zip(frames, kpts_seq):
        overlay_frame = draw_skeleton(frame.copy(), kpts, engine.edges)
        skeleton_only_frame = draw_skeleton(frame.copy(), kpts, engine.edges, skeleton_only=True)
        overlay_rendered.append(overlay_frame)
        skeleton_rendered.append(skeleton_only_frame)
        overlay_frames.append(overlay_frame)

    overlay_out = _write_video_with_fallback(outp, overlay_rendered, fps, w, h)
    skeleton_out = _write_video_with_fallback(skeleton_only_outp, skeleton_rendered, fps, w, h)
    if overlay_out is None or skeleton_out is None:
        raise RuntimeError("브라우저 재생용 영상 저장에 실패했습니다.")

    viewer_html = os.path.splitext(outp)[0] + "_viewer.html"
    skeleton_viewer_html = os.path.splitext(outp)[0] + "_skeleton_viewer.html"
    build_dynamic_viewer(overlay_frames, viewer_html, fps=fps, title="투구 포즈 시각화 (오버레이)")
    build_dynamic_viewer(
        skeleton_rendered,
        skeleton_viewer_html,
        fps=fps,
        title="투구 포즈 시각화 (스켈레톤 전용)",
    )

    if csv_path:
        _write_csv(csv_path, kpts_seq, engine.keypoint_names, fps)

    dt = time.time() - t0
    print(f"  완료: {outp}  ({dt:.1f}s, {len(frames)/max(dt,1e-6):.1f} fps 처리)")
    return {
        "frames": len(frames),
        "detected": detected,
        "fps": fps,
        "viewer_html": viewer_html,
        "skeleton_viewer_html": skeleton_viewer_html,
        "skeleton_only_video": skeleton_only_outp,
        "pose_3d_physics": None if poseformer_result is None else poseformer_result.get("physics"),
        "pose_3d_human_shape": None if poseformer_result is None else poseformer_result.get("human_shape"),
        "pose_3d_ik": None if poseformer_result is None else poseformer_result.get("ik"),
        "pose_3d_rom": None if poseformer_result is None else poseformer_result.get("rom"),
        "pose_3d_temporal": None if poseformer_result is None else poseformer_result.get("temporal"),
        "pose_3d_quality": None if poseformer_result is None else poseformer_result.get("quality"),
        "pose_3d_layout": None if poseformer_result is None else poseformer_result.get("layout"),
        "pose_3d_path": None if poseformer_result is None else poseformer_result["output_path"],
        "pose_3d_csv_path": None if poseformer_result is None else poseformer_result["csv_path"],
        "pose_3d_results": pose3d_results or None,
    }


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
    ap.add_argument("--engine", default="yolo", choices=["yolo", "rtmpose"])
    ap.add_argument("--csv", help="관절좌표 CSV 저장 경로")
    ap.add_argument("--keypoints-csv", help="이미 추출된 COCO 17 keypoints CSV 입력")
    ap.add_argument("--no-smooth", action="store_true", help="스무딩 비활성")
    ap.add_argument("--preprocess", action="store_true", help="역광/저조도 보정(CLAHE)")
    ap.add_argument("--rtmpose-repo", help="RTMPose/MMPose 저장소 경로")
    ap.add_argument("--rtmpose-config", help="RTMPose config 경로 또는 model alias")
    ap.add_argument("--rtmpose-checkpoint", help="RTMPose checkpoint 경로")
    ap.add_argument("--poseformer", action="store_true", help="PoseFormerV2 3D 추론 활성화")
    ap.add_argument("--poseformer-repo", help="PoseFormerV2 저장소 경로")
    ap.add_argument("--poseformer-checkpoint", help="PoseFormerV2 체크포인트 경로")
    ap.add_argument("--poseformer-output", help="PoseFormerV2 3D npy 저장 경로")
    ap.add_argument("--poseformer-csv", help="PoseFormerV2 3D csv 저장 경로")
    ap.add_argument("--no-physics", action="store_true", help="PoseFormer 3D 물리 제약 후처리 비활성")
    ap.add_argument("--physics-strength", type=float, default=0.35, help="뼈 길이 일관성 보정 강도 (0~1)")
    ap.add_argument("--physics-iterations", type=int, default=2, help="뼈 길이 보정 반복 횟수")
    ap.add_argument("--no-human-shape", action="store_true", help="PoseFormer 3D 인체 수직 위계 보정 비활성")
    ap.add_argument("--human-shape-strength", type=float, default=0.6, help="인체 수직 위계 보정 강도 (0~1)")
    ap.add_argument("--human-shape-iterations", type=int, default=2, help="인체 수직 위계 보정 반복 횟수")
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + f"_{args.engine}.mp4"
    rtmpose_repo = args.rtmpose_repo or default_rtmpose_repo_dir()
    rtmpose_config = args.rtmpose_config or default_rtmpose_config_path(rtmpose_repo)
    poseformer_repo = args.poseformer_repo or default_poseformer_repo_dir()
    poseformer_checkpoint = args.poseformer_checkpoint or default_poseformer_checkpoint_path(poseformer_repo)
    print(f"[분석] {args.input}  (engine={args.engine})")
    engine_kwargs = {}
    if args.engine == "rtmpose":
        engine_kwargs.update(
            repo_dir=rtmpose_repo,
            config=rtmpose_config,
            checkpoint=args.rtmpose_checkpoint,
        )
    kpts_seq_override = None
    if args.keypoints_csv:
        kpts_seq_override = _load_keypoints_csv(args.keypoints_csv, expected_names=COCO_KEYPOINTS)
    poseformer_enforce_readiness = not bool(args.keypoints_csv)
    analyze_video(args.input, out, engine_name=args.engine, csv_path=args.csv,
                  smooth=not args.no_smooth, preprocess=args.preprocess,
                  kpts_seq_override=kpts_seq_override,
                  keypoint_names=COCO_KEYPOINTS,
                  edges=COCO_EDGES,
                  poseformer_enforce_readiness=poseformer_enforce_readiness,
                  **engine_kwargs,
                  poseformer_repo_dir=poseformer_repo if args.poseformer else None,
                  poseformer_checkpoint_path=poseformer_checkpoint if args.poseformer else None,
                  poseformer_output_path=args.poseformer_output,
                  poseformer_csv_path=args.poseformer_csv,
                  poseformer_physics_constraints=not args.no_physics,
                  poseformer_physics_strength=args.physics_strength,
                  poseformer_physics_iterations=args.physics_iterations,
                  poseformer_human_shape_constraints=not args.no_human_shape,
                  poseformer_human_shape_strength=args.human_shape_strength,
                  poseformer_human_shape_iterations=args.human_shape_iterations)


if __name__ == "__main__":
    main()

"""투구 동작 분석 데모 UI (Streamlit).

영상을 업로드하면 스켈레톤("졸라맨") 오버레이 영상과 운동학 리포트를
브라우저에서 바로 확인한다. KSEF 시연용 (Phase 1 + Phase 3 통합 데모).

실행:
    .venv/bin/streamlit run src/app.py

기존 모듈(analyze.py, kinematics.py)을 재사용하며, 자세추정·운동학 로직을
여기서 재구현하지 않는다.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "overlay"
if "analysis_state" not in st.session_state:
    st.session_state.analysis_state = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None


def _clear_analysis_state():
    st.session_state.analysis_state = None
    st.session_state.view_mode = "overlay"



# src 디렉토리를 임포트 경로에 추가(모듈 재사용).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import analyze as analyze_mod  # noqa: E402
import kinematics as kin_mod  # noqa: E402
import reference_data as ref_data  # noqa: E402

ref_data = importlib.reload(ref_data)


ANALYSIS_SCHEMA_VERSION = 18

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_BONES_17 = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

POSE3D_VIEW_SPECS = {
    "oblique": {"label": "사선", "yaw_deg": 35.0, "pitch_deg": 18.0},
    "side": {"label": "측면", "yaw_deg": 90.0, "pitch_deg": 12.0},
    "top": {"label": "상단", "yaw_deg": 0.0, "pitch_deg": 88.0},
}


# ------------------------------- 유틸 -------------------------------

def _to_h264(src_mp4, dst_mp4):
    """오버레이 mp4(mp4v)를 브라우저 재생용 H.264로 재인코딩.

    ffmpeg가 없으면 원본을 그대로 복사(재생이 안 될 수 있음).
    """
    if not os.path.exists(src_mp4) or os.path.getsize(src_mp4) <= 0:
        raise FileNotFoundError(f"3D render source video is missing or empty: {src_mp4}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy(src_mp4, dst_mp4)
        return dst_mp4, False
    try:
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", src_mp4,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             dst_mp4],
            check=True,
        )
        if not os.path.exists(dst_mp4) or os.path.getsize(dst_mp4) <= 0:
            raise RuntimeError(f"ffmpeg produced empty output: {dst_mp4}")
        return dst_mp4, True
    except (subprocess.CalledProcessError, RuntimeError):
        shutil.copy(src_mp4, dst_mp4)
        return dst_mp4, False


def _detection_rate(csv_path):
    """keypoints CSV에서 검출 성공 프레임 비율(0~1) 추정."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    vis_cols = [c for c in df.columns if c.endswith("_v")]
    if not vis_cols or len(df) == 0:
        return 0.0
    detected = df[vis_cols].notna().any(axis=1) & (df[vis_cols].fillna(0) > 0).any(axis=1)
    return float(detected.mean())


def _normalize_display_text(text):
    """화면 표시에 방해되는 특수기호를 일반 표기로 치환한다."""
    table = {
        "⚠️": "주의",
        "⚠": "주의",
        "✓": "정상",
        "✔": "정상",
        "✗": "오류",
        "❌": "오류",
        "→": "->",
        "…": "...",
        "①": "1)",
        "②": "2)",
        "③": "3)",
        "§": "",
    }
    out = text
    for old, new in table.items():
        out = out.replace(old, new)
    return out


def _model_ready(engine, *, rtmpose_config_path=None, rtmpose_checkpoint_path=None):
    """2D 엔진 실행 전 사전 조건 확인. (문제 메시지, ok) 반환."""
    weights = os.path.join(ROOT_DIR, "yolo11x-pose.pt")
    if engine == "yolo":
        if not os.path.exists(weights):
            return ("YOLO pose weights가 없습니다. 프로젝트 루트에 `yolo11x-pose.pt`를 두세요."), False
        return "", True
    if engine == "rtmpose":
        if not rtmpose_config_path:
            return "RTMPose config 경로를 입력하세요.", False
        if os.path.exists(rtmpose_config_path):
            if rtmpose_checkpoint_path and not os.path.exists(rtmpose_checkpoint_path):
                return f"RTMPose checkpoint를 찾을 수 없습니다: {rtmpose_checkpoint_path}", False
        else:
            # Model alias / metafile name path: checkpoint can be omitted.
            if rtmpose_checkpoint_path and not os.path.exists(rtmpose_checkpoint_path):
                return f"RTMPose checkpoint를 찾을 수 없습니다: {rtmpose_checkpoint_path}", False
        return "", True
    return f"지원하지 않는 엔진입니다: {engine}", False


def _poseformer_ready(enabled, repo_dir, checkpoint_path):
    """선택적 PoseFormerV2 추론에 필요한 파일 경로를 확인한다."""
    if not enabled:
        return "", True
    if not repo_dir:
        return "PoseFormerV2 저장소 경로를 입력하세요.", False
    if not os.path.isdir(repo_dir):
        return f"PoseFormerV2 저장소를 찾을 수 없습니다: {repo_dir}", False
    if not checkpoint_path:
        return "PoseFormerV2 체크포인트 경로를 입력하세요.", False
    if not os.path.exists(checkpoint_path):
        return f"PoseFormerV2 체크포인트를 찾을 수 없습니다: {checkpoint_path}", False
    return "", True


def _default_poseformer_repo_dir():
    candidates = [
        os.path.join(ROOT_DIR, "PoseFormerV2-main"),
        os.path.join(ROOT_DIR, "PoseFormerV2-main", "PoseFormerV2-main"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "run_poseformer.py")):
            return candidate
    return ""


def _default_poseformer_checkpoint_path(repo_dir=None):
    roots = []
    if repo_dir:
        roots.append(repo_dir)
    roots.extend([
        os.path.join(ROOT_DIR, "PoseFormerV2-main"),
        os.path.join(ROOT_DIR, "PoseFormerV2-main", "PoseFormerV2-main"),
    ])
    candidates = []
    for root in roots:
        if not root:
            continue
        candidates.extend([
            os.path.join(root, "checkpoint", "poseformerv2_27f_3k_3c.bin"),
            os.path.join(root, "checkpoint", "27_3_3.bin"),
            os.path.join(root, "checkpoint", "best_epoch.bin"),
        ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def _default_mixste_repo_dir():
    candidates = [
        os.path.join(os.path.expanduser("~"), "OneDrive", "바탕 화면", "MixSTE-main", "MixSTE-main"),
        os.path.join(ROOT_DIR, "MixSTE-main"),
        os.path.join(ROOT_DIR, "MixSTE-main", "MixSTE-main"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "run.py")):
            return candidate
    return ""


def _default_mixste_checkpoint_path(repo_dir=None):
    roots = []
    if repo_dir:
        roots.append(repo_dir)
    roots.extend([
        os.path.join(os.path.expanduser("~"), "OneDrive", "바탕 화면", "MixSTE-main", "MixSTE-main"),
        os.path.join(ROOT_DIR, "MixSTE-main"),
        os.path.join(ROOT_DIR, "MixSTE-main", "MixSTE-main"),
    ])
    candidates = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                lower = filename.lower()
                if lower.endswith((".pth", ".pt", ".bin")):
                    candidates.append(os.path.join(dirpath, filename))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def _default_rtmpose_repo_dir():
    candidates = [
        os.path.join(os.path.expanduser("~"), "OneDrive", "바탕 화면", "mmpose-main", "mmpose-main"),
        os.path.join(ROOT_DIR, "mmpose-main", "mmpose-main"),
        os.path.join(ROOT_DIR, "mmpose-main"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "mmpose", "apis", "inference.py")):
            return candidate
    return ""


def _default_rtmpose_config_path(repo_dir=None):
    roots = []
    if repo_dir:
        roots.append(repo_dir)
    roots.extend([
        os.path.join(os.path.expanduser("~"), "OneDrive", "바탕 화면", "mmpose-main", "mmpose-main"),
        os.path.join(ROOT_DIR, "mmpose-main", "mmpose-main"),
        os.path.join(ROOT_DIR, "mmpose-main"),
    ])
    candidates = [
        "rtmpose-m_8xb256-420e_coco-256x192",
        "rtmpose-s_8xb256-420e_coco-256x192",
        "rtmpose-t_8xb256-420e_coco-256x192",
    ]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename in {"rtmpose-m_8xb256-420e_coco-256x192.py", "rtmpose-s_8xb256-420e_coco-256x192.py", "rtmpose-t_8xb256-420e_coco-256x192.py"}:
                    return os.path.join(dirpath, filename)
    return candidates[0]


def _video_mime_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".webm":
        return "video/webm"
    if ext == ".avi":
        return "video/avi"
    return "video/mp4"


def _runtime_render_capability():
    """현재 런타임의 3D 영상 렌더/인코딩 가용성을 점검한다."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_ok = bool(ffmpeg_path)

    tmpdir = tempfile.mkdtemp(prefix="render_probe_")
    probe_path = os.path.join(tmpdir, "probe.mp4")
    writer = cv2.VideoWriter(
        probe_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (320, 240),
    )
    writer_ok = bool(writer.isOpened())
    if writer_ok:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (8, 12, 20)
        writer.write(frame)
    writer.release()

    produced_ok = os.path.exists(probe_path) and os.path.getsize(probe_path) > 0
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except OSError:
        pass

    return {
        "ffmpeg_ok": ffmpeg_ok,
        "ffmpeg_path": ffmpeg_path,
        "opencv_writer_ok": writer_ok and produced_ok,
    }


def _resolve_reference_csv_path(profile=None):
    """현재 로드된 reference_data 모듈에서 기준 CSV를 가져온다."""
    try:
        return ref_data.resolve_reference_csv_path(profile)
    except TypeError as exc:
        if "positional arguments" in str(exc) and "takes 0" in str(exc):
            return ref_data.resolve_reference_csv_path()
        raise


def _render_pose3d_interactive(pose_3d_path, key_prefix="pose3d"):
    """마우스 회전이 가능한 3D 스켈레톤 프레임 뷰어를 렌더링한다."""
    if not pose_3d_path:
        st.info("현재 결과에는 3D 출력 경로가 없습니다. 3D 옵션을 켜고 다시 분석하면 3D 뷰가 표시됩니다.")
        return

    if not os.path.exists(pose_3d_path):
        st.warning("3D 포즈 파일이 생성되지 않았습니다. 체크포인트/입력 안정성 게이트 통과 여부를 확인한 뒤 다시 분석해 주세요.")
        return

    try:
        import plotly.graph_objects as go
    except Exception:
        st.warning("plotly가 없어 3D 인터랙티브 뷰를 표시할 수 없습니다.")
        return

    try:
        arr = np.load(pose_3d_path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"3D 포즈 파일을 읽지 못했습니다: {exc}")
        return

    if arr.ndim != 3 or arr.shape[-1] != 3:
        st.warning(f"3D 포즈 shape가 유효하지 않습니다: {arr.shape}")
        return

    n_frames, n_joints, _ = arr.shape
    if n_frames <= 0 or n_joints <= 0:
        st.info("3D 포즈 데이터가 비어 있습니다.")
        return

    st.markdown("#### 3D 포즈 뷰어 (마우스로 회전/확대)")
    show_labels = st.checkbox("관절 라벨 표시", value=False, key=f"{key_prefix}_show_labels")
    align_view = st.checkbox("사람 기준 좌표로 정렬", value=True, key=f"{key_prefix}_align_view")
    frame_idx = st.slider(
        "3D 프레임",
        min_value=0,
        max_value=n_frames - 1,
        value=min(n_frames // 2, n_frames - 1),
        step=1,
        key=f"{key_prefix}_frame_idx",
    )

    pts = _canonicalize_pose3d_frame_for_display(arr[frame_idx]) if align_view else arr[frame_idx].copy()

    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Preserve joint topology with line segments (None-separated polyline).
    lx, ly, lz = [], [], []
    for a, b in COCO_BONES_17:
        if a >= n_joints or b >= n_joints:
            continue
        if not (np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all()):
            continue
        lx += [float(pts[a, 0]), float(pts[b, 0]), None]
        ly += [float(pts[a, 1]), float(pts[b, 1]), None]
        lz += [float(pts[a, 2]), float(pts[b, 2]), None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=lx,
            y=ly,
            z=lz,
            mode="lines",
            line=dict(color="#6EC1FF", width=6),
            hoverinfo="skip",
            name="bones",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text" if show_labels else "markers",
            marker=dict(size=5, color="#FFB347"),
            text=[COCO_KEYPOINT_NAMES[i] if i < len(COCO_KEYPOINT_NAMES) else str(i) for i in range(n_joints)] if show_labels else None,
            textposition="top center",
            name="joints",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.2, z=1.1)),
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")


def _rotation_matrix(yaw_deg, pitch_deg):
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    rot_y = np.array(
        [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]],
        dtype=np.float32,
    )
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float32,
    )
    return rot_x @ rot_y


def _canonicalize_pose3d_frame_for_display(pts):
    """시각화용으로 몸통을 세우고 발을 아래로 정렬한다."""
    pts = np.asarray(pts, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[-1] != 3 or pts.shape[0] == 0:
        return pts

    n_joints = pts.shape[0]
    if n_joints <= 12:
        return pts

    if np.isfinite(pts[11]).all() and np.isfinite(pts[12]).all():
        pelvis = (pts[11] + pts[12]) * 0.5
    else:
        pelvis = np.nanmedian(pts, axis=0)

    if np.isfinite(pts[5]).all() and np.isfinite(pts[6]).all():
        shoulders = (pts[5] + pts[6]) * 0.5
        up = shoulders - pelvis
        right = pts[6] - pts[5]
    else:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-6 or not np.isfinite(up_norm):
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up_norm = 1.0
    y_axis = up / up_norm

    x_axis = right - np.dot(right, y_axis) * y_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6 or not np.isfinite(x_norm):
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm <= 1e-6 or not np.isfinite(x_norm):
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x_norm = 1.0
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6 or not np.isfinite(z_norm):
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        z_axis = z_axis - np.dot(z_axis, y_axis) * y_axis - np.dot(z_axis, x_axis) * x_axis
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm <= 1e-6 or not np.isfinite(z_norm):
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            z_norm = 1.0
    z_axis = z_axis / z_norm

    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    centered = pts - pelvis[None, :]
    aligned = centered @ basis

    ankle_ids = [idx for idx in (15, 16) if idx < n_joints and np.isfinite(aligned[idx]).all()]
    if ankle_ids:
        aligned[:, 1] -= float(np.nanmin(aligned[ankle_ids, 1]))

    return aligned


def _render_pose3d_video_files(pose_3d_path, out_dir, fps=30.0, width=960, height=720, _attempt=0):
    """3D 포즈 좌표를 다중 카메라 시점(mp4)으로 렌더링한다."""
    if not pose_3d_path or not os.path.exists(pose_3d_path):
        return {}

    arr = np.asarray(np.load(pose_3d_path), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.shape[0] <= 0:
        return {}

    os.makedirs(os.path.abspath(out_dir), exist_ok=True)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return {}

    aligned_frames = np.stack([_canonicalize_pose3d_frame_for_display(arr[t]) for t in range(arr.shape[0])], axis=0)

    flat = aligned_frames.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.size == 0:
        return {}

    view_paths = {}
    writers = {}
    rots = {}
    scales = {}
    for view_key, spec in POSE3D_VIEW_SPECS.items():
        out_path = os.path.join(out_dir, f"poseformer_3d_{view_key}_raw.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(1.0, fps)),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            continue
        rot = _rotation_matrix(spec["yaw_deg"], spec["pitch_deg"])
        flat_rot = (rot @ flat.T).T
        span_x = float(np.max(flat_rot[:, 0]) - np.min(flat_rot[:, 0]))
        span_y = float(np.max(flat_rot[:, 1]) - np.min(flat_rot[:, 1]))
        span = max(span_x, span_y, 1e-6)
        scales[view_key] = 0.40 * min(width, height) / span
        writers[view_key] = writer
        rots[view_key] = rot
        view_paths[view_key] = out_path

    if not writers:
        if _attempt < 2:
            next_w = max(480, int(width * (0.75 if _attempt == 0 else 0.67)))
            next_h = max(360, int(height * (0.75 if _attempt == 0 else 0.67)))
            return _render_pose3d_video_files(
                pose_3d_path,
                out_dir,
                fps=fps,
                width=next_w,
                height=next_h,
                _attempt=_attempt + 1,
            )
        raise RuntimeError("OpenCV VideoWriter initialization failed for all 3D view outputs")

    try:
        for t in range(aligned_frames.shape[0]):
            pts3 = aligned_frames[t]
            for view_key, writer in writers.items():
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (12, 18, 28)

                ptsr = (rots[view_key] @ pts3.T).T
                proj = np.full((ptsr.shape[0], 2), np.nan, dtype=np.float32)
                valid = np.isfinite(ptsr).all(axis=1)
                proj[valid, 0] = width * 0.5 + ptsr[valid, 0] * scales[view_key]
                proj[valid, 1] = height * 0.60 - ptsr[valid, 1] * scales[view_key]

                for a, b in COCO_BONES_17:
                    if a >= proj.shape[0] or b >= proj.shape[0]:
                        continue
                    if not (np.isfinite(proj[a]).all() and np.isfinite(proj[b]).all()):
                        continue
                    pa = (int(proj[a, 0]), int(proj[a, 1]))
                    pb = (int(proj[b, 0]), int(proj[b, 1]))
                    cv2.line(frame, pa, pb, (105, 193, 255), 2, cv2.LINE_AA)

                for j in range(proj.shape[0]):
                    if not np.isfinite(proj[j]).all():
                        continue
                    pj = (int(proj[j, 0]), int(proj[j, 1]))
                    cv2.circle(frame, pj, 3, (255, 190, 80), -1, cv2.LINE_AA)

                cv2.putText(
                    frame,
                    f"3D Pose {POSE3D_VIEW_SPECS[view_key]['label']}  frame {t + 1}/{aligned_frames.shape[0]}",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (210, 220, 230),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
    finally:
        for writer in writers.values():
            writer.release()

    ok_paths = {
        key: path
        for key, path in view_paths.items()
        if os.path.exists(path) and os.path.getsize(path) > 0
    }
    if not ok_paths:
        if _attempt < 2:
            next_w = max(480, int(width * (0.75 if _attempt == 0 else 0.67)))
            next_h = max(360, int(height * (0.75 if _attempt == 0 else 0.67)))
            return _render_pose3d_video_files(
                pose_3d_path,
                out_dir,
                fps=fps,
                width=next_w,
                height=next_h,
                _attempt=_attempt + 1,
            )
        raise RuntimeError("3D render files were not produced (all outputs missing or empty)")
    return ok_paths


def _render_results(state):
    overlay_h264 = state["overlay_h264"]
    skeleton_only_h264 = state["skeleton_only_h264"]
    csv_path = state["csv_path"]
    png_path = state["png_path"]
    report_path = state["report_path"]
    summary_path = state.get("summary_path")
    compare_png_path = state.get("compare_png_path")
    compare_summary_path = state.get("compare_summary_path")
    compare_report_path = state.get("compare_report_path")
    pose_3d_path = state.get("pose_3d_path")
    pose_3d_csv_path = state.get("pose_3d_csv_path")
    pose_3d_video_paths = state.get("pose_3d_video_paths") or {}
    pose_3d_results = state.get("pose_3d_results") or None
    angles_csv = state["angles_csv"]
    viewer_html = state.get("viewer_html")
    skeleton_viewer_html = state.get("skeleton_viewer_html")
    det = state["det"]
    info = state["info"]
    pose_3d_quality = info.get("pose_3d_quality") if isinstance(info, dict) else None
    pose_3d_layout = info.get("pose_3d_layout") if isinstance(info, dict) else None

    st.success(f"완료 - 검출률 {det * 100:.0f}% "
               f"({info.get('detected', '?')}/{info.get('frames', '?')} 프레임)")

    with open(overlay_h264, "rb") as f:
        overlay_bytes = f.read()
    with open(skeleton_only_h264, "rb") as f:
        skeleton_bytes = f.read()

    left, right = st.columns(2)
    with left:
        st.subheader("오버레이 영상")
        if viewer_html and os.path.exists(viewer_html):
            with open(viewer_html, "r", encoding="utf-8") as fh:
                st.components.v1.html(fh.read(), height=920, scrolling=False)
        else:
            st.video(overlay_bytes, format=_video_mime_from_path(overlay_h264))
    with right:
        st.subheader("스켈레톤 전용 영상")
        if skeleton_viewer_html and os.path.exists(skeleton_viewer_html):
            with open(skeleton_viewer_html, "r", encoding="utf-8") as fh:
                st.components.v1.html(fh.read(), height=920, scrolling=False)
        elif viewer_html and os.path.exists(viewer_html):
            with open(viewer_html, "r", encoding="utf-8") as fh:
                st.components.v1.html(fh.read(), height=920, scrolling=False)
        else:
            st.video(skeleton_bytes, format=_video_mime_from_path(skeleton_only_h264))

        if pose_3d_results and isinstance(pose_3d_results, dict):
            st.markdown("#### 3D 결과 비교")
            model_items = list(pose_3d_results.items())
            tabs = st.tabs([k.upper() for k, _ in model_items])
            for tab, (model_key, model_result) in zip(tabs, model_items):
                with tab:
                    model_quality = model_result.get("quality") if isinstance(model_result, dict) else None
                    model_layout = model_result.get("layout") if isinstance(model_result, dict) else None
                    _render_pose3d_interactive(model_result.get("output_path"), key_prefix=f"result_pose3d_{model_key}")
                    if isinstance(model_quality, dict):
                        if isinstance(model_layout, dict):
                            selected_layout = model_layout.get("selected_layout")
                            requested_layout = model_layout.get("requested_layout")
                            if requested_layout == "auto":
                                st.caption(f"출력 레이아웃 자동선택: {selected_layout}")
                            else:
                                st.caption(f"출력 레이아웃: {selected_layout}")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("발목<골반 비율", f"{100.0 * float(model_quality.get('ankle_below_hip_ratio', 0.0)):.1f}%")
                            st.metric("수직 체인 정상 비율", f"{100.0 * float(model_quality.get('vertical_chain_ok_ratio', 0.0)):.1f}%")
                        with c2:
                            bone_cv = model_quality.get("bone_length_cv")
                            sym_u = model_quality.get("upper_arm_symmetry_error")
                            st.metric("뼈길이 CV", "-" if bone_cv is None else f"{float(bone_cv):.3f}")
                            st.metric("좌우 팔대칭 오차", "-" if sym_u is None else f"{100.0 * float(sym_u):.1f}%")
                        if model_quality.get("pass_all"):
                            st.success("3D 인체 형태 품질 기준 통과")
                        else:
                            st.warning("3D 인체 형태 품질 기준 미달 - IK/ROM/스무딩 파라미터 조정 권장")
        else:
            _render_pose3d_interactive(
                pose_3d_path,
                key_prefix="result_pose3d",
            )
            if isinstance(pose_3d_quality, dict):
                st.markdown("#### 3D 인체 형태 품질")
                if isinstance(pose_3d_layout, dict):
                    selected_layout = pose_3d_layout.get("selected_layout")
                    requested_layout = pose_3d_layout.get("requested_layout")
                    if requested_layout == "auto":
                        st.caption(f"출력 레이아웃 자동선택: {selected_layout}")
                    else:
                        st.caption(f"출력 레이아웃: {selected_layout}")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("발목<골반 비율", f"{100.0 * float(pose_3d_quality.get('ankle_below_hip_ratio', 0.0)):.1f}%")
                    st.metric("수직 체인 정상 비율", f"{100.0 * float(pose_3d_quality.get('vertical_chain_ok_ratio', 0.0)):.1f}%")
                with c2:
                    bone_cv = pose_3d_quality.get("bone_length_cv")
                    sym_u = pose_3d_quality.get("upper_arm_symmetry_error")
                    st.metric("뼈길이 CV", "-" if bone_cv is None else f"{float(bone_cv):.3f}")
                    st.metric("좌우 팔대칭 오차", "-" if sym_u is None else f"{100.0 * float(sym_u):.1f}%")
                if pose_3d_quality.get("pass_all"):
                    st.success("3D 인체 형태 품질 기준 통과")
                else:
                    st.warning("3D 인체 형태 품질 기준 미달 - IK/ROM/스무딩 파라미터 조정 권장")
        if pose_3d_video_paths:
            st.markdown("#### 3D 렌더링 영상")
            view_items = [(k, v) for k, v in pose_3d_video_paths.items() if os.path.exists(v)]
            if view_items:
                tabs = st.tabs([POSE3D_VIEW_SPECS.get(k, {}).get("label", k) for k, _ in view_items])
                for tab, (view_key, view_path) in zip(tabs, view_items):
                    with tab:
                        with open(view_path, "rb") as f:
                            st.video(f.read(), format=_video_mime_from_path(view_path))

    st.subheader("운동학 시계열 그래프")
    if os.path.exists(png_path):
        g_left, g_center, g_right = st.columns([1, 2, 1])
        with g_center:
            st.image(png_path, width=760)

    if compare_png_path and os.path.exists(compare_png_path):
        st.subheader("유저 vs 선수 비교 그래프")
        g_left, g_center, g_right = st.columns([1, 2, 1])
        with g_center:
            st.image(compare_png_path, width=900)

    if compare_summary_path and os.path.exists(compare_summary_path):
        st.subheader("유저 vs 선수 비교 요약")
        with open(compare_summary_path, encoding="utf-8") as f:
            st.markdown(_normalize_display_text(f.read()))

    if compare_report_path and os.path.exists(compare_report_path):
        st.subheader("학생선수 발전 포인트")
        with open(compare_report_path, encoding="utf-8") as f:
            st.markdown(_normalize_display_text(f.read()))

    st.subheader("속도 비교 요약")
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            st.markdown(_normalize_display_text(f.read()))

    st.subheader("다운로드")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.download_button("오버레이 영상", data=overlay_bytes,
                           file_name=os.path.basename(overlay_h264),
                           mime=_video_mime_from_path(overlay_h264))
    with d2:
        st.download_button("스켈레톤 전용 영상", data=skeleton_bytes,
                           file_name=os.path.basename(skeleton_only_h264),
                           mime=_video_mime_from_path(skeleton_only_h264))
    with d3:
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button("관절좌표 (keypoints.csv)", data=f.read(),
                                   file_name="keypoints.csv", mime="text/csv")
    with d4:
        if pose_3d_path and os.path.exists(pose_3d_path):
            with open(pose_3d_path, "rb") as f:
                st.download_button("3D 포즈 (npy)", data=f.read(),
                                   file_name=os.path.basename(pose_3d_path),
                                   mime="application/octet-stream")
        else:
            st.caption("3D npy 없음")
    with d5:
        if pose_3d_csv_path and os.path.exists(pose_3d_csv_path):
            with open(pose_3d_csv_path, "rb") as f:
                st.download_button("3D 포즈 (csv)", data=f.read(),
                                   file_name=os.path.basename(pose_3d_csv_path),
                                   mime="text/csv")
        else:
            st.caption("3D csv 없음")
    if os.path.exists(angles_csv):
        with open(angles_csv, "rb") as f:
            st.download_button("운동 데이터 (angles.csv)", data=f.read(),
                               file_name="angles.csv", mime="text/csv")
    for view_key, view_path in pose_3d_video_paths.items():
        if not os.path.exists(view_path):
            continue
        label = POSE3D_VIEW_SPECS.get(view_key, {}).get("label", view_key)
        with open(view_path, "rb") as f:
            st.download_button(
                f"3D 렌더링 영상 ({label})",
                data=f.read(),
                file_name=os.path.basename(view_path),
                mime="video/mp4",
            )


# ------------------------------- 화면 -------------------------------

st.set_page_config(
    page_title="투구 동작 분석",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/* Prevent any browser/theme style from cropping the video frame. */
[data-testid="stVideo"] video {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    background: #000 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("투구 동작 분석 - 컴퓨터비전 데모")
st.caption(
    "영상을 업로드하면 관절 스켈레톤(졸라맨) 오버레이와 투구 운동학 리포트를 생성합니다. "
    "분석 결과에는 골반/몸통/팔 속도 비교 요약과 피크 순서도 함께 표시됩니다."
)

st.subheader("비교 기준 선택")
reference_profile = st.radio(
    "비교할 투수 유형을 먼저 선택하세요",
    ["우완 오버 투수", "사이드암"],
    index=0,
    horizontal=True,
    help="업로드 전 비교할 기준 투수 유형을 선택하세요.",
)
st.info(f"현재 기준: {reference_profile}")

with st.sidebar:
    st.header("설정")
    engine = st.selectbox("2D 자세추정 엔진", ["yolo", "rtmpose"], index=0)
    preprocess = st.checkbox("역광/저조도 보정 (CLAHE)", value=False,
                             help="석양/역광 정면 영상에 권장")
    hand = st.selectbox("던지는 팔", ["auto", "L", "R"], index=0,
                        help="auto: 손목 최고속으로 자동 판정")
    st.divider()
    st.subheader("선택 3D 추론")
    poseformer_enabled = st.checkbox(
        "3D 추론 사용",
        value=True,
        help="2D 키포인트 시퀀스에서 프레임별 3D 관절 좌표를 생성합니다.",
    )
    pose3d_model_name = st.selectbox(
        "3D 백엔드",
        ["poseformer", "mixste", "both"],
        index=0,
        disabled=not poseformer_enabled,
        help="both를 선택하면 두 3D 모델을 같은 입력으로 각각 실행합니다.",
    )
    default_poseformer_repo = _default_poseformer_repo_dir()
    default_mixste_repo = _default_mixste_repo_dir()
    default_rtmpose_repo = _default_rtmpose_repo_dir()
    default_mixste_checkpoint = _default_mixste_checkpoint_path(default_mixste_repo)
    poseformer_repo_dir = st.text_input(
        "PoseFormerV2 저장소 경로",
        value=default_poseformer_repo,
        disabled=not poseformer_enabled,
    )
    default_poseformer_checkpoint = _default_poseformer_checkpoint_path(default_poseformer_repo)
    poseformer_checkpoint_path = st.text_input(
        "PoseFormerV2 체크포인트 경로",
        value=default_poseformer_checkpoint,
        disabled=not poseformer_enabled,
        help="PoseFormerV2를 사용할 때만 필요합니다.",
    )
    mixste_repo_dir = st.text_input(
        "MixSTE 저장소 경로",
        value=default_mixste_repo,
        disabled=not poseformer_enabled,
        help="MixSTE를 사용할 때만 필요합니다.",
    )
    mixste_checkpoint_path = st.text_input(
        "MixSTE 체크포인트 경로",
        value=default_mixste_checkpoint,
        disabled=not poseformer_enabled,
        help="자동 탐색된 체크포인트가 있으면 채워집니다. 없으면 직접 입력해야 합니다.",
    )
    rtmpose_repo_dir = st.text_input(
        "RTMPose 저장소 경로",
        value=default_rtmpose_repo,
        disabled=(engine != "rtmpose"),
        help="RTMPose를 사용할 때만 필요합니다.",
    )
    rtmpose_config_path = st.text_input(
        "RTMPose config 경로 또는 model alias",
        value=_default_rtmpose_config_path(default_rtmpose_repo),
        disabled=(engine != "rtmpose"),
        help="로컬 config 파일 경로나 model alias 둘 다 가능합니다. 가능한 경우 실제 config 파일을 우선 찾습니다.",
    )
    rtmpose_checkpoint_path = st.text_input(
        "RTMPose checkpoint 경로",
        value="",
        disabled=(engine != "rtmpose"),
        help="config가 model alias인 경우 비워도 됩니다.",
    )
    poseformer_num_frames = st.selectbox(
        "3D 입력 프레임 수",
        [27, 81, 243],
        index=0,
        disabled=not poseformer_enabled,
    )
    poseformer_num_kept_frames = st.number_input(
        "유지 프레임 수",
        min_value=1,
        value=3,
        step=1,
        disabled=not poseformer_enabled,
    )
    poseformer_num_kept_coeffs = st.number_input(
        "유지 계수 수",
        min_value=1,
        value=3,
        step=1,
        disabled=not poseformer_enabled,
    )
    poseformer_source_layout = st.selectbox(
        "PoseFormer 출력 관절 레이아웃",
        ["auto", "h36m", "coco"],
        index=0,
        disabled=not poseformer_enabled,
        help="auto는 출력 형태를 점수화해 h36m/coco 중 더 사람다운 결과를 자동 선택합니다.",
    )
    st.caption("렌더 환경 점검")
    render_env = _runtime_render_capability()
    if render_env["opencv_writer_ok"]:
        st.success("OpenCV mp4 writer: 사용 가능")
    else:
        st.warning("OpenCV mp4 writer: 제한됨 (해상도 자동 폴백 사용)")
    if render_env["ffmpeg_ok"]:
        st.success("ffmpeg(H.264): 사용 가능")
    else:
        st.warning("ffmpeg 없음: H.264 변환 없이 원본 mp4v로 제공")
    st.divider()
    st.markdown(
        "**촬영 권장**\n\n"
        "- 최소 240fps 슬로모\n- 밝은 환경/단순 배경\n- 피사체를 크게\n"
        "- 삼각대 고정, 측면/정면 각도"
    )

uploaded = st.file_uploader("유저 투구 영상 업로드 (mp4/mov)", type=["mp4", "mov", "m4v", "avi"])
keypoints_uploaded = st.file_uploader(
    "선택: COCO 17 keypoints CSV 업로드",
    type=["csv"],
    help="이미 추출한 2D 관절 CSV가 있으면 이 파일로 검출 단계를 건너뛸 수 있습니다.",
)
st.info(f"선택된 기준: {reference_profile}. 업로드 전 이 유형과 비교할 기준을 정한 뒤 분석을 시작하세요.")

if uploaded is not None and st.session_state.get("last_uploaded_name") != uploaded.name:
    _clear_analysis_state()
    st.session_state.last_uploaded_name = uploaded.name

if keypoints_uploaded is not None and st.session_state.get("last_keypoints_name") != keypoints_uploaded.name:
    _clear_analysis_state()
    st.session_state.last_keypoints_name = keypoints_uploaded.name

col_run, col_reset = st.columns([1, 1])
run = col_run.button("분석 시작", type="primary", disabled=uploaded is None)
reset = col_reset.button("초기화", disabled=uploaded is None and keypoints_uploaded is None)

if reset:
    _clear_analysis_state()
    st.session_state.last_uploaded_name = None
    st.session_state.last_keypoints_name = None

analysis_state = st.session_state.get("analysis_state")

# 코드 업데이트 후 구버전 결과가 남아 보이지 않도록 스키마 버전 검사.
if analysis_state is not None and analysis_state.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
    _clear_analysis_state()
    analysis_state = None

if analysis_state is not None and uploaded is not None:
    _render_results(analysis_state)
elif run and uploaded is not None:
    keypoints_override = None
    keypoints_csv_temp = None
    if keypoints_uploaded is not None:
        keypoints_csv_temp = os.path.join(tempfile.mkdtemp(prefix="pitch_kpts_"), "keypoints.csv")
        with open(keypoints_csv_temp, "wb") as f:
            f.write(keypoints_uploaded.getbuffer())
        try:
            keypoints_override = analyze_mod._load_keypoints_csv(keypoints_csv_temp, expected_names=COCO_KEYPOINT_NAMES)
        except Exception as e:  # noqa: BLE001
            st.error(f"keypoints CSV를 읽지 못했습니다: {e}")
            st.stop()

    if keypoints_override is None:
        warn, ok = _model_ready(
            engine,
            rtmpose_config_path=rtmpose_config_path,
            rtmpose_checkpoint_path=rtmpose_checkpoint_path,
        )
        if not ok:
            st.error(warn)
            st.stop()
    pose_warn, pose_ok = _poseformer_ready(
        poseformer_enabled,
        poseformer_repo_dir,
        poseformer_checkpoint_path,
    )
    if not pose_ok:
        st.error(pose_warn)
        st.stop()

    if poseformer_enabled:
        if pose3d_model_name in {"poseformer", "both"}:
            if not poseformer_repo_dir or not poseformer_checkpoint_path:
                st.error("PoseFormerV2 저장소/체크포인트 경로를 확인하세요.")
                st.stop()
        if pose3d_model_name in {"mixste", "both"}:
            if not mixste_repo_dir or not mixste_checkpoint_path:
                st.error("MixSTE 저장소/체크포인트 경로를 확인하세요.")
                st.stop()

    workdir = tempfile.mkdtemp(prefix="pitch_")
    in_suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    in_path = os.path.join(workdir, "input" + in_suffix)
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    overlay_raw = os.path.join(workdir, "overlay_raw.webm")
    overlay_h264 = overlay_raw
    skeleton_only_raw = os.path.join(workdir, "overlay_raw_skeleton_only.webm")
    skeleton_only_h264 = skeleton_only_raw
    csv_path = os.path.join(workdir, "pose_keypoints.csv")
    pose_3d_path = os.path.join(workdir, "poseformer_3d.npy") if poseformer_enabled else None
    pose_3d_csv_path = os.path.join(workdir, "poseformer_3d.csv") if poseformer_enabled else None
    pose_3d_video_paths = {}
    pose_3d_results = None

    pose3d_model_names = [pose3d_model_name]
    if pose3d_model_name == "both":
        pose3d_model_names = ["poseformer", "mixste"]

    if engine == "yolo":
        engine_kwargs = {}
    else:
        engine_kwargs = {
            "config": rtmpose_config_path,
            "checkpoint": rtmpose_checkpoint_path,
            "repo_dir": rtmpose_repo_dir,
        }
    if keypoints_override is not None:
        engine_kwargs = {}

    try:
        with st.spinner("1) 자세추정 및 스켈레톤 오버레이 생성 중..."):
            info = analyze_mod.analyze_video(
                in_path, overlay_raw, engine_name=engine,
                csv_path=csv_path, smooth=True, preprocess=preprocess,
                progress=False, skeleton_only_outp=skeleton_only_raw,
                kpts_seq_override=keypoints_override,
                keypoint_names=COCO_KEYPOINT_NAMES,
                edges=COCO_BONES_17,
                poseformer_enforce_readiness=(keypoints_override is None),
                poseformer_repo_dir=poseformer_repo_dir if poseformer_enabled else None,
                poseformer_checkpoint_path=poseformer_checkpoint_path if poseformer_enabled else None,
                mixste_repo_dir=mixste_repo_dir if poseformer_enabled else None,
                mixste_checkpoint_path=mixste_checkpoint_path if poseformer_enabled else None,
                poseformer_output_path=pose_3d_path,
                poseformer_csv_path=pose_3d_csv_path,
                poseformer_runner_kwargs={
                    "num_frames": int(poseformer_num_frames),
                    "num_kept_frames": int(poseformer_num_kept_frames),
                    "num_kept_coeffs": int(poseformer_num_kept_coeffs),
                } if poseformer_enabled else None,
                poseformer_source_layout=poseformer_source_layout if poseformer_enabled else "auto",
                pose3d_model_name=pose3d_model_names[0] if pose3d_model_names else "poseformer",
                pose3d_model_names=pose3d_model_names,
                **engine_kwargs,
            )
    except Exception as e:  # noqa: BLE001
        st.error(f"자세추정 실패: {e}")
        st.stop()

    det = _detection_rate(csv_path)
    if det <= 0.0:
        st.error("사람 자세를 검출하지 못했습니다. 밝은 환경에서 피사체가 크게 나오도록 "
                 "다시 촬영하거나, 사이드바에서 CLAHE 보정 또는 YOLO 엔진을 시도하세요.")
        st.stop()

    if poseformer_enabled and info.get("pose_3d_path") and os.path.exists(info.get("pose_3d_path")):
        try:
            with st.spinner("2) 3D 포즈 렌더링 영상 생성 중..."):
                rendered_map = _render_pose3d_video_files(
                    info.get("pose_3d_path"),
                    workdir,
                    fps=float(info.get("fps", 30.0) or 30.0),
                )
                for view_key, raw_path in rendered_map.items():
                    h264_path = os.path.join(workdir, f"poseformer_3d_{view_key}.mp4")
                    try:
                        converted, _ = _to_h264(raw_path, h264_path)
                        if os.path.exists(converted) and os.path.getsize(converted) > 0:
                            pose_3d_video_paths[view_key] = converted
                    except Exception as sub_exc:  # noqa: BLE001
                        st.warning(f"3D 뷰 '{view_key}' 인코딩 실패: {type(sub_exc).__name__}: {sub_exc}")
            if not pose_3d_video_paths:
                st.warning("3D 렌더링 영상 생성 결과가 비어 있습니다. 코덱/ffmpeg 환경을 확인해 주세요.")
        except Exception as e:  # noqa: BLE001
            diag_path = os.path.join(workdir, "pose3d_render_diag.json")
            diag = {
                "error_type": type(e).__name__,
                "error": str(e),
                "render_env": _runtime_render_capability(),
                "pose_3d_path": info.get("pose_3d_path"),
                "pose_3d_exists": bool(info.get("pose_3d_path") and os.path.exists(info.get("pose_3d_path"))),
                "pose_3d_size": os.path.getsize(info.get("pose_3d_path")) if info.get("pose_3d_path") and os.path.exists(info.get("pose_3d_path")) else 0,
            }
            with open(diag_path, "w", encoding="utf-8") as fh:
                json.dump(diag, fh, ensure_ascii=False, indent=2)
            st.warning(f"3D 렌더링 영상 생성 실패: {type(e).__name__}: {e}")
            st.caption(f"진단 로그: {diag_path}")
            pose_3d_video_paths = {}

    try:
        with st.spinner("3) 속도/타이밍 계산 / 리포트 생성..."):
            main_result = kin_mod.analyze(csv_path, hand_mode=hand)
    except Exception as e:  # noqa: BLE001
        st.error(f"운동학 분석 실패: {e}\n"
                 "검출률이 낮거나 투구 동작이 짧으면 단계 검출이 어려울 수 있습니다.")
        st.stop()

    reference_result = None
    compare_png_path = None
    compare_summary_path = None
    compare_report_path = None
    reference_csv = _resolve_reference_csv_path(reference_profile)
    reference_label = ref_data.get_reference_profile_label(reference_profile)
    if reference_csv is not None:
        try:
            reference_result = kin_mod.analyze(reference_csv, hand_mode=hand)
            compare_png_path = os.path.join(workdir, "compare_kinematics.png")
            compare_summary_path = os.path.join(workdir, "compare_summary.md")
            compare_report_path = os.path.join(workdir, "compare_report.md")
            compare_series = [
                {"label": "유저", "metrics": main_result["metrics"], "timing": main_result["timing"], "fps": main_result["fps"]},
                {"label": reference_label, "metrics": reference_result["metrics"], "timing": reference_result["timing"], "fps": reference_result["fps"]},
            ]
            kin_mod.plot_compare_kinematics(compare_png_path, compare_series[0], compare_series[1])
            comparison = kin_mod.summarize_reference_comparison(main_result, reference_result)
            kin_mod.write_reference_comparison_summary(compare_summary_path, comparison)
            kin_mod.write_reference_comparison_report(compare_report_path, comparison)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{reference_label} 기준 비교 생성 실패: {e}")
    else:
        st.info(f"{reference_label} 기준 데이터를 찾지 못해 비교 그래프를 생성하지 못했습니다.")

    stem = os.path.join(workdir, "pose")
    png_path = stem + "_kinematics.png"
    report_path = stem + "_report.md"
    summary_path = stem + "_speed_summary.md"
    angles_csv = stem + "_angles.csv"
    viewer_html = info.get("viewer_html")
    skeleton_viewer_html = info.get("skeleton_viewer_html")

    analysis_state = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "workdir": workdir,
        "overlay_h264": overlay_h264,
        "skeleton_only_h264": skeleton_only_h264,
        "csv_path": csv_path,
        "png_path": png_path,
        "report_path": report_path,
        "summary_path": summary_path,
        "compare_png_path": compare_png_path,
        "compare_summary_path": compare_summary_path,
        "compare_report_path": compare_report_path,
        "pose_3d_path": info.get("pose_3d_path"),
        "pose_3d_csv_path": info.get("pose_3d_csv_path"),
        "pose_3d_results": info.get("pose_3d_results"),
        "pose_3d_video_paths": pose_3d_video_paths,
        "poseformer_enabled": bool(poseformer_enabled),
        "angles_csv": angles_csv,
        "viewer_html": viewer_html,
        "skeleton_viewer_html": skeleton_viewer_html,
        "det": det,
        "info": info,
    }
    st.session_state.analysis_state = analysis_state
    st.session_state.view_mode = "overlay"
    st.session_state.last_uploaded_name = uploaded.name
    _render_results(analysis_state)

st.divider()
st.caption(
    "주의: 2D 단일 카메라 한계 - 소비자용 30fps는 릴리스 순간을 과소표집하며, "
    "횡단면 어깨 회전은 신뢰성 있게 측정되지 않습니다(개발계획서 1.2). "
    "본 데모의 지표 값은 참값 대조 검증(Phase 4) 전까지 상대/추세 비교용입니다."
)

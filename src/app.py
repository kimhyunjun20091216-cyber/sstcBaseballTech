"""투구 동작 분석 데모 UI (Streamlit).

영상을 업로드하면 스켈레톤("졸라맨") 오버레이 영상과 운동학 리포트를
브라우저에서 바로 확인한다. KSEF 시연용 (Phase 1 + Phase 3 통합 데모).

실행:
    .venv/bin/streamlit run src/app.py

기존 모듈(analyze.py, kinematics.py)을 재사용하며, 자세추정·운동학 로직을
여기서 재구현하지 않는다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import streamlit as st

# src 디렉토리를 임포트 경로에 추가(모듈 재사용).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import analyze as analyze_mod        # noqa: E402
import kinematics as kin_mod          # noqa: E402


# ------------------------------- 유틸 -------------------------------

def _to_h264(src_mp4, dst_mp4):
    """오버레이 mp4(mp4v)를 브라우저 재생용 H.264로 재인코딩.

    ffmpeg가 없으면 원본을 그대로 복사(재생이 안 될 수 있음).
    """
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
        return dst_mp4, True
    except subprocess.CalledProcessError:
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


def _model_ready(engine):
    """엔진 실행 전 사전 조건 확인. (문제 메시지, ok) 반환."""
    if engine == "mediapipe":
        model = os.path.join(ROOT_DIR, "models", "pose_landmarker_heavy.task")
        if not os.path.exists(model):
            return ("MediaPipe 모델이 없습니다. README의 다운로드 명령을 실행하세요:\n"
                    "`models/pose_landmarker_heavy.task`"), False
    return "", True


# ------------------------------- 화면 -------------------------------

st.set_page_config(page_title="투구 동작 분석", page_icon="⚾", layout="wide")

st.title("⚾ 투구 동작 분석 — 컴퓨터비전 데모")
st.caption(
    "영상을 업로드하면 관절 스켈레톤(졸라맨) 오버레이와 투구 운동학 리포트를 "
    "생성합니다. (Phase 1 자세추정 + Phase 3 운동학 통합 데모)"
)

with st.sidebar:
    st.header("설정")
    engine = st.selectbox("자세추정 엔진", ["mediapipe", "yolo"], index=0,
                          help="mediapipe: 빠름·간편 / yolo: 역광·소형 피사체에 강건")
    preprocess = st.checkbox("역광/저조도 보정 (CLAHE)", value=False,
                             help="석양·역광 정면 영상에 권장")
    hand = st.selectbox("던지는 팔", ["auto", "L", "R"], index=0,
                        help="auto: 손목 최고속으로 자동 판정")
    st.divider()
    st.markdown(
        "**촬영 권장**\n\n"
        "- 최소 240fps 슬로모\n- 밝은 환경·단순 배경\n- 피사체를 크게\n"
        "- 삼각대 고정, 측면/정면 각도"
    )

uploaded = st.file_uploader("투구 영상 업로드 (mp4/mov)", type=["mp4", "mov", "m4v", "avi"])

col_run, _ = st.columns([1, 3])
run = col_run.button("분석 시작", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    warn, ok = _model_ready(engine)
    if not ok:
        st.error(warn)
        st.stop()

    workdir = tempfile.mkdtemp(prefix="pitch_")
    in_suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    in_path = os.path.join(workdir, "input" + in_suffix)
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    overlay_raw = os.path.join(workdir, "overlay_raw.mp4")
    overlay_h264 = os.path.join(workdir, "overlay.mp4")
    # kinematics 스템 규칙(_keypoints 제거)에 맞춘 CSV 이름.
    csv_path = os.path.join(workdir, "pose_keypoints.csv")

    try:
        with st.spinner("① 자세추정 및 스켈레톤 오버레이 생성 중..."):
            info = analyze_mod.analyze_video(
                in_path, overlay_raw, engine_name=engine,
                csv_path=csv_path, smooth=True, preprocess=preprocess,
                progress=False,
            )
    except Exception as e:  # noqa: BLE001
        st.error(f"자세추정 실패: {e}")
        st.stop()

    det = _detection_rate(csv_path)
    if det <= 0.0:
        st.error("사람 자세를 검출하지 못했습니다. 밝은 환경에서 피사체가 크게 나오도록 "
                 "다시 촬영하거나, 사이드바에서 CLAHE 보정 또는 YOLO 엔진을 시도하세요.")
        st.stop()

    with st.spinner("② 브라우저 재생용 인코딩(H.264)..."):
        _to_h264(overlay_raw, overlay_h264)

    try:
        with st.spinner("③ 관절각 계산 · 투구 단계 분할 · 리포트 생성..."):
            result = kin_mod.analyze(csv_path, hand_mode=hand)
    except Exception as e:  # noqa: BLE001
        st.error(f"운동학 분석 실패: {e}\n"
                 "검출률이 낮거나 투구 동작이 짧으면 단계 검출이 어려울 수 있습니다.")
        st.stop()

    stem = os.path.join(workdir, "pose")
    png_path = stem + "_kinematics.png"
    report_path = stem + "_report.md"
    angles_csv = stem + "_angles.csv"

    st.success(f"완료 — 검출률 {det*100:.0f}% "
               f"({info.get('detected', '?')}/{info.get('frames', '?')} 프레임)")

    left, right = st.columns(2)
    with left:
        st.subheader("스켈레톤 오버레이")
        with open(overlay_h264, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
    with right:
        st.subheader("운동학 시계열 그래프")
        if os.path.exists(png_path):
            st.image(png_path, use_container_width=True)

    st.subheader("분석 리포트")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            st.markdown(f.read())

    st.subheader("다운로드")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("오버레이 영상 (mp4)", data=video_bytes,
                           file_name="overlay.mp4", mime="video/mp4")
    with d2:
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button("관절좌표 (keypoints.csv)", data=f.read(),
                                   file_name="keypoints.csv", mime="text/csv")
    with d3:
        if os.path.exists(angles_csv):
            with open(angles_csv, "rb") as f:
                st.download_button("관절각 (angles.csv)", data=f.read(),
                                   file_name="angles.csv", mime="text/csv")

st.divider()
st.caption(
    "⚠️ 2D 단일 카메라 한계: 소비자용 30fps는 릴리스 순간을 과소표집하며, "
    "횡단면 어깨 회전은 신뢰성 있게 측정되지 않습니다(개발계획서 §1.2). "
    "본 데모의 각도 값은 참값 대조 검증(Phase 4) 전까지 상대·추세 비교용입니다."
)

"""자세 추정 엔진 추상화.

MVP는 MediaPipe(BlazePose)를 사용한다. 저조도/역광·소형 피사체에서
추정이 실패하면 YOLO-pose 등으로 교체할 수 있도록, 모든 엔진이 동일한
출력 규약(프레임별 관절 좌표 배열)을 따르도록 인터페이스를 통일한다.

출력 규약
---------
estimate(frame_bgr) -> np.ndarray shape (K, 3) 또는 None
    각 행 = (x_px, y_px, visibility[0..1]). 검출 실패 시 None.
    K = len(engine.keypoint_names), 스켈레톤 연결은 engine.edges.
"""

from __future__ import annotations

import os

import numpy as np


# COCO 17 관절을 공통 스켈레톤 표준으로 삼는다(엔진 간 호환·문헌 정합).
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 스켈레톤 연결(졸라맨 뼈대) — COCO 인덱스 기준.
COCO_EDGES = [
    (5, 7), (7, 9),        # 왼팔: 어깨-팔꿈치-손목
    (6, 8), (8, 10),       # 오른팔
    (11, 13), (13, 15),    # 왼다리: 고관절-무릎-발목
    (12, 14), (14, 16),    # 오른다리
    (5, 6), (11, 12),      # 어깨선, 골반선
    (5, 11), (6, 12),      # 몸통 좌/우
    (0, 5), (0, 6),        # 머리-어깨
]


class MediaPipeEngine:
    """MediaPipe PoseLandmarker(BlazePose, Tasks API) 래퍼.

    33개 BlazePose 랜드마크를 COCO 17로 매핑한다. VIDEO 모드로 실행하여
    프레임 간 시간 정보를 활용(추적 안정화).
    """

    name = "mediapipe"
    keypoint_names = COCO_KEYPOINTS
    edges = COCO_EDGES

    # BlazePose 33 랜드마크 → COCO 17 인덱스 매핑.
    _MP_TO_COCO = {
        0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
        11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
        23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
    }
    _DEFAULT_MODEL = "models/pose_landmarker_heavy.task"

    def __init__(self, model_path: str | None = None, min_detection_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode,
        )

        self._mp = mp
        path = model_path or self._DEFAULT_MODEL
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MediaPipe 모델 없음: {path}\n"
                "다운로드: curl -sL -o models/pose_landmarker_heavy.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            num_poses=1,
        )
        self._lm = PoseLandmarker.create_from_options(opts)
        self._ts = 0  # ms 타임스탬프(VIDEO 모드는 단조 증가 필요)

    def estimate(self, frame_bgr):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33  # ~30fps 가정(상대 간격만 의미 있음)
        result = self._lm.detect_for_video(mp_img, self._ts)
        if not result.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        lm = result.pose_landmarks[0]
        out = np.full((17, 3), np.nan, dtype=np.float32)
        for mp_idx, coco_idx in self._MP_TO_COCO.items():
            p = lm[mp_idx]
            vis = getattr(p, "visibility", 1.0)
            out[coco_idx] = (p.x * w, p.y * h, vis)
        return out

    def close(self):
        self._lm.close()


class YoloEngine:
    """Ultralytics YOLO-pose 래퍼(역광·소형 피사체 대비용). COCO 17 네이티브."""

    name = "yolo"
    keypoint_names = COCO_KEYPOINTS
    edges = COCO_EDGES

    def __init__(self, weights: str = "yolo11x-pose.pt", conf: float = 0.25):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._conf = conf

    def estimate(self, frame_bgr):
        res = self._model.predict(frame_bgr, conf=self._conf, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return None
        # 신뢰도 총합이 가장 높은 사람 1명 선택(투수만).
        kp_xy = res.keypoints.xy.cpu().numpy()          # (n, 17, 2)
        kp_conf = res.keypoints.conf.cpu().numpy()      # (n, 17)
        best = int(kp_conf.sum(axis=1).argmax())
        out = np.concatenate([kp_xy[best], kp_conf[best][:, None]], axis=1)
        return out.astype(np.float32)

    def close(self):
        pass


def build_engine(name: str, **kwargs):
    if name == "mediapipe":
        return MediaPipeEngine(**kwargs)
    if name == "yolo":
        return YoloEngine(**kwargs)
    raise ValueError(f"unknown engine: {name!r} (mediapipe|yolo)")

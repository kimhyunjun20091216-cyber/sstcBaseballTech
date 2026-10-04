"""자세 추정 엔진 추상화.

현재 파이프라인은 Ultralytics YOLO-pose를 단일 2D 추정 엔진으로 사용한다.
출력 규약(프레임별 관절 좌표 배열)을 고정해 후속 3D 추론과 운동학 분석이
같은 형식을 소비하도록 유지한다.

출력 규약
---------
estimate(frame_bgr) -> np.ndarray shape (K, 3) 또는 None
    각 행 = (x_px, y_px, visibility[0..1]). 검출 실패 시 None.
    K = len(engine.keypoint_names), 스켈레톤 연결은 engine.edges.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
        return self._extract_best_person(res)

    def estimate_batch(self, frames_bgr, batch_size=16):
        """여러 프레임을 배치 추론으로 처리해 CPU 오버헤드를 줄인다."""
        outputs = []
        frames = list(frames_bgr)
        for start in range(0, len(frames), batch_size):
            chunk = frames[start:start + batch_size]
            results = self._model.predict(chunk, conf=self._conf, verbose=False)
            outputs.extend(self._extract_best_person(res) for res in results)
        return outputs

    def _extract_best_person(self, res):
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


class RtmPoseEngine:
    """MMPose RTMPose top-down 2D 추정기 래퍼."""

    name = "rtmpose"
    keypoint_names = COCO_KEYPOINTS
    edges = COCO_EDGES

    def __init__(self, config: str, checkpoint: str | None = None, device: str = "cuda:0", repo_dir: str | None = None):
        repo_candidates = []
        if repo_dir:
            repo_candidates.append(repo_dir)
        repo_candidates.extend([
            str(Path(__file__).resolve().parents[1] / "mmpose-main" / "mmpose-main"),
            str(Path(__file__).resolve().parents[2] / "mmpose-main" / "mmpose-main"),
        ])
        for candidate in repo_candidates:
            if candidate and os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)

        self._config_or_alias = str(config)
        self._checkpoint = checkpoint
        self._device = device

        if os.path.exists(self._config_or_alias):
            from mmpose.apis import init_model, inference_topdown

            self._init_model = init_model
            self._inference_topdown = inference_topdown
            self._model = init_model(self._config_or_alias, checkpoint, device=device)
            self._inferencer = None
        else:
            from mmpose.apis import MMPoseInferencer

            self._inferencer = MMPoseInferencer(
                pose2d=self._config_or_alias,
                pose2d_weights=checkpoint or None,
                device=device,
                det_model=None,
                show_progress=False,
            )
            self._model = None
        self._device = device

    def _full_frame_bbox(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        return np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)

    def estimate(self, frame_bgr):
        if self._inferencer is not None:
            results = next(self._inferencer(frame_bgr, return_datasamples=True, batch_size=1))
            predictions = results.get("predictions") or []
            return self._extract_first_person(predictions)
        results = self._inference_topdown(self._model, frame_bgr, bboxes=self._full_frame_bbox(frame_bgr), bbox_format="xyxy")
        return self._extract_first_person(results)

    def estimate_batch(self, frames_bgr, batch_size=1):
        outputs = []
        for frame in frames_bgr:
            outputs.append(self.estimate(frame))
        return outputs

    def _extract_first_person(self, results):
        if not results:
            return None
        sample = results[0]
        if isinstance(sample, dict):
            sample = sample.get("predictions", [None])[0] if sample.get("predictions") else None
        if sample is None or not hasattr(sample, "pred_instances"):
            return None
        instances = sample.pred_instances
        if not hasattr(instances, "keypoints") or len(instances.keypoints) == 0:
            return None
        keypoints = np.asarray(instances.keypoints[0], dtype=np.float32)
        if hasattr(instances, "keypoint_scores"):
            scores = np.asarray(instances.keypoint_scores[0], dtype=np.float32)
        else:
            scores = np.ones((keypoints.shape[0],), dtype=np.float32)
        out = np.concatenate([keypoints[:, :2], scores[:, None]], axis=1)
        return out.astype(np.float32)

    def close(self):
        pass


def build_engine(name: str, **kwargs):
    if name == "yolo":
        return YoloEngine(**kwargs)
    if name == "rtmpose":
        return RtmPoseEngine(**kwargs)
    raise ValueError(f"unknown engine: {name!r} (expected yolo or rtmpose)")

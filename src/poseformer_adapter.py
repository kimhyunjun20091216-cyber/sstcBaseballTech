from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO_LEFT_JOINTS = [1, 3, 5, 7, 9, 11, 13, 15]
COCO_RIGHT_JOINTS = [2, 4, 6, 8, 10, 12, 14, 16]
POSEFORMER_MIN_CONFIDENCE_BY_JOINT = {
    "left_shoulder": 0.45,
    "right_shoulder": 0.45,
    "left_elbow": 0.5,
    "right_elbow": 0.5,
    "left_wrist": 0.6,
    "right_wrist": 0.6,
}


def validate_poseformer_joint_contract(joint_names, left_joints=None, right_joints=None):
    """PoseFormerV2 입력 관절 순서와 좌우 인덱스 계약을 검증한다."""
    names = list(joint_names)
    if names != COCO_KEYPOINT_NAMES:
        raise ValueError(
            "PoseFormerV2 integration expects COCO_KEYPOINT_NAMES in fixed order; "
            f"got {names!r}"
        )

    left = list(COCO_LEFT_JOINTS if left_joints is None else left_joints)
    right = list(COCO_RIGHT_JOINTS if right_joints is None else right_joints)
    if len(left) != len(right):
        raise ValueError("left_joints and right_joints must have the same length")
    if len(set(left + right)) != len(left) + len(right):
        raise ValueError("left_joints and right_joints must not overlap")
    if any(idx < 0 or idx >= len(names) for idx in left + right):
        raise ValueError("left_joints and right_joints must be valid joint indices")
    return {"joint_names": names, "left_joints": left, "right_joints": right}


def resolve_poseformer_confidence_thresholds(joint_names, min_confidence=0.35, per_joint_min_confidence=None):
    """관절별 최소 confidence 임계값 벡터를 만든다."""
    names = list(joint_names)
    overrides = dict(POSEFORMER_MIN_CONFIDENCE_BY_JOINT)
    if per_joint_min_confidence:
        overrides.update(per_joint_min_confidence)

    thresholds = np.full(len(names), float(min_confidence), dtype=np.float32)
    for idx, name in enumerate(names):
        thresholds[idx] = float(max(min_confidence, overrides.get(name, min_confidence)))
    return thresholds


def prepare_poseformer_sequence(kpts_seq, min_confidence=0.35, joint_names=None, per_joint_min_confidence=None):
    """2D 키포인트 시퀀스를 PoseFormerV2 실험용 입력 포맷으로 변환한다.

    입력은 각 프레임별 키포인트 배열의 리스트이며, 각 배열은 shape (17, 3)
    또는 None 이어야 한다. 출력은 np.ndarray 기반으로 구성되며, 모델이
    받기 쉬운 좌표/가중치/valid mask 형태를 제공한다.
    """
    if not kpts_seq:
        raise ValueError("kpts_seq must not be empty")

    names = list(joint_names or COCO_KEYPOINT_NAMES)
    validate_poseformer_joint_contract(names)

    valid_frames = []
    for kpts in kpts_seq:
        if kpts is None:
            valid_frames.append(np.full((17, 3), np.nan, dtype=np.float32))
        else:
            arr = np.asarray(kpts, dtype=np.float32)
            if arr.shape != (17, 3):
                raise ValueError(f"each keypoint frame must have shape (17, 3), got {arr.shape}")
            valid_frames.append(arr)

    sequence = np.stack([
        np.asarray(frame[:, :2], dtype=np.float32) for frame in valid_frames
    ], axis=0)
    confidence = np.stack([
        np.asarray(frame[:, 2:3], dtype=np.float32) for frame in valid_frames
    ], axis=0)
    min_confidence_by_joint = resolve_poseformer_confidence_thresholds(
        names,
        min_confidence=min_confidence,
        per_joint_min_confidence=per_joint_min_confidence,
    )
    valid_mask = np.isfinite(sequence).all(axis=-1)
    valid_mask &= np.isfinite(confidence).all(axis=-1)
    valid_mask &= (confidence[:, :, 0] >= min_confidence_by_joint[None, :])

    sequence = sequence.copy()
    sequence[~valid_mask] = np.nan

    return {
        "sequence": sequence,
        "confidence": confidence,
        "valid_mask": valid_mask,
        "num_frames": int(sequence.shape[0]),
        "num_joints": int(sequence.shape[1]),
        "min_confidence": float(min_confidence),
        "min_confidence_by_joint": min_confidence_by_joint,
        "ready": bool(valid_mask.all()),
    }


def fill_poseformer_gaps(sequence, valid_mask):
    """결측 2D 좌표를 최근접 유효값으로 메운다."""
    seq = np.asarray(sequence, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    if seq.ndim != 3 or seq.shape[-1] != 2:
        raise ValueError(f"sequence must have shape (T, J, 2), got {seq.shape}")
    if mask.shape != seq.shape[:2]:
        raise ValueError(f"valid_mask must have shape {seq.shape[:2]}, got {mask.shape}")

    num_frames, num_joints = seq.shape[:2]
    for joint_idx in range(num_joints):
        joint_mask = mask[:, joint_idx]
        if not joint_mask.any():
            seq[:, joint_idx, :] = 0.0
            continue

        valid_indices = np.flatnonzero(joint_mask)
        first_valid = int(valid_indices[0])
        last_valid = int(valid_indices[-1])

        seq[:first_valid, joint_idx, :] = seq[first_valid, joint_idx, :]
        seq[last_valid + 1:, joint_idx, :] = seq[last_valid, joint_idx, :]

        for start_idx, end_idx in zip(valid_indices[:-1], valid_indices[1:]):
            if end_idx - start_idx <= 1:
                continue
            seq[start_idx + 1:end_idx, joint_idx, :] = seq[start_idx, joint_idx, :]

        for frame_idx in range(num_frames):
            if not np.isfinite(seq[frame_idx, joint_idx]).all():
                seq[frame_idx, joint_idx, :] = seq[first_valid, joint_idx, :]

    return seq.astype(np.float32)


def normalize_screen_coordinates(sequence, width, height):
    """PoseFormerV2 관례에 맞게 픽셀 좌표를 정규화한다."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[-1] != 2:
        raise ValueError(f"sequence must have shape (T, J, 2), got {seq.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    normalized = seq.copy()
    normalized[..., 0] = normalized[..., 0] / float(width) * 2.0 - 1.0
    normalized[..., 1] = normalized[..., 1] / float(width) * 2.0 - float(height) / float(width)
    return normalized.astype(np.float32)


def build_centered_poseformer_windows(sequence, window_size=27):
    """각 프레임을 중심으로 하는 고정 길이 윈도우를 만든다."""
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[-1] != 2:
        raise ValueError(f"sequence must have shape (T, J, 2), got {seq.shape}")

    radius = window_size // 2
    padded = np.pad(seq, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    windows = [padded[idx:idx + window_size] for idx in range(seq.shape[0])]
    return np.stack(windows, axis=0).astype(np.float32)


def build_poseformer_windows(sequence, window_size=17, stride=1):
    """시간 윈도우 단위로 2D 시퀀스를 잘라 실험용 입력 배치를 만든다."""
    if sequence.ndim != 3 or sequence.shape[-1] != 2:
        raise ValueError(f"sequence must have shape (T, J, 2), got {sequence.shape}")

    T = sequence.shape[0]
    if T < window_size:
        return np.empty((0, window_size, sequence.shape[1], sequence.shape[2]), dtype=np.float32)

    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(sequence[start:start + window_size])
    return np.stack(windows, axis=0).astype(np.float32)


def collapse_poseformer_predictions(predictions):
    """PoseFormerV2 출력 shape을 (N, J, 3)으로 정규화한다."""
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"predictions must have shape (N, J, 3) or (N, 1, J, 3), got {arr.shape}")
    return arr.astype(np.float32)


def run_poseformer_inference(sequence, predictor, window_size=27):
    """정규화된 2D 시퀀스에서 프레임별 3D 추론을 실행한다."""
    windows = build_centered_poseformer_windows(sequence, window_size=window_size)
    predictions = predictor(windows)
    return collapse_poseformer_predictions(predictions)


def prepare_poseformer_inputs(
    kpts_seq,
    width,
    height,
    min_confidence=0.35,
    joint_names=None,
    per_joint_min_confidence=None,
):
    """2D 키포인트 시퀀스를 PoseFormerV2 추론 입력으로 준비한다."""
    prepared = prepare_poseformer_sequence(
        kpts_seq,
        min_confidence=min_confidence,
        joint_names=joint_names,
        per_joint_min_confidence=per_joint_min_confidence,
    )
    filled = fill_poseformer_gaps(prepared["sequence"], prepared["valid_mask"])
    normalized = normalize_screen_coordinates(filled, width=width, height=height)
    prepared["filled_sequence"] = filled
    prepared["normalized_sequence"] = normalized
    return prepared


def extract_poseformer_checkpoint_fingerprint(state_dict):
    """체크포인트 state_dict에서 모델 호환성 지문을 추출한다."""
    temporal = state_dict.get("Temporal_pos_embed")
    temporal_freq = state_dict.get("Temporal_pos_embed_")
    spatial = state_dict.get("Spatial_pos_embed")
    head_weight = state_dict.get("head.1.weight")
    if temporal is None or temporal_freq is None or spatial is None or head_weight is None:
        raise ValueError("checkpoint is missing required PoseFormerV2 keys")

    embed_dim = int(temporal.shape[-1])
    num_joints = int(spatial.shape[1])
    if embed_dim % max(1, num_joints) != 0:
        raise ValueError("checkpoint embed dimension is not divisible by num_joints")

    return {
        "num_kept_frames": int(temporal.shape[1]),
        "num_kept_coeffs": int(temporal_freq.shape[1]),
        "num_joints": num_joints,
        "embed_dim_ratio": int(embed_dim // num_joints),
        "out_dim": int(head_weight.shape[0]),
    }


def extract_poseformer_checkpoint_metadata(checkpoint):
    """체크포인트에 포함된 명시적 런타임 메타데이터를 읽는다."""
    metadata = checkpoint.get("poseformer_metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("poseformer_metadata must be a dictionary when present")
    return metadata


def validate_poseformer_checkpoint_metadata(metadata, *, num_frames, num_kept_frames, num_kept_coeffs, embed_dim_ratio, num_joints):
    """체크포인트 내부 메타데이터와 요청 설정이 일치하는지 검증한다."""
    expected = {
        "num_frames": int(num_frames),
        "num_kept_frames": int(num_kept_frames),
        "num_kept_coeffs": int(num_kept_coeffs),
        "embed_dim_ratio": int(embed_dim_ratio),
        "num_joints": int(num_joints),
    }
    mismatches = {
        key: (metadata.get(key), expected[key])
        for key in expected
        if key in metadata and metadata.get(key) != expected[key]
    }
    if mismatches:
        details = ", ".join(f"{key}={actual} expected={want}" for key, (actual, want) in mismatches.items())
        raise ValueError(f"PoseFormerV2 checkpoint metadata mismatch: {details}")
    return True


def validate_poseformer_checkpoint_fingerprint(fingerprint, *, num_kept_frames, num_kept_coeffs, embed_dim_ratio, num_joints):
    """체크포인트 구조와 요청 설정이 일치하는지 확인한다."""
    expected = {
        "num_kept_frames": int(num_kept_frames),
        "num_kept_coeffs": int(num_kept_coeffs),
        "embed_dim_ratio": int(embed_dim_ratio),
        "num_joints": int(num_joints),
        "out_dim": int(num_joints) * 3,
    }
    mismatches = {
        key: (fingerprint.get(key), expected[key])
        for key in expected
        if fingerprint.get(key) != expected[key]
    }
    if mismatches:
        details = ", ".join(f"{key}={actual} expected={want}" for key, (actual, want) in mismatches.items())
        raise ValueError(f"PoseFormerV2 checkpoint/config mismatch: {details}")
    return True


class PoseFormerV2Runner:
    """로컬 PoseFormerV2 코드와 체크포인트를 사용해 3D 추론을 수행한다."""

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        *,
        num_frames=27,
        num_kept_frames=3,
        num_kept_coeffs=3,
        embed_dim_ratio=32,
        depth=4,
        num_joints=17,
        num_heads=8,
        mlp_ratio=2.0,
        left_joints=None,
        right_joints=None,
        device=None,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"PoseFormerV2 repo not found: {self.repo_dir}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PoseFormerV2 checkpoint not found: {self.checkpoint_path}")
        if num_frames <= 0 or num_frames % 2 == 0:
            raise ValueError("num_frames must be a positive odd integer")

        self.num_frames = int(num_frames)
        contract = validate_poseformer_joint_contract(
            COCO_KEYPOINT_NAMES,
            left_joints=left_joints,
            right_joints=right_joints,
        )
        self.left_joints = contract["left_joints"]
        self.right_joints = contract["right_joints"]
        self.num_joints = int(num_joints)
        self.embed_dim_ratio = int(embed_dim_ratio)
        self.num_kept_frames = int(num_kept_frames)
        self.num_kept_coeffs = int(num_kept_coeffs)

        if self.num_frames < self.num_kept_frames:
            raise ValueError("num_frames must be greater than or equal to num_kept_frames")
        if self.num_kept_coeffs <= 0 or self.num_kept_frames <= 0:
            raise ValueError("num_kept_frames and num_kept_coeffs must be positive")

        torch = importlib.import_module("torch")
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        repo_str = os.fspath(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        model_module = importlib.import_module("common.model_poseformer")
        self._PoseTransformerV2 = model_module.PoseTransformerV2
        self._args = SimpleNamespace(
            embed_dim_ratio=self.embed_dim_ratio,
            depth=int(depth),
            number_of_kept_frames=self.num_kept_frames,
            number_of_kept_coeffs=self.num_kept_coeffs,
        )
        self._model = self._build_model(
            num_joints=num_joints,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def _build_model(self, *, num_joints, num_heads, mlp_ratio):
        model = self._PoseTransformerV2(
            num_frame=self.num_frames,
            num_joints=int(num_joints),
            in_chans=2,
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=True,
            qk_scale=None,
            drop_path_rate=0.0,
            args=self._args,
        )

        checkpoint = self._torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        metadata = extract_poseformer_checkpoint_metadata(checkpoint if isinstance(checkpoint, dict) else {})
        state_dict = checkpoint.get("model_pos", checkpoint.get("state_dict", checkpoint))
        cleaned_state = {}
        for key, value in state_dict.items():
            cleaned_key = key[7:] if key.startswith("module.") else key
            cleaned_state[cleaned_key] = value
        if metadata is not None:
            validate_poseformer_checkpoint_metadata(
                metadata,
                num_frames=self.num_frames,
                num_kept_frames=self.num_kept_frames,
                num_kept_coeffs=self.num_kept_coeffs,
                embed_dim_ratio=self.embed_dim_ratio,
                num_joints=self.num_joints,
            )
        validate_poseformer_checkpoint_fingerprint(
            extract_poseformer_checkpoint_fingerprint(cleaned_state),
            num_kept_frames=self.num_kept_frames,
            num_kept_coeffs=self.num_kept_coeffs,
            embed_dim_ratio=self.embed_dim_ratio,
            num_joints=self.num_joints,
        )
        model.load_state_dict(cleaned_state, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict_windows(self, windows, *, augment=True):
        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1] != self.num_frames or arr.shape[-1] != 2:
            raise ValueError(
                f"windows must have shape (N, {self.num_frames}, J, 2), got {arr.shape}"
            )

        tensor = self._torch.from_numpy(arr).to(self.device)
        with self._torch.no_grad():
            output = self._model(tensor)
            if augment:
                aug = tensor.clone()
                aug[:, :, :, 0] *= -1
                aug[:, :, self.left_joints + self.right_joints] = aug[:, :, self.right_joints + self.left_joints]
                output_flip = self._model(aug)
                output_flip[:, :, :, 0] *= -1
                output_flip[:, :, self.left_joints + self.right_joints, :] = output_flip[:, :, self.right_joints + self.left_joints, :]
                output = (output + output_flip) / 2.0
        return collapse_poseformer_predictions(output.detach().cpu().numpy())

    def predict_sequence(self, normalized_sequence, *, augment=True):
        return run_poseformer_inference(
            normalized_sequence,
            predictor=lambda windows: self.predict_windows(windows, augment=augment),
            window_size=self.num_frames,
        )

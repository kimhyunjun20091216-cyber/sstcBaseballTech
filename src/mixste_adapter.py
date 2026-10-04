from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from poseformer_adapter import (
    COCO_KEYPOINT_NAMES,
    collapse_poseformer_predictions,
    validate_poseformer_joint_contract,
)


class MixSTERunner:
    """Local MixSTE2 checkpoint runner with the same I/O contract as PoseFormerV2Runner."""

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        *,
        num_frames=27,
        embed_dim_ratio=32,
        depth=4,
        num_joints=17,
        num_heads=8,
        mlp_ratio=2.0,
        left_joints=None,
        right_joints=None,
        device=None,
        **_ignored_kwargs,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"MixSTE repo not found: {self.repo_dir}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MixSTE checkpoint not found: {self.checkpoint_path}")
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
        self.depth = int(depth)

        torch = importlib.import_module("torch")
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        repo_str = os.fspath(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        model_module = importlib.import_module("common.model_cross")
        self._MixSTE2 = model_module.MixSTE2
        self._args = SimpleNamespace(embed_dim_ratio=self.embed_dim_ratio, depth=int(depth))
        self._model = self._build_model(num_joints=num_joints, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def _build_model(self, *, num_joints, num_heads, mlp_ratio):
        model = self._MixSTE2(
            num_frame=self.num_frames,
            num_joints=int(num_joints),
            in_chans=2,
            embed_dim_ratio=self.embed_dim_ratio,
            depth=self.depth,
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=True,
            qk_scale=None,
            drop_path_rate=0.0,
        )

        checkpoint = self._torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_pos", checkpoint.get("state_dict", checkpoint))
        cleaned_state = {}
        for key, value in state_dict.items():
            cleaned_key = key[7:] if key.startswith("module.") else key
            cleaned_state[cleaned_key] = value
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
        from poseformer_adapter import run_poseformer_inference

        return run_poseformer_inference(
            normalized_sequence,
            predictor=lambda windows: self.predict_windows(windows, augment=augment),
            window_size=self.num_frames,
        )
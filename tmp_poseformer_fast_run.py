from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.analyze import (
    _enhance,
    _write_csv,
    assess_poseformer_readiness,
    default_poseformer_checkpoint_path,
    default_poseformer_repo_dir,
    evaluate_2d_stability,
    run_poseformer_3d_inference,
)
from src.pose_engine import build_engine
from src.smoothing import smooth_sequence


def main():
    root = Path(r"c:\Users\kimhy\OneDrive\문서\야구코드")
    video = root / "KakaoTalk_20260804_194608230.mp4"
    out_dir = root / "rec" / "poseformer_fast_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    keypoints_csv = out_dir / "keypoints.csv"
    pose3d_npy = out_dir / "pose3d.npy"
    pose3d_csv = out_dir / "pose3d.csv"

    payload = {
        "video": str(video),
        "repo": default_poseformer_repo_dir(),
        "checkpoint": None,
        "success": False,
    }

    readiness_thresholds = {
        "jitter_threshold": 24.0,
    }
    physics_cfg = {
        "enabled": True,
        "strength": 0.35,
        "iterations": 2,
    }
    human_shape_cfg = {
        "enabled": True,
        "strength": 0.85,
        "iterations": 3,
    }
    ik_cfg = {
        "enabled": True,
        "strength": 0.75,
        "iterations": 3,
    }
    rom_cfg = {
        "enabled": True,
        "strength": 0.55,
    }
    temporal_cfg = {
        "enabled": True,
        "alpha_slow": 0.30,
        "alpha_fast": 0.90,
        "velocity_scale": 0.10,
    }

    t0 = time.time()
    try:
        repo = default_poseformer_repo_dir()
        ckpt = default_poseformer_checkpoint_path(repo)
        payload["checkpoint"] = ckpt
        engine = build_engine("yolo")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_kpts = []
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        processed_frames = [_enhance(frame) for frame in frames]
        if hasattr(engine, "estimate_batch"):
            raw_kpts = engine.estimate_batch(processed_frames)
        else:
            raw_kpts = [engine.estimate(frame) for frame in processed_frames]
        engine.close()

        kpts_seq = smooth_sequence(raw_kpts, fps, keypoint_names=engine.keypoint_names)
        _write_csv(str(keypoints_csv), kpts_seq, engine.keypoint_names, fps)

        stability = evaluate_2d_stability(kpts_seq)
        readiness = assess_poseformer_readiness(kpts_seq, **readiness_thresholds)
        pose_result = run_poseformer_3d_inference(
            kpts_seq,
            frame_width=width,
            frame_height=height,
            repo_dir=repo,
            checkpoint_path=ckpt,
            fps=fps,
            output_path=str(pose3d_npy),
            csv_path=str(pose3d_csv),
            joint_names=engine.keypoint_names,
            runner_kwargs={
                "num_frames": 27,
                "num_kept_frames": 3,
                "num_kept_coeffs": 3,
            },
            readiness_thresholds=readiness_thresholds,
            physics_constraints=physics_cfg["enabled"],
            physics_strength=physics_cfg["strength"],
            physics_iterations=physics_cfg["iterations"],
            human_shape_constraints=human_shape_cfg["enabled"],
            human_shape_strength=human_shape_cfg["strength"],
            human_shape_iterations=human_shape_cfg["iterations"],
            ik_constraints=ik_cfg["enabled"],
            ik_strength=ik_cfg["strength"],
            ik_iterations=ik_cfg["iterations"],
            rom_constraints=rom_cfg["enabled"],
            rom_strength=rom_cfg["strength"],
            temporal_smoothing=temporal_cfg["enabled"],
            temporal_alpha_slow=temporal_cfg["alpha_slow"],
            temporal_alpha_fast=temporal_cfg["alpha_fast"],
            temporal_velocity_scale=temporal_cfg["velocity_scale"],
        )

        arr = pose_result["pose_3d"]
        payload.update({
            "success": True,
            "elapsed_s": round(time.time() - t0, 3),
            "fps": fps,
            "frame_width": width,
            "frame_height": height,
            "frames": len(kpts_seq),
            "detected_frames": int(sum(k is not None for k in raw_kpts)),
            "stability": stability,
            "readiness": readiness,
            "readiness_thresholds": readiness_thresholds,
            "poseformer_sanity": pose_result["sanity"],
            "physics": pose_result.get("physics", physics_cfg),
            "human_shape": pose_result.get("human_shape", human_shape_cfg),
            "ik": pose_result.get("ik", ik_cfg),
            "rom": pose_result.get("rom", rom_cfg),
            "temporal": pose_result.get("temporal", temporal_cfg),
            "quality": pose_result.get("quality"),
            "artifacts": {
                "keypoints_csv": str(keypoints_csv),
                "pose3d_npy": str(pose3d_npy),
                "pose3d_csv": str(pose3d_csv),
            },
            "pose3d_summary": {
                "shape": list(arr.shape),
                "depth_std": float(arr[..., 2].std()),
                "min_z": float(arr[..., 2].min()),
                "max_z": float(arr[..., 2].max()),
            },
        })
    except Exception as exc:  # noqa: BLE001
        payload.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": round(time.time() - t0, 3),
        })

    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(status_path)


if __name__ == "__main__":
    main()

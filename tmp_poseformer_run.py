from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path

import numpy as np

from src.analyze import (
    analyze_video,
    assess_poseformer_readiness,
    default_poseformer_checkpoint_path,
    default_poseformer_repo_dir,
)


COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_keypoint_csv(csv_path: Path):
    seq = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            arr = np.full((17, 3), np.nan, dtype=np.float32)
            any_valid = False
            for idx, name in enumerate(COCO_KEYPOINTS):
                x = row[f"{name}_x"]
                y = row[f"{name}_y"]
                v = row[f"{name}_v"]
                if x and y and v:
                    arr[idx] = [float(x), float(y), float(v)]
                    any_valid = True
            seq.append(arr if any_valid else None)
    return seq


def main():
    root = Path(r"c:\Users\kimhy\OneDrive\문서\야구코드")
    video = root / "KakaoTalk_20260802_102906528.mp4"
    out_dir = root / "rec" / "poseformer_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"

    overlay_path = out_dir / "overlay.webm"
    csv_path = out_dir / "keypoints.csv"
    pose3d_path = out_dir / "pose3d.npy"
    pose3d_csv_path = out_dir / "pose3d.csv"

    repo = default_poseformer_repo_dir()
    ckpt = default_poseformer_checkpoint_path(repo)
    print(json.dumps({"video": str(video), "repo": repo, "checkpoint": ckpt}, ensure_ascii=False))

    payload = {
        "video": str(video),
        "repo": repo,
        "checkpoint": ckpt,
        "success": False,
    }
    try:
        info = analyze_video(
            str(video),
            str(overlay_path),
            engine_name="yolo",
            csv_path=str(csv_path),
            smooth=True,
            preprocess=False,
            progress=False,
            poseformer_repo_dir=repo,
            poseformer_checkpoint_path=ckpt,
            poseformer_output_path=str(pose3d_path),
            poseformer_csv_path=str(pose3d_csv_path),
            poseformer_runner_kwargs={
                "num_frames": 27,
                "num_kept_frames": 3,
                "num_kept_coeffs": 3,
            },
        )

        readiness = assess_poseformer_readiness(load_keypoint_csv(csv_path))
        payload.update({
            "success": True,
            "info": info,
            "readiness": readiness,
            "artifacts": {
                "overlay": str(overlay_path),
                "keypoints_csv": str(csv_path),
                "pose3d_npy": str(pose3d_path),
                "pose3d_csv": str(pose3d_csv_path),
            },
        })
    except Exception as exc:  # noqa: BLE001
        payload.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(status_path)


if __name__ == "__main__":
    main()

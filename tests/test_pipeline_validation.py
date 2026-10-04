import cv2
import numpy as np
import pytest
import sys
import types

from src.analyze import (
    canonicalize_vertical_axis_3d,
    enforce_two_bone_ik_consistency_3d,
    enforce_bone_length_consistency_3d,
    enforce_human_vertical_hierarchy_3d,
    evaluate_input_joint_mapping_consistency_2d,
    evaluate_poseformer_human_shape_quality_3d,
    remap_pose3d_joint_layout_to_coco,
    resolve_pose3d_layout_to_coco,
    run_poseformer_3d_inference,
    smooth_pose3d_temporal_velocity_aware,
    validate_analysis_inputs,
    validate_poseformer_output,
)
from src.poseformer_adapter import COCO_KEYPOINT_NAMES


def test_validate_analysis_inputs_rejects_missing_video():
    with pytest.raises(FileNotFoundError, match="영상 파일이 없습니다"):
        validate_analysis_inputs("does_not_exist.mp4", engine_name="yolo")


def test_validate_analysis_inputs_rejects_missing_yolo_weights(tmp_path):
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16))
    try:
        assert writer.isOpened()
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        writer.write(frame)
    finally:
        writer.release()

    with pytest.raises(FileNotFoundError, match="YOLO pose weights"):
        validate_analysis_inputs(str(video_path), engine_name="yolo", weights=str(tmp_path / "missing_yolo.pt"))


def test_run_poseformer_3d_inference_writes_outputs(tmp_path):
    kpts_seq = []
    for frame_idx in range(3):
        frame = np.zeros((17, 3), dtype=np.float32)
        frame[:, 0] = 50 + frame_idx
        frame[:, 1] = 25 + frame_idx
        frame[:, 2] = 0.9
        kpts_seq.append(frame)

    class FakeRunner:
        def __init__(self, repo_dir, checkpoint_path, **kwargs):
            self.repo_dir = repo_dir
            self.checkpoint_path = checkpoint_path
            self.kwargs = kwargs

        def predict_sequence(self, normalized_sequence):
            assert normalized_sequence.shape == (3, 17, 2)
            out = np.ones((3, 17, 3), dtype=np.float32)
            out[..., 2] = np.linspace(0.0, 1.0, 3, dtype=np.float32)[:, None]
            return out

    output_path = tmp_path / "pose3d.npy"
    csv_path = tmp_path / "pose3d.csv"
    result = run_poseformer_3d_inference(
        kpts_seq,
        frame_width=100,
        frame_height=50,
        repo_dir=tmp_path / "repo",
        checkpoint_path=tmp_path / "model.bin",
        fps=30.0,
        output_path=str(output_path),
        csv_path=str(csv_path),
        runner_cls=FakeRunner,
        joint_names=COCO_KEYPOINT_NAMES,
    )

    assert result["pose_3d"].shape == (3, 17, 3)
    assert output_path.exists()
    assert csv_path.exists()

    saved = np.load(output_path)
    assert saved.shape == (3, 17, 3)


def test_run_poseformer_3d_inference_rejects_unstable_sequence(tmp_path):
    kpts_seq = []
    for frame_idx in range(5):
        frame = np.zeros((17, 3), dtype=np.float32)
        frame[:, 0] = 10 + frame_idx * 30
        frame[:, 1] = 20 + frame_idx * 30
        frame[:, 2] = 0.2
        kpts_seq.append(frame)

    class FakeRunner:
        def __init__(self, repo_dir, checkpoint_path, **kwargs):
            self.repo_dir = repo_dir
            self.checkpoint_path = checkpoint_path

        def predict_sequence(self, normalized_sequence):
            return np.ones((5, 17, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="input sequence is not ready"):
        run_poseformer_3d_inference(
            kpts_seq,
            frame_width=100,
            frame_height=50,
            repo_dir=tmp_path / "repo",
            checkpoint_path=tmp_path / "model.bin",
            fps=30.0,
            runner_cls=FakeRunner,
        )


def test_validate_poseformer_output_rejects_flat_depth():
    pose_3d = np.ones((3, 17, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="depth span is too small"):
        validate_poseformer_output(pose_3d)


def test_enforce_bone_length_consistency_3d_reduces_length_variation():
    pose_3d = np.zeros((3, 17, 3), dtype=np.float32)
    pose_3d[:, 5, :] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pose_3d[:, 7, :] = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    corrected = enforce_bone_length_consistency_3d(
        pose_3d,
        bones=[(5, 7)],
        strength=1.0,
        iterations=1,
    )

    before = np.linalg.norm(pose_3d[:, 7, :] - pose_3d[:, 5, :], axis=1)
    after = np.linalg.norm(corrected[:, 7, :] - corrected[:, 5, :], axis=1)
    assert np.var(after) < np.var(before)


def test_canonicalize_vertical_axis_3d_flips_when_ankles_are_above_hips():
    pose_3d = np.zeros((4, 17, 3), dtype=np.float32)
    pose_3d[:, 11, 1] = 0.3
    pose_3d[:, 12, 1] = 0.3
    pose_3d[:, 15, 1] = 1.2
    pose_3d[:, 16, 1] = 1.1

    corrected = canonicalize_vertical_axis_3d(pose_3d)
    hip_y = np.mean(corrected[:, [11, 12], 1], axis=1)
    ankle_y = np.mean(corrected[:, [15, 16], 1], axis=1)
    assert np.median(ankle_y - hip_y) < 0.0


def test_enforce_human_vertical_hierarchy_3d_orders_leg_chain_y():
    pose_3d = np.zeros((1, 17, 3), dtype=np.float32)
    # Intentionally inverted order: ankle > knee > hip > shoulder (wrong)
    pose_3d[0, 15, 1] = 2.0
    pose_3d[0, 13, 1] = 1.0
    pose_3d[0, 11, 1] = 0.0
    pose_3d[0, 5, 1] = -1.0

    corrected = enforce_human_vertical_hierarchy_3d(pose_3d, strength=1.0, iterations=1)
    ay = float(corrected[0, 15, 1])
    ky = float(corrected[0, 13, 1])
    hy = float(corrected[0, 11, 1])
    sy = float(corrected[0, 5, 1])
    assert ay <= ky <= hy <= sy


def test_enforce_two_bone_ik_consistency_3d_stabilizes_chain_lengths():
    pose_3d = np.zeros((3, 17, 3), dtype=np.float32)
    pose_3d[:, 5, :] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pose_3d[:, 7, :] = np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
    pose_3d[:, 9, :] = np.array([[1.5, 0.2, 0.0], [1.8, -0.2, 0.0], [1.1, 0.1, 0.0]], dtype=np.float32)

    corrected = enforce_two_bone_ik_consistency_3d(
        pose_3d,
        chains=[(5, 7, 9)],
        strength=1.0,
        iterations=2,
    )
    l1 = np.linalg.norm(corrected[:, 7, :] - corrected[:, 5, :], axis=1)
    l2 = np.linalg.norm(corrected[:, 9, :] - corrected[:, 7, :], axis=1)
    assert np.std(l1) < 0.15
    assert np.std(l2) < 0.15


def test_smooth_pose3d_temporal_velocity_aware_reduces_spike():
    pose_3d = np.zeros((4, 17, 3), dtype=np.float32)
    pose_3d[2, 9, 0] = 1.0
    smoothed = smooth_pose3d_temporal_velocity_aware(
        pose_3d,
        alpha_slow=0.2,
        alpha_fast=0.2,
        velocity_scale=1.0,
    )
    assert float(smoothed[2, 9, 0]) < 1.0


def test_evaluate_poseformer_human_shape_quality_3d_returns_metrics():
    pose_3d = np.zeros((2, 17, 3), dtype=np.float32)
    pose_3d[:, 5, 1] = 0.4
    pose_3d[:, 6, 1] = 0.4
    pose_3d[:, 11, 1] = 0.1
    pose_3d[:, 12, 1] = 0.1
    pose_3d[:, 13, 1] = 0.0
    pose_3d[:, 14, 1] = 0.0
    pose_3d[:, 15, 1] = -0.2
    pose_3d[:, 16, 1] = -0.2
    pose_3d[:, 5, 0] = -0.1
    pose_3d[:, 6, 0] = 0.1
    pose_3d[:, 11, 0] = -0.08
    pose_3d[:, 12, 0] = 0.08
    pose_3d[:, 7, 0] = -0.15
    pose_3d[:, 8, 0] = 0.15
    pose_3d[:, 9, 0] = -0.22
    pose_3d[:, 10, 0] = 0.22

    quality = evaluate_poseformer_human_shape_quality_3d(pose_3d)
    assert quality["ankle_below_hip_ratio"] >= 0.99
    assert "pass_all" in quality


def test_evaluate_input_joint_mapping_consistency_2d_detects_consistency():
    kpts_seq = []
    for _ in range(3):
        f = np.zeros((17, 3), dtype=np.float32)
        f[:, 2] = 0.9
        f[5, 0] = -1.0
        f[6, 0] = 1.0
        f[11, 0] = -0.8
        f[12, 0] = 0.8
        f[7, :2] = (-1.5, 0.0)
        f[8, :2] = (1.5, 0.0)
        f[9, :2] = (-2.0, 0.0)
        f[10, :2] = (2.0, 0.0)
        kpts_seq.append(f)

    diag = evaluate_input_joint_mapping_consistency_2d(kpts_seq)
    assert diag["valid"] is True
    assert diag["left_right_consistent"] is True


def test_run_poseformer_3d_inference_can_disable_physics_constraints(tmp_path):
    kpts_seq = []
    for frame_idx in range(3):
        frame = np.zeros((17, 3), dtype=np.float32)
        frame[:, 0] = 50 + frame_idx
        frame[:, 1] = 25 + frame_idx
        frame[:, 2] = 0.9
        kpts_seq.append(frame)

    class FakeRunner:
        def __init__(self, repo_dir, checkpoint_path, **kwargs):
            self.repo_dir = repo_dir
            self.checkpoint_path = checkpoint_path
            self.kwargs = kwargs

        def predict_sequence(self, normalized_sequence):
            out = np.ones((3, 17, 3), dtype=np.float32)
            out[..., 2] = np.linspace(0.0, 1.0, 3, dtype=np.float32)[:, None]
            return out

    result = run_poseformer_3d_inference(
        kpts_seq,
        frame_width=100,
        frame_height=50,
        repo_dir=tmp_path / "repo",
        checkpoint_path=tmp_path / "model.bin",
        fps=30.0,
        runner_cls=FakeRunner,
        joint_names=COCO_KEYPOINT_NAMES,
        physics_constraints=False,
        human_shape_constraints=False,
    )

    assert result["pose_3d"].shape == (3, 17, 3)
    assert result["human_shape"]["enabled"] is False


def test_run_poseformer_3d_inference_routes_to_mixste_backend(monkeypatch, tmp_path):
    kpts_seq = []
    for frame_idx in range(3):
        frame = np.zeros((17, 3), dtype=np.float32)
        frame[:, 0] = 50 + frame_idx
        frame[:, 1] = 25 + frame_idx
        frame[:, 2] = 0.9
        kpts_seq.append(frame)

    class FakePoseFormerRunner:
        def __init__(self, repo_dir, checkpoint_path, **kwargs):
            self.repo_dir = repo_dir
            self.checkpoint_path = checkpoint_path
            self.kwargs = kwargs

        def predict_sequence(self, normalized_sequence):
            out = np.zeros((3, 17, 3), dtype=np.float32)
            out[..., 2] = np.array([0.0, 0.5, 1.0], dtype=np.float32)[:, None]
            return out

    class FakeMixSTERunner:
        def __init__(self, repo_dir, checkpoint_path, **kwargs):
            self.repo_dir = repo_dir
            self.checkpoint_path = checkpoint_path
            self.kwargs = kwargs
            assert "num_kept_frames" in kwargs
            assert "num_kept_coeffs" in kwargs

        def predict_sequence(self, normalized_sequence):
            out = np.zeros((3, 17, 3), dtype=np.float32)
            out[..., 0] = 1.0
            out[..., 2] = np.array([0.0, 0.5, 1.0], dtype=np.float32)[:, None]
            return out

    fake_poseformer_module = types.SimpleNamespace(
        COCO_KEYPOINT_NAMES=COCO_KEYPOINT_NAMES,
        PoseFormerV2Runner=FakePoseFormerRunner,
        prepare_poseformer_inputs=lambda *args, **kwargs: {
            "normalized_sequence": np.zeros((3, 17, 2), dtype=np.float32)
        },
        validate_poseformer_joint_contract=lambda names, **kwargs: {
            "joint_names": list(names),
            "left_joints": [1, 3, 5, 7, 9, 11, 13, 15],
            "right_joints": [2, 4, 6, 8, 10, 12, 14, 16],
        },
    )
    fake_mixste_module = types.SimpleNamespace(MixSTERunner=FakeMixSTERunner)
    monkeypatch.setitem(sys.modules, "poseformer_adapter", fake_poseformer_module)
    monkeypatch.setitem(sys.modules, "mixste_adapter", fake_mixste_module)

    result = run_poseformer_3d_inference(
        kpts_seq,
        frame_width=100,
        frame_height=50,
        repo_dir=tmp_path / "repo",
        checkpoint_path=tmp_path / "model.bin",
        fps=30.0,
        runner_kwargs={"num_frames": 3, "num_kept_frames": 3, "num_kept_coeffs": 3},
        pose3d_model_name="mixste",
        joint_names=COCO_KEYPOINT_NAMES,
    )

    assert result["pose3d_model_name"] == "mixste"
    assert result["pose_3d"].shape == (3, 17, 3)
    assert result["pose_3d"][..., 2].max() > result["pose_3d"][..., 2].min()


def test_remap_pose3d_joint_layout_to_coco_h36m_maps_core_joints():
    pose_h36m = np.zeros((1, 17, 3), dtype=np.float32)
    # Unique x values per H36M joint index for deterministic mapping checks.
    for j in range(17):
        pose_h36m[0, j, 0] = float(j)

    pose_coco = remap_pose3d_joint_layout_to_coco(pose_h36m, source_layout="h36m")
    assert pose_coco.shape == (1, 17, 3)
    # left/right shoulders
    assert float(pose_coco[0, 5, 0]) == 11.0
    assert float(pose_coco[0, 6, 0]) == 14.0
    # left/right hips
    assert float(pose_coco[0, 11, 0]) == 4.0
    assert float(pose_coco[0, 12, 0]) == 1.0
    # left/right ankles
    assert float(pose_coco[0, 15, 0]) == 6.0
    assert float(pose_coco[0, 16, 0]) == 3.0


def test_resolve_pose3d_layout_to_coco_auto_prefers_h36m_when_more_plausible():
    pose_h36m = np.zeros((2, 17, 3), dtype=np.float32)
    # Construct a plausible upright body in H36M order.
    # hips
    pose_h36m[:, 1, :] = np.array([0.10, 0.10, 0.00], dtype=np.float32)  # rhip
    pose_h36m[:, 4, :] = np.array([-0.10, 0.10, 0.00], dtype=np.float32)  # lhip
    # knees
    pose_h36m[:, 2, :] = np.array([0.10, 0.00, 0.00], dtype=np.float32)
    pose_h36m[:, 5, :] = np.array([-0.10, 0.00, 0.00], dtype=np.float32)
    # ankles
    pose_h36m[:, 3, :] = np.array([0.10, -0.20, 0.00], dtype=np.float32)
    pose_h36m[:, 6, :] = np.array([-0.10, -0.20, 0.00], dtype=np.float32)
    # shoulders
    pose_h36m[:, 14, :] = np.array([0.16, 0.35, 0.00], dtype=np.float32)  # rshoulder
    pose_h36m[:, 11, :] = np.array([-0.16, 0.35, 0.00], dtype=np.float32)  # lshoulder
    # elbows/wrists
    pose_h36m[:, 15, :] = np.array([0.24, 0.25, 0.00], dtype=np.float32)
    pose_h36m[:, 12, :] = np.array([-0.24, 0.25, 0.00], dtype=np.float32)
    pose_h36m[:, 16, :] = np.array([0.30, 0.15, 0.00], dtype=np.float32)
    pose_h36m[:, 13, :] = np.array([-0.30, 0.15, 0.00], dtype=np.float32)
    # nose/neck/head proxy
    pose_h36m[:, 9, :] = np.array([0.00, 0.46, 0.00], dtype=np.float32)
    pose_h36m[:, 10, :] = np.array([0.00, 0.55, 0.00], dtype=np.float32)
    pose_h36m[:, 0, :] = np.array([0.00, 0.12, 0.00], dtype=np.float32)  # pelvis
    pose_h36m[:, 7, :] = np.array([0.00, 0.22, 0.00], dtype=np.float32)  # spine
    pose_h36m[:, 8, :] = np.array([0.00, 0.32, 0.00], dtype=np.float32)  # thorax

    resolved, info = resolve_pose3d_layout_to_coco(pose_h36m, source_layout="auto")
    assert resolved.shape == (2, 17, 3)
    assert info["selected_layout"] == "h36m"

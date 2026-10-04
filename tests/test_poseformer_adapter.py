import numpy as np

from src.poseformer_adapter import (
    COCO_KEYPOINT_NAMES,
    COCO_LEFT_JOINTS,
    build_centered_poseformer_windows,
    collapse_poseformer_predictions,
    extract_poseformer_checkpoint_fingerprint,
    fill_poseformer_gaps,
    normalize_screen_coordinates,
    prepare_poseformer_inputs,
    prepare_poseformer_sequence,
    resolve_poseformer_confidence_thresholds,
    run_poseformer_inference,
    validate_poseformer_checkpoint_fingerprint,
    validate_poseformer_joint_contract,
)


def test_prepare_poseformer_sequence_exports_stable_xy_and_confidence():
    seq = []
    for i in range(4):
        arr = np.zeros((17, 3), dtype=np.float32)
        arr[:, 0] = 100 + i
        arr[:, 1] = 200 + i * 0.5
        arr[:, 2] = 0.9
        seq.append(arr)

    prepared = prepare_poseformer_sequence(seq)

    assert prepared["sequence"].shape == (4, 17, 2)
    assert prepared["confidence"].shape == (4, 17, 1)
    assert prepared["valid_mask"].shape == (4, 17)
    assert prepared["num_frames"] == 4
    assert prepared["num_joints"] == 17


def test_prepare_poseformer_sequence_marks_low_confidence_as_invalid():
    frame = np.zeros((17, 3), dtype=np.float32)
    frame[:, 0] = 10.0
    frame[:, 1] = 20.0
    frame[:, 2] = 0.2

    prepared = prepare_poseformer_sequence([frame], min_confidence=0.35)

    assert prepared["valid_mask"].sum() == 0
    assert np.isnan(prepared["sequence"]).all()


def test_prepare_poseformer_sequence_uses_stricter_wrist_thresholds():
    frame = np.zeros((17, 3), dtype=np.float32)
    frame[:, 0] = 10.0
    frame[:, 1] = 20.0
    frame[:, 2] = 0.5
    frame[9, 2] = 0.55

    prepared = prepare_poseformer_sequence([frame], min_confidence=0.35)

    assert prepared["valid_mask"][0, 5]
    assert not prepared["valid_mask"][0, 9]
    assert np.isnan(prepared["sequence"][0, 9]).all()


def test_fill_poseformer_gaps_uses_nearest_valid_values():
    sequence = np.array([
        [[1.0, 2.0], [np.nan, np.nan]],
        [[np.nan, np.nan], [5.0, 6.0]],
        [[3.0, 4.0], [np.nan, np.nan]],
    ], dtype=np.float32)
    valid_mask = np.array([
        [True, False],
        [False, True],
        [True, False],
    ])

    filled = fill_poseformer_gaps(sequence, valid_mask)

    np.testing.assert_allclose(filled[1, 0], [1.0, 2.0])
    np.testing.assert_allclose(filled[:, 1], [[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]])


def test_normalize_screen_coordinates_matches_poseformer_convention():
    sequence = np.array([[[50.0, 25.0], [100.0, 50.0]]], dtype=np.float32)

    normalized = normalize_screen_coordinates(sequence, width=100, height=50)

    np.testing.assert_allclose(normalized[0, 0], [0.0, 0.0])
    np.testing.assert_allclose(normalized[0, 1], [1.0, 0.5])


def test_build_centered_poseformer_windows_pads_sequence_edges():
    sequence = np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2)

    windows = build_centered_poseformer_windows(sequence, window_size=3)

    assert windows.shape == (5, 3, 2, 2)
    np.testing.assert_array_equal(windows[0, 0], sequence[0])
    np.testing.assert_array_equal(windows[0, 1], sequence[0])
    np.testing.assert_array_equal(windows[0, 2], sequence[1])
    np.testing.assert_array_equal(windows[-1, 0], sequence[-2])
    np.testing.assert_array_equal(windows[-1, 1], sequence[-1])
    np.testing.assert_array_equal(windows[-1, 2], sequence[-1])


def test_run_poseformer_inference_collapses_single_frame_predictions():
    sequence = np.zeros((4, 17, 2), dtype=np.float32)

    def predictor(windows):
        assert windows.shape == (4, 3, 17, 2)
        return np.zeros((4, 1, 17, 3), dtype=np.float32)

    predictions = run_poseformer_inference(sequence, predictor, window_size=3)

    assert predictions.shape == (4, 17, 3)


def test_prepare_poseformer_inputs_fills_missing_and_normalizes():
    seq = []
    first = np.zeros((17, 3), dtype=np.float32)
    first[:, 0] = 50.0
    first[:, 1] = 25.0
    first[:, 2] = 0.9
    seq.append(first)
    seq.append(None)

    prepared = prepare_poseformer_inputs(seq, width=100, height=50)

    assert prepared["normalized_sequence"].shape == (2, 17, 2)
    np.testing.assert_allclose(prepared["filled_sequence"][1], prepared["filled_sequence"][0])
    np.testing.assert_allclose(prepared["normalized_sequence"][0, 0], [0.0, 0.0])


def test_validate_poseformer_joint_contract_rejects_non_coco_order():
    bad_names = COCO_KEYPOINT_NAMES.copy()
    bad_names[0], bad_names[1] = bad_names[1], bad_names[0]

    try:
        validate_poseformer_joint_contract(bad_names)
    except ValueError as exc:
        assert "COCO_KEYPOINT_NAMES" in str(exc)
    else:
        raise AssertionError("validate_poseformer_joint_contract should reject reordered joints")


def test_validate_poseformer_checkpoint_fingerprint_rejects_mismatch():
    state_dict = {
        "Temporal_pos_embed": np.zeros((1, 3, 544), dtype=np.float32),
        "Temporal_pos_embed_": np.zeros((1, 3, 544), dtype=np.float32),
        "Spatial_pos_embed": np.zeros((1, 17, 32), dtype=np.float32),
        "head.1.weight": np.zeros((51, 1088), dtype=np.float32),
    }
    fingerprint = extract_poseformer_checkpoint_fingerprint(state_dict)

    try:
        validate_poseformer_checkpoint_fingerprint(
            fingerprint,
            num_kept_frames=9,
            num_kept_coeffs=3,
            embed_dim_ratio=32,
            num_joints=17,
        )
    except ValueError as exc:
        assert "checkpoint/config mismatch" in str(exc)
    else:
        raise AssertionError("validate_poseformer_checkpoint_fingerprint should reject mismatched config")


def test_resolve_poseformer_confidence_thresholds_applies_overrides():
    thresholds = resolve_poseformer_confidence_thresholds(
        COCO_KEYPOINT_NAMES,
        min_confidence=0.35,
        per_joint_min_confidence={"left_wrist": 0.8},
    )

    assert thresholds[9] == np.float32(0.8)
    assert thresholds[COCO_LEFT_JOINTS[0]] >= np.float32(0.35)


def test_collapse_poseformer_predictions_rejects_invalid_shape():
    bad = np.zeros((2, 17, 2), dtype=np.float32)

    try:
        collapse_poseformer_predictions(bad)
    except ValueError as exc:
        assert "predictions must have shape" in str(exc)
    else:
        raise AssertionError("collapse_poseformer_predictions should reject invalid shape")

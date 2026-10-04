import numpy as np

from src.smoothing import smooth_sequence


def test_smooth_sequence_holds_low_confidence_critical_joint_position():
    frame_a = np.zeros((17, 3), dtype=np.float32)
    frame_a[:, 0] = 100.0
    frame_a[:, 1] = 200.0
    frame_a[:, 2] = 0.9

    frame_b = frame_a.copy()
    frame_b[9, 0] = 400.0
    frame_b[9, 1] = 500.0
    frame_b[9, 2] = 0.2

    smoothed = smooth_sequence([frame_a, frame_b], fps=30.0)

    np.testing.assert_allclose(smoothed[1][9, :2], smoothed[0][9, :2])
    assert smoothed[1][9, 2] == np.float32(0.2)


def test_smooth_sequence_allows_noncritical_joint_to_update():
    frame_a = np.zeros((17, 3), dtype=np.float32)
    frame_a[:, 0] = 50.0
    frame_a[:, 1] = 75.0
    frame_a[:, 2] = 0.9

    frame_b = frame_a.copy()
    frame_b[0, 0] = 80.0
    frame_b[0, 1] = 120.0
    frame_b[0, 2] = 0.2

    smoothed = smooth_sequence([frame_a, frame_b], fps=30.0)

    assert smoothed[1][0, 0] != smoothed[0][0, 0]
    assert smoothed[1][0, 1] != smoothed[0][0, 1]

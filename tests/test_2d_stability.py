import numpy as np

from src.analyze import assess_poseformer_readiness, evaluate_2d_stability


def test_evaluate_2d_stability_reports_detection_and_jitter():
    frames = []
    for i in range(5):
        arr = np.zeros((17, 3), dtype=np.float32)
        arr[:, 0] = 100 + i
        arr[:, 1] = 200 + i * 0.5
        arr[:, 2] = 0.9
        frames.append(arr)

    result = evaluate_2d_stability(frames)

    assert result["detection_rate"] == 1.0
    assert result["mean_joint_confidence"] > 0.8
    assert result["temporal_jitter_px"] >= 0.0


def test_assess_poseformer_readiness_requires_stable_sequence():
    good = []
    for i in range(5):
        arr = np.zeros((17, 3), dtype=np.float32)
        arr[:, 0] = 100 + i * 0.2
        arr[:, 1] = 200 + i * 0.3
        arr[:, 2] = 0.9
        good.append(arr)

    bad = []
    for i in range(5):
        arr = np.zeros((17, 3), dtype=np.float32)
        arr[:, 0] = 100 + i * 20.0
        arr[:, 1] = 200 + i * 20.0
        arr[:, 2] = 0.2 if i % 2 == 0 else 0.3
        if i == 3:
            bad.append(None)
            continue
        bad.append(arr)

    good_result = assess_poseformer_readiness(good)
    bad_result = assess_poseformer_readiness(bad)

    assert good_result["ready"] is True
    assert bad_result["ready"] is False
    assert bad_result["issues"]

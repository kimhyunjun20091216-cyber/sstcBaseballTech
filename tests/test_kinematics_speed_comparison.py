import numpy as np

from src.kinematics import summarize_reference_comparison, summarize_speed_comparison


def test_summarize_speed_comparison_orders_peaks_and_release_lag():
    metrics = {
        "pelvis_speed": np.array([100.0, 200.0, 150.0]),
        "trunk_speed": np.array([80.0, 220.0, 180.0]),
        "arm_speed": np.array([90.0, 130.0, 240.0]),
    }
    timing = {
        "pelvis_peak": 1,
        "trunk_peak": 1,
        "arm_peak": 2,
        "release": 2,
    }

    summary = summarize_speed_comparison(metrics, timing, fps=30)

    assert summary["peak_sequence"] == ["pelvis", "trunk", "arm"]
    assert summary["pelvis_peak_s"] == 1 / 30
    assert summary["arm_peak_s"] == 2 / 30
    assert summary["release_lag_s"]["arm"] == 0.0
    assert "골반" in summary["summary_text"]


def test_summarize_reference_comparison_reports_gaps_and_priorities():
    user_result = {
        "metrics": {
            "pelvis_speed": np.array([10.0, 20.0, 30.0, 15.0]),
            "trunk_speed": np.array([8.0, 18.0, 25.0, 10.0]),
            "arm_speed": np.array([5.0, 10.0, 15.0, 8.0]),
        },
        "timing": {
            "leg_lift_peak": 0,
            "pelvis_peak": 2,
            "trunk_peak": 2,
            "arm_peak": 2,
            "release": 3,
        },
        "fps": 30.0,
    }
    ref_result = {
        "metrics": {
            "pelvis_speed": np.array([15.0, 25.0, 45.0, 20.0]),
            "trunk_speed": np.array([10.0, 24.0, 35.0, 12.0]),
            "arm_speed": np.array([8.0, 14.0, 22.0, 9.0]),
        },
        "timing": {
            "leg_lift_peak": 0,
            "pelvis_peak": 2,
            "trunk_peak": 2,
            "arm_peak": 2,
            "release": 3,
        },
        "fps": 30.0,
    }

    comparison = summarize_reference_comparison(user_result, ref_result)

    assert comparison["sections"]["pelvis"]["peak_gap"] > 0
    assert comparison["sections"]["arm"]["peak_ratio_pct"] < 100.0
    assert len(comparison["priorities"]) == 3
    assert comparison["insights"]

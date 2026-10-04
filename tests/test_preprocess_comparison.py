import cv2
import numpy as np

import src.analyze as analyze_mod


def test_compare_preprocessing_modes_detects_clahe_gain(tmp_path, monkeypatch):
    video_path = tmp_path / "dark.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
    try:
        for _ in range(2):
            dark = np.full((16, 16, 3), 5, dtype=np.uint8)
            writer.write(dark)
    finally:
        writer.release()

    class FakeEngine:
        def estimate(self, frame):
            if frame.mean() > 25:
                return np.ones((17, 3), dtype=np.float32)
            return None

        def close(self):
            pass

    monkeypatch.setattr(analyze_mod, "build_engine", lambda name, **kwargs: FakeEngine())

    result = analyze_mod.compare_preprocessing_modes(str(video_path), engine_name="yolo", sample_limit=2)

    assert result["raw"]["detection_rate"] == 0.0
    assert result["clahe"]["detection_rate"] > 0.0
    assert result["clahe"]["detection_rate"] > result["raw"]["detection_rate"]

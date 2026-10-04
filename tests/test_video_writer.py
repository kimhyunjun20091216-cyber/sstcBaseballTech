import os
import tempfile

import cv2
import numpy as np

from src.analyze import _write_video_with_fallback


def test_write_video_with_fallback_creates_browser_compatible_mp4(tmp_path):
    out_path = tmp_path / "clip.mp4"
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    produced_path = _write_video_with_fallback(
        out_path,
        [frame, frame],
        fps=10.0,
        width=32,
        height=32,
    )

    assert produced_path is not None
    assert os.path.exists(produced_path)
    assert os.path.getsize(produced_path) > 0

    cap = cv2.VideoCapture(produced_path)
    try:
        assert cap.isOpened()
        ret, read_frame = cap.read()
        assert ret is True
        assert read_frame.shape[0] == 32
        assert read_frame.shape[1] == 32
    finally:
        cap.release()

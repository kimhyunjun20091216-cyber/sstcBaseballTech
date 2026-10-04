from __future__ import annotations

from pathlib import Path
import cv2

src = Path(r"c:\Users\kimhy\OneDrive\문서\야구코드\KakaoTalk_20260802_102906528.mp4")
out = Path(r"c:\Users\kimhy\OneDrive\문서\야구코드\rec\poseformer_fast_run\input_clip.avi")
out.parent.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(src))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start = 10
end = 49
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
idx = 0
written = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if start <= idx <= end:
        writer.write(frame)
        written += 1
    idx += 1
cap.release()
writer.release()
print(out)
print(written)

#!/usr/bin/env bash
# 두 각도 투구 영상에 스켈레톤("졸라맨") 오버레이를 일괄 생성한다.
# 사용: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "[1/2] 측면 영상 (주간)"
$PY src/analyze.py "rec/KakaoTalk_Video_2026-06-23-06-06-01.mp4" \
    -o "out/영상1_측면_pose.mp4" --csv "out/영상1_측면_keypoints.csv"

echo "[2/2] 정면 영상 (역광) — CLAHE 보정 적용"
$PY src/analyze.py "rec/KakaoTalk_Video_2026-06-23-06-07-06.mp4" \
    -o "out/영상2_정면_pose.mp4" --csv "out/영상2_정면_keypoints.csv" --preprocess

echo "[운동학] 관절각·단계분할·리포트"
$PY src/kinematics.py "out/영상1_측면_keypoints.csv"
$PY src/kinematics.py "out/영상2_정면_keypoints.csv"

echo "완료. 결과: out/ (오버레이 mp4, keypoints/angles CSV, kinematics.png, report.md)"

# 투구 동작 분석 — 컴퓨터비전 (Phase 1)

야구 투구 영상에 관절 스켈레톤("졸라맨")을 오버레이하고 관절 좌표를 추출하는
파이프라인. 전체 연구 계획은 [`개발계획서.md`](개발계획서.md) 참고.

## 구성

```
src/
  analyze.py      # 메인 파이프라인 (영상→자세추정→스무딩→오버레이)
  pose_engine.py  # 자세추정 엔진 (MediaPipe / YOLO 스위치)
  smoothing.py    # One-Euro 필터 (키포인트 지터 제거)
  kinematics.py   # 관절각 계산·투구 단계 분할·리포트 (Phase 3)
  validation.py   # 측정 검증 통계 (Bland-Altman/ICC/RMSE, Phase 4)
  compare_engines.py  # MediaPipe vs YOLO 정량 비교
  app.py          # Streamlit 데모 UI
  make_paper.py   # 논문.md → docx 변환
  make_slides.py  # 발표자료 pptx 생성
models/           # pose_landmarker_heavy.task (자동 다운로드 필요)
rec/              # 입력 영상
out/              # 출력: *_pose.mp4, *_keypoints.csv, 리포트·검증·비교
paper/            # 논문(md/docx), 발표자료(pptx), figs/
```

## 논문·발표자료 (KSEF/ISEF)

```bash
.venv/bin/pip install python-docx python-pptx   # 최초 1회
.venv/bin/python src/make_paper.py paper/논문.md -o paper/투구분석_논문.docx
.venv/bin/python src/make_slides.py -o paper/투구분석_발표.pptx
```

- `paper/논문.md` — 한국어 IMRaD 정식 논문(초록·서론·관련연구·방법·결과·고찰·결론·참고문헌 17편)
- `paper/투구분석_논문.docx` — 제출용 문서(한글 폰트, 표 2·그림 3 임베드)
- `paper/투구분석_발표.pptx` — 발표 슬라이드 15장(결과 이미지 포함)
- ⚠ 제출 전 인용 확인: PitcherNet 저자는 **Bright et al.**(Nekoui 아님),
  Aguinaldo/Nebel/Diffendaffer 일부 DOI·권호는 원문 재확인 권장.

## 설치

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

설치 후 아래 **모델 가중치 다운로드**를 반드시 수행한다.

## 모델 가중치 다운로드

자세추정 모델 가중치는 용량이 커 저장소에 포함되지 않는다(`.gitignore` 처리).
클론 후 아래 절차로 내려받아야 파이프라인이 동작한다.

### 1. MediaPipe Pose (필수, 약 30MB)

`mediapipe` 엔진(기본값)에 필요하다. `models/` 아래에 저장한다.

```bash
mkdir -p models
curl -sL -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

- 저장 경로가 `models/pose_landmarker_heavy.task`인지 확인(엔진 기본 경로).
- 더 가벼운 대안: URL의 `heavy`를 `full` 또는 `lite`로 교체(정확도↓, 속도↑).

### 2. YOLO11-pose (선택, 약 113MB)

`--engine yolo` 사용 시에만 필요하다. **최초 실행 시 Ultralytics가 자동으로
내려받으므로 수동 다운로드는 불필요**하다. 오프라인 등으로 수동 배치하려면:

```bash
curl -sL -o yolo11x-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
```

- 더 가벼운 대안: `yolo11m-pose.pt` / `yolo11s-pose.pt`(정확도↓, 속도↑).
  이 경우 `src/pose_engine.py`의 `YoloEngine(weights=...)` 기본값을 맞춘다.

### 확인

```bash
ls -lh models/pose_landmarker_heavy.task   # 약 30MB면 정상
```

## 실행

```bash
# 두 영상 일괄 처리
bash run_all.sh

# 개별 실행
.venv/bin/python src/analyze.py 입력.mp4 -o 출력.mp4 --csv 좌표.csv
```

주요 옵션:
- `--engine {mediapipe,yolo}` — 자세추정 엔진 (기본 mediapipe)
- `--preprocess` — 역광/저조도 CLAHE 보정 (정면 영상 권장)
- `--no-smooth` — 스무딩 비활성 (원시 검출 확인용)

## 현재 결과 (Phase 1 완료)

| 영상 | 각도 | 조건 | 검출률 |
|---|---|---|---|
| 영상1 | 측/후면 | 주간·순광 | 122/122 (100%) |
| 영상2 | 정면 | 석양·역광 | 197/197 (100%) |

- 스켈레톤: COCO 17관절, 좌(주황)/우(하늘) 색 구분.
- 출력 CSV: 프레임별 관절 x,y,visibility → Phase 3(운동학 변수) 입력.

## 운동학 분석 (Phase 3)

```bash
.venv/bin/python src/kinematics.py out/영상1_측면_keypoints.csv
```

출력: `*_angles.csv`(프레임별 관절각), `*_kinematics.png`(시계열 그래프+단계 마커),
`*_report.md`(핵심 지표 vs 정상값). 계산 항목:
- 팔꿈치·무릎 **굴곡각**(flexion = 180−내각), 몸통 기울기, 팔 슬롯
- 투구 단계 자동 검출: 다리들기 최고점 · 앞발 착지 · 릴리스(손목 최고속)
- 정상값 대조(Fleisig 1995 / Diffendaffer 2022)

## 데모 UI (Streamlit)

브라우저에서 영상 업로드 → 오버레이 영상 + 운동학 그래프 + 리포트를 한 번에 확인.

```bash
.venv/bin/pip install streamlit          # 최초 1회
.venv/bin/streamlit run src/app.py
```

- 사이드바: 엔진(mediapipe/yolo), 역광 보정(CLAHE), 던지는 팔(auto/L/R)
- "분석 시작" → 자세추정 오버레이(브라우저용 H.264 자동 인코딩) + Phase 3 리포트
- 오버레이 mp4 · keypoints.csv · angles.csv 다운로드 제공
- KSEF 시연용. 기존 `analyze.py` / `kinematics.py`를 재사용(로직 재구현 없음).

## 한계 / 향후 과제

- **30fps**: 릴리스 순간 과소표집 → 정식 촬영은 **240fps** 권장 (계획서 §1.2).
- **단일 2D**: 횡단면 어깨 회전 부정확 → 정량 분석엔 다중카메라 3D 필요.
- 각도는 영상 평면 투영값 → **실제 참값 주석 대조(Phase 4)** 전엔 상대·추세 비교용.
- 향후: 실측 참값 주석 수집·검증, 240fps·2카메라 정식 촬영, GRF(포스플레이트) 융합.

## 검증 (Phase 4)

CV 측정 각도를 **참값(수동 주석)** 과 대조해 일치도·신뢰도를 정량 평가한다
(계획서 §3.2). 변수별로 Bland-Altman(편향+95% LoA), ICC(2,1)[95% CI],
RMSE, MAE, Pearson r 을 산출하고 마크다운 리포트 + Bland-Altman PNG를 만든다.

```bash
# 데모(합성 참값으로 파이프라인 동작 증명 — 실제 검증 아님)
.venv/bin/python src/validation.py --demo        # → out/validation_demo/

# 실제 검증(참값 주석 CSV 준비 후)
.venv/bin/python src/validation.py \
    --cv out/영상1_측면_angles.csv \
    --ref annotations/영상1_측면_ref.csv \
    --out out/영상1_검증.md --png-dir out/영상1_검증_plots
```

**참값(수동 주석) CSV 스키마** — 주석자가 각도기/수동 디지타이징으로 측정:
- `frame` 컬럼(정수)으로 CV 출력 CSV와 프레임 정렬(내부 조인).
- 검증할 각도는 CV CSV와 **동일한 컬럼명** 사용: `L_elbow_deg`, `R_elbow_deg`,
  `L_knee_deg`, `R_knee_deg`, `L_arm_slot_deg`, `R_arm_slot_deg`, `trunk_tilt_deg`.
- 전 프레임을 주석할 필요 없음 — 주석한 프레임만 남기면 됨. 예:
  ```
  frame,L_elbow_deg,trunk_tilt_deg
  30,128.0,6.0
  60,95.0,-22.0
  70,80.0,-38.0
  ```
- **신뢰도**(inter-rater/test-retest)는 `reliability_icc(rater1, rater2, ...)` 사용.
- ICC 구현은 numpy/scipy만 사용(pingouin 불필요). Shrout & Fleiss(1979) 표준
  예제로 ICC(2,1)=0.290 정합성 검증 완료.

## 엔진 비교 (MediaPipe vs YOLO)

두 엔진을 동일 영상에서 정량 비교 → `out/engine_comparison.md`.

```bash
.venv/bin/pip install ultralytics      # 최초 1회(torch 포함)
# YOLO로도 처리
.venv/bin/python src/analyze.py 입력.mp4 -o 출력_yolo.mp4 --csv 좌표_yolo.csv --engine yolo
# 비교 리포트
.venv/bin/python src/compare_engines.py \
  --pairs "측면:out/영상1_측면_keypoints.csv:out/영상1_측면_yolo_keypoints.csv"
```

**결과 요약**: 두 엔진 모두 주간·역광 영상에서 **100% 검출**. MediaPipe가 지터
근소하게 낮고 **~7배 빠름**(34 vs 4.5 fps). 두 엔진은 몸통 길이 **6–7% 이내**로
일치 → **기본=MediaPipe, YOLO는 교차검증·백업**. 참값 없을 때 두 엔진 일치도를
상호 신뢰도 대리지표로 활용 가능.

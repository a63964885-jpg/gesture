# Gesture Recognizer

[MediaPipe Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)로 웹캠 화면에서 손 제스처를 실시간으로 인식하는 Python 예제입니다.

## 기능

- 손을 최대 2개까지 동시에 인식합니다.
- 손 랜드마크 21개와 뼈대를 화면에 그립니다.
- 손마다 `왼손/오른손: 제스처 (신뢰도)`를 표시합니다.
- 제스처가 바뀔 때 `handle_gesture()`가 한 번 실행됩니다. 여기에 원하는 동작을 연결하면 됩니다.

## 인식 제스처

| 제스처 | 의미 |
|---|---|
| `Thumb_Up` | 👍 엄지 위 |
| `Thumb_Down` | 👎 엄지 아래 |
| `Open_Palm` | ✋ 손바닥 펴기 |
| `Closed_Fist` | ✊ 주먹 |
| `Pointing_Up` | ☝ 검지 위로 |
| `Victory` | ✌ 브이 |
| `ILoveYou` | 🤟 사랑해 |

## 파일 구성

```
gesture/
├── gesture_app.py            # 실시간 웹캠 제스처 인식 코드 (기본 7종)
├── gesture_recognizer.task   # MediaPipe 모델 (float16, 약 8.4MB)
├── features.py               # 랜드마크 → 특징 벡터 변환 (공용)
├── web/                      # 웹 버전 (nike → Nike 로고, ok → 👌)
│   ├── index.html, app.js
│   ├── custom_model.json     # 학습 시 자동 생성되는 웹용 모델
│   └── nike.svg              # 로고 이미지 (교체 가능)
├── gesture_studio.py         # 커스텀 제스처 수집·학습·추론 UI
├── train.py                  # 커스텀 제스처 분류기 학습 (UI가 사용)
├── data/gestures.csv         # 수집한 데이터 (Gesture Studio가 생성)
├── models/custom_gesture.joblib  # 학습된 모델 (학습 시 생성)
├── requirements.txt
└── README.md
```

## 설치

Python 3.14에서 테스트했습니다.

```bash
git clone https://github.com/a63964885-jpg/gesture.git
cd gesture
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # mediapipe, opencv, scikit-learn
```

모델 파일(`gesture_recognizer.task`)은 저장소에 들어 있습니다. 직접 받으려면:

```bash
curl -L -o gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

## 실행

```bash
.venv/bin/python gesture_app.py
```

- 종료: 영상 창을 클릭한 뒤 `q` 또는 `ESC`
- 카메라 고르기: `--camera 1`처럼 번호를 붙이세요. iPhone 연속성 카메라가 연결돼 있으면 iPhone이 0번, MacBook 내장 카메라가 1번으로 잡힐 수 있습니다.
- macOS에서 "웹캠을 열 수 없습니다"가 나오면 **시스템 설정 → 개인정보 보호 및 보안 → 카메라**에서 터미널을 허용한 뒤 다시 실행하세요.

## 동작 바꾸기

`gesture_app.py`의 `GESTURE_ACTIONS`와 `handle_gesture()`를 고치면 됩니다.

```python
def handle_gesture(name, state):
    if name != state["last"]:
        state["last"] = name
        if name == "Thumb_Up":
            ...  # 원하는 동작
```

인식 감도는 `main()`의 `min_hand_detection_confidence`, `min_tracking_confidence` 등으로 조절합니다. 기본값은 0.5입니다.

## 동작 방식

1. OpenCV로 웹캠 프레임을 읽고 좌우 반전(거울 모드)한 뒤 RGB로 바꿉니다.
2. `GestureRecognizer`를 `LIVE_STREAM` 모드로 실행해 `recognize_async()`로 프레임을 비동기로 넘깁니다.
3. 콜백(`on_result`)으로 받은 최신 결과를 프레임 위에 그립니다.

## 나만의 제스처 추가 학습 (Gesture Studio)

```bash
.venv/bin/python gesture_studio.py
```

화면 하나에서 **수집 → 학습 → 추론**을 모두 할 수 있습니다. 오른쪽 위 **카메라** 메뉴에서 쓸 카메라를 바로 바꿀 수 있습니다. 시작할 때부터 정하려면 `--camera 1`을 붙이세요.

```
┌───────────────────────────┬──────────────────────────┐
│                           │  현재 제스처 (큰 글씨)     │
│                           ├──────────────────────────┤
│       웹캠 화면            │ 1. 데이터 수집            │
│  (손 뼈대 + 제스처 표시)    │   제스처 이름 / 녹화 개수  │
│                           │   [● 녹화 시작] ▓▓▓░░     │
│                           │   제스처별 샘플 수 목록     │
│                           ├──────────────────────────┤
│                           │ 2. 학습  [학습 시작]       │
│                           │   정확도 / 혼동 행렬       │
│                           ├──────────────────────────┤
│                           │ 3. 추론  확신도 기준 슬라이더│
└───────────────────────────┴──────────────────────────┘
```

### 1. 데이터 수집
1. **제스처 이름**을 입력합니다. 예: `rock`, `call_me`, `none`
2. **● 녹화 시작**을 누르거나 `Space`를 누르면 3초 카운트다운 뒤 녹화가 시작됩니다.
3. 설정한 개수(기본 300개)가 모이면 자동으로 멈춥니다. 중간에 멈추려면 다시 `Space`를 누르세요.
4. 다른 제스처도 같은 방법으로 모읍니다. 목록에서 제스처를 클릭하면 이름이 자동으로 입력되어 이어서 녹화할 수 있습니다.

**잘 모으는 요령**
- 녹화하는 동안 손을 조금씩 돌리고, 기울이고, 카메라와의 거리도 바꿔 주세요.
- 양손 다 쓸 거라면 왼손과 오른손을 모두 보여주세요.
- **`none`도 꼭 모으세요.** 평소 손 모양, 기본 제스처(주먹, 손바닥 등), 어중간한 손 모양을 섞어 넣으면 됩니다. 이게 없으면 아무 손 모양이나 커스텀 제스처로 잘못 인식합니다.

잘못 녹화했다면 목록에서 그 제스처를 선택하고 **선택한 제스처 데이터 삭제**를 누르세요.

### 2. 학습
**학습 시작**을 누르면 몇 초 안에 끝납니다. 결과에는 테스트 정확도와 혼동 행렬(어떤 제스처끼리 헷갈렸는지)이 나옵니다. 헷갈리는 제스처가 있으면 그 데이터를 더 모은 뒤 다시 학습하세요.

학습이 끝나면 새 모델이 바로 적용됩니다. 다음에 실행할 때도 저장된 모델을 자동으로 불러옵니다.

### 3. 추론
- 커스텀 제스처는 **보라색**, MediaPipe 기본 제스처는 **하늘색**으로 표시됩니다.
- **확신도 기준**: 커스텀 모델이 이 값 이상으로 확신할 때만 커스텀 제스처로 표시합니다. 오인식이 많으면 올리세요.
- **커스텀 제스처 사용**을 끄면 기본 7종만 인식합니다.

### 동작 원리
MediaPipe 공식 학습 도구(Model Maker)는 Python 3.14를 지원하지 않습니다. 그래서 Gesture Recognizer가 뽑아주는 **손 관절 좌표 21개(x, y, z)**를 정규화해서 특징으로 쓰고, 그 위에 작은 신경망(scikit-learn MLP)을 학습합니다.

정규화는 손목 기준 이동, 크기 나누기, 왼손 좌우 뒤집기입니다(`features.py`). 그래서 손의 위치·크기와 왼손/오른손에 상관없이 인식됩니다.

- 수집 데이터: `data/gestures.csv`
- 학습된 모델: `models/custom_gesture.joblib`
- 터미널에서 학습만 하려면 `.venv/bin/python train.py`도 쓸 수 있습니다.

## 웹 버전 (nike → Nike 로고, ok → 👌)

Gesture Studio로 학습한 모델을 브라우저에서 그대로 사용합니다. `nike` 제스처를 하면 손 위에 Nike 로고가, `ok`를 하면 👌 이모지가 나타납니다.

```bash
cd web
python3 -m http.server 8765
```

Chrome에서 **http://localhost:8765** 를 열고 카메라 권한을 허용하세요. 카메라 권한은 `localhost`에서만 동작하니, 파일을 더블클릭해서 여는 방식으로는 안 됩니다.

- MacBook 내장 카메라를 자동으로 고릅니다. 다른 카메라는 오른쪽 **카메라** 메뉴에서 바꿀 수 있습니다.
- 오른쪽 막대에서 커스텀 모델의 실시간 확률을 볼 수 있습니다.
- 깜빡임을 막기 위해 같은 제스처가 3프레임 연속 나와야 표시되고, 손을 바꿔도 0.4초 동안 유지됩니다.
- 로고를 바꾸려면 `web/nike.svg`를 다른 이미지로 교체하세요.
- 다른 제스처에 다른 이미지를 붙이려면 `web/app.js`의 `OVERLAYS`와 `index.html`의 overlay 요소를 추가하세요.

**동작 방식**
- 손 인식은 MediaPipe Tasks Vision JS(CDN)가 합니다.
- 커스텀 분류는 학습할 때 생성되는 `web/custom_model.json`(정규화 값 + 신경망 가중치)으로 브라우저에서 직접 계산합니다. Python 모델과 결과가 같은 것을 확인했습니다.
- 기존 모델만 다시 내보내려면 `.venv/bin/python train.py --export-only`를 실행하세요.

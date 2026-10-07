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
├── gesture_app.py            # 실시간 웹캠 제스처 인식 코드
├── gesture_recognizer.task   # MediaPipe 모델 (float16, 약 8.4MB)
├── requirements.txt
└── README.md
```

## 설치

Python 3.14에서 테스트했습니다.

```bash
git clone https://github.com/a63964885-jpg/gesture.git
cd gesture
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
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

"""MediaPipe Gesture Recognizer - 웹캠 실시간 제스처 인식

인식 제스처: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down,
            Thumb_Up, Victory, ILoveYou
실행: .venv/bin/python gesture_app.py [--camera 1]   (종료: q 또는 ESC)
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"

# 손 21개 랜드마크 연결선
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 새끼
    (0, 17),
]

# 제스처별 동작 (원하는 기능으로 바꿔서 사용)
GESTURE_ACTIONS = {
    "Thumb_Up": "좋아요 👍",
    "Thumb_Down": "싫어요 👎",
    "Open_Palm": "정지 ✋",
    "Closed_Fist": "주먹 ✊",
    "Pointing_Up": "위로 ☝",
    "Victory": "브이 ✌",
    "ILoveYou": "사랑해 🤟",
}

latest_result = None


def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def draw(frame, result):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        if result.gestures and i < len(result.gestures) and result.gestures[i]:
            g = result.gestures[i][0]
            hand = result.handedness[i][0].category_name
            label = f"{hand}: {g.category_name} ({g.score:.2f})"
            x = min(p[0] for p in pts)
            y = max(min(p[1] for p in pts) - 10, 20)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)


def handle_gesture(name, state):
    """제스처가 바뀌었을 때 한 번만 실행"""
    if name != state["last"]:
        state["last"] = name
        if name in GESTURE_ACTIONS:
            print(f"[{time.strftime('%H:%M:%S')}] {name} -> {GESTURE_ACTIONS[name]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (0, 1, ...)")
    args = parser.parse_args()

    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없습니다. (macOS: 터미널에 카메라 권한 허용 필요)")

    state = {"last": None}
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.monotonic() - start) * 1000)
            recognizer.recognize_async(mp_image, ts)

            result = latest_result
            if result is not None:
                draw(frame, result)
                top = result.gestures[0][0].category_name if result.gestures else None
                handle_gesture(top, state)

            cv2.imshow("Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

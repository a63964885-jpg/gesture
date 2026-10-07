"""손 랜드마크 → 학습용 특징 벡터 변환 (collect / train / custom_app 공용)"""
import numpy as np

NUM_FEATURES = 21 * 3


def landmarks_to_features(landmarks, handedness_name):
    """21개 랜드마크를 위치·크기·좌우에 무관한 63차원 벡터로 변환

    - 손목(0번)을 원점으로 이동
    - 왼손은 x를 뒤집어 오른손 기준으로 통일 (한 모델로 양손 인식)
    - 손목에서 가장 먼 점까지의 거리로 나눠 크기 정규화
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness_name == "Left":
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()

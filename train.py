"""data/gestures.csv로 커스텀 제스처 분류기 학습 (gesture_studio.py에서 사용)

단독 실행도 가능: .venv/bin/python train.py
결과: models/custom_gesture.joblib, web/custom_model.json (웹 버전용)
"""
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "gestures.csv"
OUT_PATH = ROOT / "models" / "custom_gesture.joblib"
WEB_MODEL_PATH = ROOT / "web" / "custom_model.json"


def load_data(path=DATA_PATH):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    y = np.array([r[0] for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)
    return X, y


def export_web_model(model, path=WEB_MODEL_PATH):
    """스케일러 + MLP 가중치를 JSON으로 저장 (web/app.js에서 같은 계산을 함)"""
    scaler, mlp = model[0], model[-1]
    data = {
        "classes": mlp.classes_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "layers": [{"W": W.tolist(), "b": b.tolist()}
                   for W, b in zip(mlp.coefs_, mlp.intercepts_)],
    }
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text(json.dumps(data))


def train(data_path=DATA_PATH, out_path=OUT_PATH):
    """학습 후 모델을 저장하고 결과 리포트 문자열을 돌려줌. 실패하면 ValueError."""
    if not Path(data_path).exists():
        raise ValueError("수집된 데이터가 없습니다. 먼저 제스처를 녹화하세요.")
    X, y = load_data(data_path)
    counts = Counter(y.tolist())
    if len(counts) < 2:
        raise ValueError("제스처가 2개 이상 필요합니다. (예: 원하는 제스처 + none)")
    if min(counts.values()) < 10:
        raise ValueError("제스처마다 샘플이 최소 10개는 필요합니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                      random_state=42),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(counts)
    acc = (pred == y_test).mean()

    report = [
        f"테스트 정확도: {acc:.1%}  (학습 {len(y_train)} / 테스트 {len(y_test)})",
        "",
        classification_report(y_test, pred, zero_division=0),
        "혼동 행렬 (행=정답, 열=예측)",
        "  " + " ".join(labels),
        str(confusion_matrix(y_test, pred, labels=labels)),
    ]

    # 평가 후 전체 데이터로 다시 학습해서 저장
    model.fit(X, y)
    Path(out_path).parent.mkdir(exist_ok=True)
    joblib.dump(model, out_path)
    export_web_model(model)
    report.append(f"\n모델 저장 → {Path(out_path).name}, web/{WEB_MODEL_PATH.name}")
    return "\n".join(report)


if __name__ == "__main__":
    import sys
    if "--export-only" in sys.argv:  # 학습 없이 기존 모델만 웹용으로 내보내기
        export_web_model(joblib.load(OUT_PATH))
        raise SystemExit(f"내보냄 → {WEB_MODEL_PATH}")
    try:
        print(train())
    except ValueError as e:
        raise SystemExit(e)

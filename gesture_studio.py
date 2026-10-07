"""Gesture Studio - 커스텀 제스처 수집 · 학습 · 추론을 한 화면에서

실행: .venv/bin/python gesture_studio.py [--camera 1]
"""
import argparse
import csv
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from features import NUM_FEATURES, landmarks_to_features
from gesture_app import HAND_CONNECTIONS, MODEL_PATH
from train import DATA_PATH, OUT_PATH, train

NONE_LABELS = {"none", "None"}
COUNTDOWN_SEC = 3
VIDEO_W, VIDEO_H = 640, 480


def find_cameras(max_index=4):
    """열리는 카메라 번호 목록"""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found or [0]


class GestureStudio:
    def __init__(self, root, camera=0):
        self.root = root
        root.title("Gesture Studio")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cameras = find_cameras()
        self.cap = None
        if not self.open_camera(camera):
            messagebox.showerror("웹캠 오류", f"{camera}번 카메라를 열 수 없습니다.\n"
                                 "시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 터미널을 허용하세요.")
            root.destroy()
            return

        self.recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            ))
        self.start_time = time.monotonic()

        self.recording = False
        self.countdown_end = None
        self.session_count = 0
        self.counts = self.read_counts()
        self.clf = joblib.load(OUT_PATH) if OUT_PATH.exists() else None
        self.training = False
        self.photo = None

        self.build_ui()
        self.refresh_counts()
        self.update_model_status()
        root.bind("<space>", self.on_space)
        self.loop()

    def open_camera(self, index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.camera_index = index
        return True

    def on_camera_change(self, _):
        index = int(self.camera_var.get().split()[0])
        if index != self.camera_index and not self.open_camera(index):
            messagebox.showerror("카메라", f"{index}번 카메라를 열 수 없습니다.")
            self.camera_var.set(f"{self.camera_index} 번")

    # ---------------- UI ----------------
    def build_ui(self):
        self.video = ttk.Label(self.root)
        self.video.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        panel = ttk.Frame(self.root, width=340)
        panel.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")

        row = ttk.Frame(panel)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="카메라").pack(side="left")
        self.camera_var = tk.StringVar(value=f"{self.camera_index} 번")
        cam_box = ttk.Combobox(row, textvariable=self.camera_var, state="readonly", width=8,
                               values=[f"{i} 번" for i in self.cameras])
        cam_box.pack(side="left", padx=6)
        cam_box.bind("<<ComboboxSelected>>", self.on_camera_change)

        self.gesture_var = tk.StringVar(value="손을 보여주세요")
        ttk.Label(panel, textvariable=self.gesture_var,
                  font=("Helvetica", 22, "bold")).pack(fill="x", pady=(0, 10))

        # 1. 수집
        box = ttk.LabelFrame(panel, text=" 1. 데이터 수집 ", padding=8)
        box.pack(fill="x", pady=4)

        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="제스처 이름").pack(side="left")
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Combobox(row, textvariable=self.label_var, width=16)
        self.label_entry.pack(side="left", padx=6, fill="x", expand=True)

        row = ttk.Frame(box)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="녹화 개수").pack(side="left")
        self.target_var = tk.IntVar(value=300)
        ttk.Spinbox(row, from_=50, to=2000, increment=50, width=7,
                    textvariable=self.target_var).pack(side="left", padx=6)

        self.rec_btn = ttk.Button(box, text="● 녹화 시작 (Space)", command=self.toggle_record)
        self.rec_btn.pack(fill="x", pady=2)
        self.progress = ttk.Progressbar(box, maximum=300)
        self.progress.pack(fill="x", pady=2)

        self.tree = ttk.Treeview(box, columns=("count",), height=6)
        self.tree.heading("#0", text="제스처")
        self.tree.heading("count", text="샘플 수")
        self.tree.column("#0", width=170)
        self.tree.column("count", width=80, anchor="e")
        self.tree.pack(fill="x", pady=4)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        ttk.Button(box, text="선택한 제스처 데이터 삭제", command=self.delete_label).pack(fill="x")
        ttk.Label(box, text="Tip: 'none'(아무 제스처 아님)도 꼭 모으세요",
                  foreground="gray").pack(anchor="w", pady=(4, 0))

        # 2. 학습
        box = ttk.LabelFrame(panel, text=" 2. 학습 ", padding=8)
        box.pack(fill="x", pady=4)
        self.train_btn = ttk.Button(box, text="학습 시작", command=self.start_training)
        self.train_btn.pack(fill="x")
        self.log = tk.Text(box, height=9, width=42, font=("Menlo", 10), wrap="none")
        self.log.pack(fill="both", pady=4)

        # 3. 추론
        box = ttk.LabelFrame(panel, text=" 3. 추론 ", padding=8)
        box.pack(fill="x", pady=4)
        self.use_custom = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="커스텀 제스처 사용", variable=self.use_custom).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="확신도 기준").pack(side="left")
        self.threshold = tk.DoubleVar(value=0.8)
        self.th_label = ttk.Label(row, text="0.80", width=5)
        self.th_label.pack(side="right")
        ttk.Scale(row, from_=0.5, to=0.99, variable=self.threshold,
                  command=lambda v: self.th_label.config(text=f"{float(v):.2f}")
                  ).pack(side="left", fill="x", expand=True, padx=6)
        self.model_status = ttk.Label(box, foreground="gray")
        self.model_status.pack(anchor="w")

    # ---------------- 데이터 ----------------
    def read_counts(self):
        if not DATA_PATH.exists():
            return Counter()
        with open(DATA_PATH, newline="") as f:
            return Counter(r[0] for r in list(csv.reader(f))[1:])

    def refresh_counts(self):
        self.tree.delete(*self.tree.get_children())
        for label, n in sorted(self.counts.items()):
            self.tree.insert("", "end", iid=label, text=label, values=(n,))
        self.label_entry["values"] = sorted(self.counts)

    def on_tree_select(self, _):
        sel = self.tree.selection()
        if sel and not self.recording:
            self.label_var.set(sel[0])

    def delete_label(self):
        sel = self.tree.selection()
        if not sel or self.recording:
            return
        label = sel[0]
        if not messagebox.askyesno("삭제", f"'{label}' 샘플 {self.counts[label]}개를 삭제할까요?"):
            return
        with open(DATA_PATH, newline="") as f:
            rows = list(csv.reader(f))
        with open(DATA_PATH, "w", newline="") as f:
            csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if r[0] != label])
        del self.counts[label]
        self.refresh_counts()
        self.write_log(f"'{label}' 데이터 삭제됨. 다시 학습하세요.")

    # ---------------- 녹화 ----------------
    def on_space(self, event):
        if not isinstance(event.widget, (ttk.Entry, tk.Entry, tk.Text, ttk.Spinbox)):
            self.toggle_record()

    def toggle_record(self):
        if self.recording or self.countdown_end:
            self.stop_record()
            return
        label = self.label_var.get().strip()
        if not label or "," in label:
            messagebox.showwarning("제스처 이름", "제스처 이름을 입력하세요. (쉼표 제외)")
            return
        if self.training:
            return
        try:
            self.target = max(1, int(self.target_var.get()))
        except (tk.TclError, ValueError):
            self.target = 300
        self.current_label = label
        self.session_count = 0
        self.progress.config(maximum=self.target, value=0)
        self.countdown_end = time.monotonic() + COUNTDOWN_SEC
        self.rec_btn.config(text="■ 정지 (Space)")
        self.label_entry.config(state="disabled")
        self.root.focus_set()

    def begin_record(self):
        self.countdown_end = None
        DATA_PATH.parent.mkdir(exist_ok=True)
        new_file = not DATA_PATH.exists()
        self.csv_file = open(DATA_PATH, "a", newline="")
        self.writer = csv.writer(self.csv_file)
        if new_file:
            self.writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        self.recording = True

    def stop_record(self):
        if self.recording:
            self.csv_file.close()
            self.write_log(f"'{self.current_label}' {self.session_count}개 녹화 완료")
        self.recording = False
        self.countdown_end = None
        self.rec_btn.config(text="● 녹화 시작 (Space)")
        self.label_entry.config(state="normal")
        self.refresh_counts()

    # ---------------- 학습 ----------------
    def start_training(self):
        if self.training or self.recording or self.countdown_end:
            return
        self.training = True
        self.train_btn.config(state="disabled", text="학습 중...")
        self.write_log("학습 중...", clear=True)
        threading.Thread(target=self.train_worker, daemon=True).start()

    def train_worker(self):
        try:
            report, ok = train(), True
        except ValueError as e:
            report, ok = str(e), False
        self.root.after(0, self.train_done, report, ok)

    def train_done(self, report, ok):
        self.training = False
        self.train_btn.config(state="normal", text="학습 시작")
        self.write_log(report, clear=True)
        if ok:
            self.clf = joblib.load(OUT_PATH)
        self.update_model_status()

    def update_model_status(self):
        if self.clf is None:
            self.model_status.config(text="학습된 모델 없음 (기본 제스처만 인식)")
        else:
            self.model_status.config(text="모델: " + ", ".join(self.clf.classes_))

    def write_log(self, text, clear=False):
        if clear:
            self.log.delete("1.0", "end")
        self.log.insert("end", text + "\n")
        self.log.see("end")

    # ---------------- 메인 루프 ----------------
    def classify(self, result, i, features):
        if self.clf is not None and self.use_custom.get():
            probs = self.clf.predict_proba([features])[0]
            best = probs.argmax()
            label, score = self.clf.classes_[best], probs[best]
            if score >= self.threshold.get() and label not in NONE_LABELS:
                return label, score, (255, 0, 255)
        g = result.gestures[i][0]
        return g.category_name, g.score, (255, 255, 0)

    def loop(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int((time.monotonic() - self.start_time) * 1000))
            self.process(frame, result)
            self.show(frame)
        self.root.after(10, self.loop)

    def process(self, frame, result):
        if self.countdown_end and time.monotonic() >= self.countdown_end:
            self.begin_record()

        top = None
        for i, landmarks in enumerate(result.hand_landmarks):
            hand = result.handedness[i][0].category_name
            pts = [(int(lm.x * VIDEO_W), int(lm.y * VIDEO_H)) for lm in landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(frame, p, 3, (0, 0, 255), -1)

            features = landmarks_to_features(landmarks, hand)
            if i == 0 and self.recording:
                self.writer.writerow([self.current_label] + features.tolist())
                self.session_count += 1
                self.counts[self.current_label] += 1

            name, score, color = self.classify(result, i, features)
            if i == 0:
                top = name
            x, y = min(p[0] for p in pts), max(min(p[1] for p in pts) - 10, 20)
            cv2.putText(frame, f"{hand}: {name} ({score:.2f})", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self.gesture_var.set(top or "손을 보여주세요")

        if self.countdown_end:
            remain = int(self.countdown_end - time.monotonic()) + 1
            cv2.putText(frame, str(remain), (VIDEO_W // 2 - 30, VIDEO_H // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        elif self.recording:
            cv2.circle(frame, (25, 25), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {self.current_label} {self.session_count}/{self.target}",
                        (45, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not result.hand_landmarks:
                cv2.putText(frame, "no hand", (45, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            self.progress.config(value=self.session_count)
            if self.session_count % 10 == 0 and self.tree.exists(self.current_label):
                self.tree.set(self.current_label, "count", self.counts[self.current_label])
            if self.session_count >= self.target:
                self.stop_record()

    def show(self, frame):
        ok, ppm = cv2.imencode(".ppm", frame)
        if ok:
            self.photo = tk.PhotoImage(data=ppm.tobytes(), format="ppm")
            self.video.config(image=self.photo)

    def on_close(self):
        if self.recording:
            self.stop_record()
        if self.cap is not None:
            self.cap.release()
        self.recognizer.close()
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (0, 1, ...)")
    args = parser.parse_args()
    root = tk.Tk()
    GestureStudio(root, args.camera)
    root.mainloop()

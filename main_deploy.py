import sys
import os
import cv2
import time
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# GUI Imports (PyQt5!)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets, uic

# --- PATH SETUP (Works anywhere!) ---
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
UI_DIR = ASSETS_DIR / "ui"

# Optional: SSIM for duplicate check
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("⚠️ scikit-image not found. Using Histogram fallback.")

try:
    import resources_rc  # ← This loads all embedded resources
except ImportError:
    print("⚠️ resources_rc.py not found. GUI images may be missing.")

# --- CONFIGURATION (PT MODELS ONLY) ---
EVENT_MODES = {
    "Formal": {
        "model": MODELS_DIR / "formal.pt",
        "conf": 0.70,
        "rules": {0: {"name": "Smart Suit", "points": 50}},
        "ui": UI_DIR / "formal.ui"
    },
    "Wedding": {
        "model": MODELS_DIR / "wedding.pt",
        "conf": 0.80,
        "rules": {
            0: {"name": "Baju Melayu", "points": 30},
            1: {"name": "Sampin", "points": 25},
            2: {"name": "Songkok", "points": 15},
            3: {"name": "Baju Kebaya", "points": 50}
        },
        "ui": UI_DIR / "wedding.ui"
    },
    "Mosque": {
        "model": MODELS_DIR / "mosque.pt",
        "conf": 0.50,
        "rules": {0: {"name": "Tshirt", "points": 75}},
        "ui": UI_DIR / "mosque.ui"
    }
}

class BaseKioskWindow(QMainWindow):
    def __init__(self, mode_key):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Load UI
        self.mode_config = EVENT_MODES[mode_key]
        ui_file = self.mode_config["ui"]
        if not ui_file.exists():
            raise FileNotFoundError(f"❌ UI file missing: {ui_file}")
        uic.loadUi(str(ui_file), self)

        # State
        self.leaderboard = []
        self.last_scan_time = time.time()
        self.msg_text = "READY TO SCAN"
        self.msg_color = (255, 255, 255)
        self.detected_classes = []

        # Load Human Model (.pt only)
        human_model_path = MODELS_DIR / "yolov8m.pt"
        if not human_model_path.exists():
            raise FileNotFoundError(f"❌ Human model not found: {human_model_path}")
        self.human_model = YOLO(str(human_model_path), task="detect")
        self.human_model.to("cpu")  # ✅ Valid for .pt

        # Load Clothes Model (.pt only)
        clothes_model_path = self.mode_config["model"]
        if not clothes_model_path.exists():
            print(f"⚠️ Clothes model {clothes_model_path} not found. Using human model as fallback.")
            self.clothes_model = self.human_model
        else:
            self.clothes_model = YOLO(str(clothes_model_path), task="detect")
            self.clothes_model.to("cpu")  # ✅ Valid for .pt

        # Camera
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError("❌ Cannot open camera (index 0).")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Leaderboard layout
        if hasattr(self, 'rank_container'):
            if not self.rank_container.layout():
                self.rank_layout = QVBoxLayout(self.rank_container)
                self.rank_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
                self.rank_layout.setContentsMargins(0, 0, 0, 0)
            else:
                self.rank_layout = self.rank_container.layout()

    def is_duplicate(self, new_crop, new_box_center=None):
        if not self.leaderboard: return False
        new_gray = cv2.cvtColor(new_crop, cv2.COLOR_BGR2GRAY)
        new_h, new_w = new_gray.shape
        for entry in self.leaderboard:
            if 'image' not in entry: continue
            if HAS_SSIM:
                old_gray = cv2.cvtColor(entry['image'], cv2.COLOR_BGR2GRAY)
                old_gray = cv2.resize(old_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                score, _ = ssim(new_gray, old_gray, full=True, data_range=255)
                if score > 0.85: return True
            else:
                hsv_new = cv2.cvtColor(new_crop, cv2.COLOR_BGR2HSV)
                hist_new = cv2.calcHist([hsv_new], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist_new, hist_new, 0, 1, cv2.NORM_MINMAX)
                hsv_old = cv2.cvtColor(entry['image'], cv2.COLOR_BGR2HSV)
                hist_old = cv2.calcHist([hsv_old], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist_old, hist_old, 0, 1, cv2.NORM_MINMAX)
                similarity = cv2.compareHist(hist_new, hist_old, cv2.HISTCMP_CORREL)
                if similarity > 0.85: return True
        return False

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        h_cam, w_cam, _ = frame.shape
        current_time = time.time()
        self.msg_text = "READY TO SCAN"
        self.msg_color = (255, 255, 255)
        self.detected_classes = []
        if (current_time - self.last_scan_time) >= 5.0:
            results = self.human_model.predict(frame, classes=[0], verbose=False, conf=0.5)
            best_box = None
            center_x = w_cam // 2
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cx = (x1 + x2) // 2
                    if abs(cx - center_x) < 300:
                        best_box = (x1, y1, x2, y2)
                        break
            if best_box:
                x1, y1, x2, y2 = best_box
                box_h = y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if box_h / h_cam < 0.40:
                    self.msg_text = "⚠️ STEP CLOSER"
                    self.msg_color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    crop = frame[max(0, y1):min(h_cam, y2), max(0, x1):min(w_cam, x2)]
                    if crop.size > 0:
                        c_res = self.clothes_model.predict(crop, verbose=False, conf=self.mode_config['conf'])
                        score = 0
                        rules = self.mode_config['rules']
                        for cr in c_res:
                            for cb in cr.boxes:
                                cls_id = int(cb.cls[0].cpu().item())
                                conf_val = float(cb.conf[0].cpu().item())
                                class_name = rules.get(cls_id, {}).get('name', f'Item {cls_id}')
                                self.detected_classes.append(f"{class_name} {int(conf_val*100)}%")
                                if cls_id in rules:
                                    score += rules[cls_id]['points']
                        if score > 0:
                            if self.is_duplicate(crop, (cx, cy)):
                                self.msg_text = "⚠️ ALREADY SCANNED"
                                self.msg_color = (0, 165, 255)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 5)
                            else:
                                score += random.randint(1, 10)
                                self.msg_text = f"✅ SCORE: {score} PTS"
                                self.msg_color = (0, 255, 0)
                                self.last_scan_time = current_time
                                self.update_leaderboard(score, crop)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                        else:
                            self.msg_text = "❌ NO FASHION DETECTED"
                            self.msg_color = (0, 165, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
        num_lines = len(self.detected_classes) if self.detected_classes else 0
        base_height = 60
        extra_height = 30 * min(3, num_lines)
        total_bar_height = base_height + extra_height
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h_cam - total_bar_height), (w_cam, h_cam), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        font = cv2.FONT_HERSHEY_TRIPLEX
        text_y = h_cam - extra_height - 20
        cv2.putText(frame, self.msg_text, (20, text_y), font, 1.0, self.msg_color, 2, cv2.LINE_AA)
        if self.detected_classes:
            start_y = text_y + 35
            for i, txt in enumerate(self.detected_classes[:3]):
                cv2.putText(frame, f"• {txt}", (20, start_y + (i * 30)), font, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_frame.data, w_cam, h_cam, w_cam * 3, QImage.Format_RGB888)
        if hasattr(self, 'camera_frame'):
            self.camera_frame.setPixmap(
                QPixmap.fromImage(qt_img).scaled(
                    self.camera_frame.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

    def update_leaderboard(self, score, crop, box_center=None):
        target_w, target_h = 400, 600
        h, w = crop.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        entry = {'score': score, 'image': canvas.copy()}
        self.leaderboard.append(entry)
        self.leaderboard.sort(key=lambda x: x['score'], reverse=True)
        self.leaderboard = self.leaderboard[:5]
        if not hasattr(self, 'rank_layout'): return
        while self.rank_layout.count():
            item = self.rank_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for entry in self.leaderboard:
            h, w, _ = entry['image'].shape
            rgb = cv2.cvtColor(entry['image'], cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
            lbl = QLabel()
            lbl.setPixmap(QPixmap.fromImage(q_img).scaled(300, 450, Qt.AspectRatioMode.KeepAspectRatio))
            lbl.setStyleSheet("border: 2px solid gold; border-radius: 10px; background: black;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rank_layout.addWidget(lbl)
            score_lbl = QLabel(f"{entry['score']} PTS")
            score_lbl.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
            score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rank_layout.addWidget(score_lbl)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.capture.release()
            self.close()

# --- WINDOW CLASSES ---
class FormalWindow(BaseKioskWindow):
    def __init__(self): super().__init__("Formal")

class WeddingWindow(BaseKioskWindow):
    def __init__(self): super().__init__("Wedding")

class MosqueWindow(BaseKioskWindow):
    def __init__(self): super().__init__("Mosque")

# --- MAIN MENU ---
class CodaGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        main_ui = UI_DIR / "testpicture.ui"
        if not main_ui.exists():
            raise FileNotFoundError(f"❌ Main UI not found: {main_ui}")
        uic.loadUi(str(main_ui), self)
        self.btn_mosque.clicked.connect(self.open_mosque)
        self.btn_formal.clicked.connect(self.open_formal)
        self.btn_wedding.clicked.connect(self.open_wedding)
        self.show()
    
    def open_mosque(self):
        self.mosque_screen = MosqueWindow()
        self.mosque_screen.showFullScreen()
    
    def open_formal(self):
        self.formal_screen = FormalWindow()
        self.formal_screen.showFullScreen()
        
    def open_wedding(self):
        self.wedding_screen = WeddingWindow()
        self.wedding_screen.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CodaGUI()
    sys.exit(app.exec_())
"""
Hand Finger Size Engine — Webcam + MANUAL CALIBRATION
Works with HP True Vision FHD (or any laptop webcam).

Why this version?
- You asked for **manual calibration** to ensure real-size (mm) measurements.
- Two manual calibration modes are included:
  1) 2‑Point SCALE: Click two points across an object with a known length (enter length in mm).
     → Good if your hand stays at the same distance from the camera as that object.
  2) 4‑Corner PLANE: Click the 4 corners of a known rectangle (e.g., credit/ID card).
     → We build a top-down metric (rectified) view; mm stays consistent anywhere on that plane.

Keys & Buttons
- Buttons (top-right): Save, Debug, Reset, 2-pt, 4-pt, Quit
- Keys (case-insensitive):
    s/S  Save snapshot (CSV + FORM CSV + JSONL)
    d/D  Toggle debug edges window
    r/R  Reset calibration points
    2    Start 2‑Point calibration
    4    Start 4‑Corner calibration
    q/Q or ESC Quit

Install
  pip install mediapipe opencv-python numpy pandas

Usage
  1) Start the app.
  2) Choose a calibration mode (click **2-pt** or **4-pt**, or press 2/4).
  3) Click points in the video:
     • 2-pt: click endpoints of a known-length object (e.g., 100 mm on a ruler).
     • 4-pt: click the 4 rectangle corners (order doesn't matter).
  4) Adjust the on-screen sliders to set the known mm length/rectangle size.
  5) Once you see “SCALE OK” or “PLANE OK”, place your hand appropriately and read mm.
  6) Save when ready.

Notes
- 2‑pt mode assumes your hand is at the same depth as the reference segment.
- 4‑pt mode assumes your hand remains on the same flat **plane** as the rectangle.
- For “true 3D” anywhere in space, a depth camera is required (not covered here).
"""
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd

# ---------- MediaPipe ----------
try:
    import mediapipe as mp
except Exception:
    raise SystemExit("mediapipe is required: pip install mediapipe")

# ---------- Constants ----------
SCALE_PX_PER_MM = 4.0  # for rectified plane in 4-pt mode: 1 mm = 4 px
DEFAULT_RECT_W_MM = 85.60   # credit card long side
DEFAULT_RECT_H_MM = 53.98   # credit card short side
DEFAULT_2PT_MM    = 100.0   # default known length for 2-point calibration

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# (landmark indices for MCP/PIP/DIP/TIP; thumb uses CMC/MCP/IP/TIP)
FINGER_LMS = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}

@dataclass
class Measure:
    length_mm: Optional[float]
    width_mm: Optional[float]

# ---------- Helpers ----------
def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL]."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)

# ---------- Hand detector ----------
class HandDetector:
    def __init__(self, max_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_hands,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.draw = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    def detect(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def landmarks_px(self, hand_landmarks, w: int, h: int) -> Dict[int, Tuple[int,int]]:
        pts = {}
        for i, lm in enumerate(hand_landmarks.landmark):
            pts[i] = (int(lm.x * w), int(lm.y * h))
        return pts

# ---------- Manual Calibrator ----------
class ManualCalibrator:
    """
    Two modes:
      - 'scale2': set mm_per_px via 2 clicked points + known length (mm)
      - 'plane4': set H (homography) via 4 clicked corners of a known rectangle (mm x mm)
    """
    def __init__(self, window_name: str):
        self.window = window_name
        self.mode: str = 'none'   # 'none' | 'scale2' | 'plane4'
        self.pts: List[Tuple[int,int]] = []

        # outputs
        self.mm_per_px: Optional[float] = None   # for 2-pt
        self.H: Optional[np.ndarray] = None      # for 4-pt
        self.rect_size_px: Optional[Tuple[int,int]] = None

        # parameters
        self.known_len_mm: float = DEFAULT_2PT_MM
        self.rect_w_mm: float = DEFAULT_RECT_W_MM
        self.rect_h_mm: float = DEFAULT_RECT_H_MM

        # trackbars
        self._ensure_trackbars()

    def _ensure_trackbars(self):
        # Common callback stubs (OpenCV trackbars need a callback)
        def on_len(val):
            self.known_len_mm = max(1, float(val))
        def on_rect_w(val):
            self.rect_w_mm = max(10, float(val))
            self.H = None  # force recompute
        def on_rect_h(val):
            self.rect_h_mm = max(10, float(val))
            self.H = None

        # Create trackbars (idempotent: will just update on each run)
        try:
            cv2.createTrackbar('2pt_known_mm', self.window, int(round(self.known_len_mm)), 400, on_len)
            cv2.setTrackbarPos('2pt_known_mm', self.window, int(round(self.known_len_mm)))
        except Exception:
            pass
        try:
            cv2.createTrackbar('4pt_rect_w_mm', self.window, int(round(self.rect_w_mm)), 200, on_rect_w)
            cv2.setTrackbarPos('4pt_rect_w_mm', self.window, int(round(self.rect_w_mm)))
        except Exception:
            pass
        try:
            cv2.createTrackbar('4pt_rect_h_mm', self.window, int(round(self.rect_h_mm)), 200, on_rect_h)
            cv2.setTrackbarPos('4pt_rect_h_mm', self.window, int(round(self.rect_h_mm)))
        except Exception:
            pass

    def set_mode(self, mode: str):
        if mode not in ('scale2','plane4','none'):
            return
        self.mode = mode
        self.pts = []
        if mode == 'scale2':
            self.H = None
            self.mm_per_px = None
        elif mode == 'plane4':
            self.mm_per_px = None
            self.H = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'scale2':
                if len(self.pts) >= 2:
                    self.pts = []
                self.pts.append((x, y))
                if len(self.pts) == 2:
                    self._compute_scale2()
            elif self.mode == 'plane4':
                if len(self.pts) >= 4:
                    self.pts = []
                self.pts.append((x, y))
                if len(self.pts) == 4:
                    self._compute_plane4()

    def _compute_scale2(self):
        if len(self.pts) != 2:
            return
        p1 = np.array(self.pts[0], dtype=np.float32)
        p2 = np.array(self.pts[1], dtype=np.float32)
        d_px = float(np.linalg.norm(p2 - p1))
        if d_px > 1:
            self.mm_per_px = self.known_len_mm / d_px
        else:
            self.mm_per_px = None

    def _compute_plane4(self):
        if len(self.pts) != 4:
            return
        quad = order_quad(np.array(self.pts, dtype=np.float32))
        dst_w = int(round(self.rect_w_mm * SCALE_PX_PER_MM))
        dst_h = int(round(self.rect_h_mm * SCALE_PX_PER_MM))
        dst = np.array([[0,0], [dst_w-1,0], [dst_w-1,dst_h-1], [0,dst_h-1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(quad, dst)
        self.H = H
        self.rect_size_px = (dst_w, dst_h)

    def draw(self, frame: np.ndarray):
        # Visualize calibration points/lines
        if self.mode == 'scale2':
            for pt in self.pts:
                cv2.circle(frame, pt, 5, (0, 255, 255), -1)
            if len(self.pts) == 2:
                cv2.line(frame, self.pts[0], self.pts[1], (0,255,255), 2)
        elif self.mode == 'plane4':
            for pt in self.pts:
                cv2.circle(frame, pt, 5, (0, 255, 255), -1)
            if len(self.pts) == 4:
                quad = np.array(self.pts, dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(frame, [quad], True, (0,255,255), 2)

    def status_text(self) -> str:
        if self.mode == 'scale2':
            if self.mm_per_px:
                return f"SCALE OK — {self.mm_per_px:.4f} mm/px (len={self.known_len_mm:.1f} mm)"
            return "SCALE: click 2 pts on a known length; adjust '2pt_known_mm' slider"
        elif self.mode == 'plane4':
            if self.H is not None:
                return f"PLANE OK — rect {self.rect_w_mm:.1f}×{self.rect_h_mm:.1f} mm"
            return "PLANE: click 4 corners of a known rectangle; adjust '4pt_rect_*_mm' sliders"
        return "MODE: none — choose 2‑pt or 4‑pt"

# ---------- Engine ----------
class FingerMeasureEngine:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        # Helpful defaults
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.det = HandDetector(max_hands=1)
        self.rows: List[Dict] = []
        self.window = "Hand Size — Manual Calibration"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        self.cal = ManualCalibrator(self.window)
        self.show_debug = False
        self.dbg_name = "Edges Debug"
        cv2.setMouseCallback(self.window, self._on_mouse)

        # UI buttons (top-right)
        self.btns = {'save':None, 'debug':None, 'reset':None, 'mode2':None, 'mode4':None, 'quit':None}

    # ---- UI helpers ----
    def _draw_buttons(self, img: np.ndarray):
        h, w = img.shape[:2]
        pad = 10
        bw, bh = 110, 36
        x = w - pad - bw
        y = pad
        order = [('save','Save'), ('debug','Debug'), ('reset','Reset'),
                 ('mode2','2-pt'), ('mode4','4-pt'), ('quit','Quit')]
        self.btns = {}
        for key,label in reversed(order):
            rect = (x, y, x+bw, y+bh)
            self._draw_btn(img, rect, label)
            self.btns[key] = rect
            x -= (bw + pad)

    @staticmethod
    def _draw_btn(img, rect, label):
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        tx = x1 + 10
        ty = y1 + 24
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _on_mouse(self, event, x, y, flags, param):
        # forward to calibrator for point collection
        self.cal.on_mouse(event, x, y, flags, param)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # buttons
        for name, rect in (self.btns or {}).items():
            if rect is None: continue
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == 'save':
                    self._pending_action = 'save'
                elif name == 'debug':
                    self._pending_action = 'debug'
                elif name == 'reset':
                    self._pending_action = 'reset'
                elif name == 'mode2':
                    self._pending_action = 'mode2'
                elif name == 'mode4':
                    self._pending_action = 'mode4'
                elif name == 'quit':
                    self._pending_action = 'quit'

    # ---- measurement helpers ----
    @staticmethod
    def _width_at_mid(rect_gray: np.ndarray, p1: np.ndarray, p2: np.ndarray, max_scan_px=100) -> Optional[float]:
        v = p2 - p1
        if np.linalg.norm(v) < 1e-6:
            return None
        mid = (p1 + p2) / 2.0
        n = np.array([-v[1], v[0]], dtype=np.float32)
        n /= (np.linalg.norm(n) + 1e-9)
        edges = cv2.Canny(rect_gray, 40, 140)
        H, W = edges.shape

        def first_edge(sign):
            for t in range(2, max_scan_px):
                x = int(round(mid[0] + sign * t * n[0]))
                y = int(round(mid[1] + sign * t * n[1]))
                if x < 0 or y < 0 or x >= W or y >= H:
                    break
                if edges[y, x]:
                    return (x, y)
            return None
        a = first_edge(+1)
        b = first_edge(-1)
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _snapshot(self, measures: Dict[str, Measure]):
        """
        Save in two formats:
        1) Wide CSV  → finger_measurements_mm.csv
        2) Long/Form CSV → finger_measurements_form.csv (one row per finger)
        Also writes JSONL alongside the form CSV for easy ingestion.
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        # --- Wide format (one row, many columns) ---
        wide_row = {"timestamp": ts}
        for name in FINGER_NAMES:
            m = measures.get(name)
            wide_row[f"{name}_length_mm"] = round(m.length_mm, 2) if m and m.length_mm else None
            wide_row[f"{name}_width_mm"]  = round(m.width_mm, 2)  if m and m.width_mm  else None
        self.rows.append(wide_row)
        pd.DataFrame(self.rows).to_csv("finger_measurements_mm.csv", index=False)

        # --- Long/Form format (one record per finger) ---
        form_rows: List[Dict[str, object]] = []
        for name in FINGER_NAMES:
            m = measures.get(name)
            if m is None:
                continue
            form_rows.append({
                "timestamp": ts,
                "finger": name,
                "length_mm": round(m.length_mm, 2) if m.length_mm else None,
                "width_mm":  round(m.width_mm,  2) if m.width_mm  else None,
            })

        if form_rows:
            df_form = pd.DataFrame(form_rows)
            try:
                old = pd.read_csv("finger_measurements_form.csv")
            except Exception:
                old = None
            if old is not None and not old.empty:
                df_form = pd.concat([old, df_form], ignore_index=True)
            df_form.to_csv("finger_measurements_form.csv", index=False)

            # JSONL too (newline-delimited JSON)
            try:
                import json
                with open("finger_measurements_form.jsonl", "a", encoding="utf-8") as f:
                    for rec in form_rows:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[WARN] JSONL append failed: {e}")

        print("[INFO] Saved → finger_measurements_mm.csv & finger_measurements_form.csv (and .jsonl)")

    def run(self):
        print("[INFO] Manual calibration ready. Press 2 or 4 (or click buttons) to calibrate.")
        self._pending_action = None

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            overlay = frame.copy()

            # Draw calibration guides/points
            self.cal.draw(overlay)

            # Status line
            status = self.cal.status_text()
            cv2.putText(overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            measures: Dict[str, Measure] = {}

            # Detection + measurement depending on calibration mode
            results = self.det.detect(frame)
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.det.draw.draw_landmarks(
                        overlay, hand, self.det.mp_hands.HAND_CONNECTIONS,
                        self.det.styles.get_default_hand_landmarks_style(),
                        self.det.styles.get_default_hand_connections_style(),
                    )
                    pts_px = self.det.landmarks_px(hand, w, h)

                    if self.cal.mode == 'scale2' and self.cal.mm_per_px:
                        # Measure directly in image plane
                        mmpp = self.cal.mm_per_px
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        for name, (a,b,c,d) in FINGER_LMS.items():
                            p0 = np.array(pts_px[a], dtype=np.float32)
                            p1 = np.array(pts_px[b], dtype=np.float32)
                            p2 = np.array(pts_px[c], dtype=np.float32)
                            p3 = np.array(pts_px[d], dtype=np.float32)
                            L_px = np.linalg.norm(p1-p0)+np.linalg.norm(p2-p1)+np.linalg.norm(p3-p2)
                            L_mm = float(L_px * mmpp)
                            # width along normal to the mid phalanx
                            W_px = self._width_at_mid(gray, p1, p2)
                            W_mm = float(W_px * mmpp) if W_px is not None else None
                            measures[name] = Measure(L_mm, W_mm)

                    elif self.cal.mode == 'plane4' and self.cal.H is not None and self.cal.rect_size_px is not None:
                        # Measure in rectified metric plane
                        Hm = self.cal.H
                        rect_w, rect_h = self.cal.rect_size_px
                        rect_img = cv2.warpPerspective(frame, Hm, (rect_w, rect_h))
                        rect_gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
                        pts = np.array([pts_px[i] for i in range(21)], dtype=np.float32)
                        pts_rect = apply_H(Hm, pts)
                        mmpp = 1.0 / SCALE_PX_PER_MM
                        for name, (a,b,c,d) in FINGER_LMS.items():
                            p0 = pts_rect[a]; p1 = pts_rect[b]; p2 = pts_rect[c]; p3 = pts_rect[d]
                            L_px = np.linalg.norm(p1-p0)+np.linalg.norm(p2-p1)+np.linalg.norm(p3-p2)
                            L_mm = float(L_px * mmpp)
                            W_px = self._width_at_mid(rect_gray, p1, p2)
                            W_mm = float(W_px * mmpp) if W_px is not None else None
                            measures[name] = Measure(L_mm, W_mm)

            # HUD measurements
            cv2.rectangle(overlay, (0, h-90), (w, h), (0, 0, 0), -1)
            y0 = h - 60
            for i, name in enumerate(FINGER_NAMES):
                m = measures.get(name)
                txt = f"{name}:"
                if m and m.length_mm:
                    txt += f" L={m.length_mm:.1f}mm"
                if m and m.width_mm:
                    txt += f"  W~{m.width_mm:.1f}mm"
                cv2.putText(overlay, txt, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Buttons and key hints
            self._draw_buttons(overlay)
            cv2.putText(overlay, "Keys: 2=2-pt, 4=4-pt, r=reset, s=save, d=debug, q/Esc=quit",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow(self.window, overlay)

            # Debug window (edges)
            if self.show_debug:
                dbg = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
                cv2.imshow(self.dbg_name, dbg)
            else:
                try:
                    cv2.destroyWindow(self.dbg_name)
                except Exception:
                    pass

            # Input handling
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q'), 27) or getattr(self, "_pending_action", None) == 'quit':
                break
            elif k in (ord('s'), ord('S')) or getattr(self, "_pending_action", None) == 'save':
                if measures:
                    self._snapshot(measures)
                else:
                    print("[WARN] No measurements yet. Calibrate and ensure a hand is detected.")
                self._pending_action = None
            elif k in (ord('d'), ord('D')) or getattr(self, "_pending_action", None) == 'debug':
                self.show_debug = not self.show_debug
                self._pending_action = None
            elif k in (ord('r'), ord('R')) or getattr(self, "_pending_action", None) == 'reset':
                self.cal.set_mode(self.cal.mode)  # clears points & outputs for current mode
                self._pending_action = None
            elif k == ord('2') or getattr(self, "_pending_action", None) == 'mode2':
                self.cal.set_mode('scale2')
                self._pending_action = None
            elif k == ord('4') or getattr(self, "_pending_action", None) == 'mode4':
                self.cal.set_mode('plane4')
                self._pending_action = None

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = FingerMeasureEngine(camera_id=0)
    engine.run()

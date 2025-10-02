"""
Hand Finger Size Engine — Webcam‑only (HP True Vision FHD ok)
Auto‑scale with a **credit/ID card** as passive reference (no clicks, no markers).

Features
- Auto-detects credit/ID card each frame → builds a top‑down metric view (homography).
- Measures per finger: length (MCP→TIP) and width at mid‑phalanx, in millimeters.
- No calibration, no clicks. Works with any standard laptop webcam.
- On‑screen buttons (Save / Debug / Quit) + case‑insensitive keys.
- Saves both wide CSV and FORM (long) CSV + JSONL.

Controls
  • Click buttons (top-right), or
  • Keys: s/S save, d/D debug, q/Q or ESC quit

Install
  pip install mediapipe opencv-python numpy pandas

Physics note: With a 2D webcam, metric units require a known‑size object **in frame**.
Keep your hand on the same plane as the card; if you lift it, absolute mm is no longer valid.
"""
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd

# ---------- Constants ----------
CARD_LONG_MM = 85.60
CARD_SHORT_MM = 53.98
ASPECT = CARD_LONG_MM / CARD_SHORT_MM  # ≈ 1.586
ASPECT_TOL = 0.25  # relaxed tolerance
MIN_CARD_AREA_FRAC = 0.002  # accept small card in frame (~0.2% area)
SCALE_PX_PER_MM = 4.0  # rectified plane: 1 mm = 4 px

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_LMS = {
    "Thumb": (1, 2, 3, 4),   # (CMC, MCP, IP, TIP)
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}

@dataclass
class Measure:
    length_mm: Optional[float]
    width_mm: Optional[float]

# ---------- MediaPipe ----------
try:
    import mediapipe as mp
except Exception:
    raise SystemExit("mediapipe is required: pip install mediapipe")

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

# ---------- Geometry helpers ----------
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

def find_card_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    min_area = MIN_CARD_AREA_FRAC * (w * h)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    # Adaptive threshold helps under uneven light
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 19, 2)
    edges = cv2.Canny(thr, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(approx)
        (_, _), (wrect, hrect), _ = rect
        if wrect == 0 or hrect == 0:
            continue
        ratio = max(wrect, hrect) / max(1e-6, min(wrect, hrect))
        if abs(ratio - ASPECT) > ASPECT_TOL:
            continue
        rectangularity = area / (wrect * hrect + 1e-6)
        score = rectangularity * area
        if score > best_score:
            best_score = score
            best = order_quad(approx)
    return best

def compute_homography(card_quad: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    dst_w = int(round(CARD_LONG_MM * SCALE_PX_PER_MM))
    dst_h = int(round(CARD_SHORT_MM * SCALE_PX_PER_MM))
    # Map to landscape by default; orientation doesn't matter for scale
    dst = np.array([[0,0], [dst_w-1,0], [dst_w-1,dst_h-1], [0,dst_h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(card_quad, dst)
    return H, (dst_w, dst_h)

def apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)

# ---------- Measurement logic on rectified plane ----------
class FingerMeasureEngine:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        # Nicer defaults on many laptop cams
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.det = HandDetector(max_hands=1)
        self.rows: List[Dict] = []
        self.window = "Hand Size — Webcam (Card‑scaled)"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.show_debug = True
        self.dbg_name = "Edges Debug"
        cv2.namedWindow(self.dbg_name, cv2.WINDOW_NORMAL)

        # UI buttons (top-right)
        self.btns = {
            'save':  None,
            'debug': None,
            'quit':  None,
        }
        self._pending_action = None
        cv2.setMouseCallback(self.window, self._on_mouse)

    # ---- UI helpers ----
    def _draw_buttons(self, img: np.ndarray):
        h, w = img.shape[:2]
        pad = 10
        bw, bh = 110, 36
        x3, y = w - pad - bw, pad
        x2 = x3 - pad - bw
        x1 = x2 - pad - bw
        # Save
        self.btns['save'] = (x1, y, x1 + bw, y + bh)
        self._draw_btn(img, self.btns['save'], 'Save')
        # Debug
        self.btns['debug'] = (x2, y, x2 + bw, y + bh)
        self._draw_btn(img, self.btns['debug'], 'Debug')
        # Quit
        self.btns['quit'] = (x3, y, x3 + bw, y + bh)
        self._draw_btn(img, self.btns['quit'], 'Quit')

    @staticmethod
    def _draw_btn(img, rect, label):
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        tx = x1 + 10
        ty = y1 + 24
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, rect in self.btns.items():
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == 'save':
                    self._pending_action = 'save'
                elif name == 'debug':
                    self._pending_action = 'debug'
                elif name == 'quit':
                    self._pending_action = 'quit'

    # ---- measurement helpers ----
    def _width_at_mid(self, rect_gray: np.ndarray, p1: np.ndarray, p2: np.ndarray, max_scan_px=100) -> Optional[float]:
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
        print("[INFO] Put a credit/ID card flat in view next to your hand. Keep hand on the same plane.")
        mm_per_px = 1.0 / SCALE_PX_PER_MM
        H = None
        rect_size = None
        self._pending_action = None

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 1) Detect card and compute homography
            quad = find_card_quad(frame)
            if quad is not None:
                H, rect_size = compute_homography(quad)

            overlay = frame.copy()
            status_text = "CARD: OK" if quad is not None else "CARD: NOT FOUND — place a credit/ID card flat near your hand"
            cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
            if quad is not None:
                cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 255), 2)

            measures: Dict[str, Measure] = {}
            # 2) Hand detection (on original)
            results = self.det.detect(frame)
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.det.draw.draw_landmarks(
                        overlay, hand, self.det.mp_hands.HAND_CONNECTIONS,
                        self.det.styles.get_default_hand_landmarks_style(),
                        self.det.styles.get_default_hand_connections_style(),
                    )
                    pts_px = self.det.landmarks_px(hand, w, h)

                    if H is not None and rect_size is not None:
                        # 3) Transform landmarks to rectified plane
                        pts = np.array([pts_px[i] for i in range(21)], dtype=np.float32)
                        pts_rect = apply_H(H, pts)  # in rectified px
                        # 4) Compute length & width per finger in rectified px → mm
                        rect_w, rect_h = rect_size
                        rect_img = cv2.warpPerspective(frame, H, (rect_w, rect_h))
                        rect_gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)

                        for name, (a,b,c,d) in FINGER_LMS.items():
                            p0 = pts_rect[a]; p1 = pts_rect[b]; p2 = pts_rect[c]; p3 = pts_rect[d]
                            L_px = (np.linalg.norm(p1-p0) + np.linalg.norm(p2-p1) + np.linalg.norm(p3-p2))
                            L_mm = float(L_px * mm_per_px)
                            W_px = self._width_at_mid(rect_gray, p1, p2)
                            W_mm = float(W_px * mm_per_px) if W_px is not None else None
                            measures[name] = Measure(L_mm, W_mm)

            # HUD + buttons
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
            self._draw_buttons(overlay)

            cv2.putText(overlay, "Keys: s/S save, d/D debug, q/Q or ESC quit", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(self.window, overlay)

            # Debug window
            if self.show_debug:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
                dbg = cv2.Canny(thr, 40, 140)
                cv2.imshow(self.dbg_name, dbg)
            else:
                try:
                    cv2.destroyWindow(self.dbg_name)
                except Exception:
                    pass

            # --- input handling (case‑insensitive + buttons) ---
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q'), 27) or self._pending_action == 'quit':
                break
            elif k in (ord('s'), ord('S'), 13) or self._pending_action == 'save':
                if measures:
                    self._snapshot(measures)
                else:
                    print("[WARN] No measurements to save yet (card or hand not detected).")
                self._pending_action = None
            elif k in (ord('d'), ord('D')) or self._pending_action == 'debug':
                self.show_debug = not self.show_debug
                self._pending_action = None

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = FingerMeasureEngine(camera_id=0)
    engine.run()

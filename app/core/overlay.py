"""
app/core/overlay.py
Decouples AI results from streaming completely.

The stream runs at full camera FPS.
AI results (boxes, labels) are stored as lightweight draw commands.
The stream stamps those commands onto each raw frame in <1ms.
No waiting for YOLO. No waiting for face recognition.
Stream is always live at camera speed.
"""
import threading
import time
import cv2
import numpy as np
from app.models.threat import ThreatLevel


class OverlayRenderer:
    """
    Stores the latest AI annotation commands.
    Applies them onto any raw frame on demand — extremely fast (<1ms).
    Thread-safe.
    """
    def __init__(self):
        self._lock     = threading.Lock()
        self._commands = []   # list of draw command dicts
        self._hud_info = {}   # fps_mode, fps, cooldown etc.

    def update(self, detections: list, hud_info: dict):
        """Called by AI thread with latest detections."""
        commands = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color     = d.threat.color_bgr
            thickness = 3 if d.threat == ThreatLevel.RED else 2
            name_lbl  = d.face_name if d.is_known else "TRESPASSER"
            top_lbl   = f"{name_lbl}  {d.confidence:.0%}"

            parts = [f"#{d.track_id}", f"loiter:{int(d.loiter_seconds)}s"]
            if d.is_running:   parts.append(f"RUN")
            if d.red_reasons:  parts.append("|".join(d.red_reasons))
            bot_lbl = "  ".join(parts)

            commands.append({
                "bbox":      (x1, y1, x2, y2),
                "color":     color,
                "thickness": thickness,
                "top_lbl":   top_lbl,
                "bot_lbl":   bot_lbl,
                "blur":      d.threat.requires_blur,
            })

        with self._lock:
            self._commands = commands
            self._hud_info = hud_info

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Stamps last known AI results onto frame.
        Called by stream thread — must be fast.
        Returns annotated frame without modifying original.
        """
        with self._lock:
            commands = list(self._commands)
            hud      = dict(self._hud_info)

        if not commands:
            return frame

        out = frame.copy()

        for cmd in commands:
            x1, y1, x2, y2 = cmd["bbox"]
            color           = cmd["color"]

            # Optional SDP blur
            if cmd["blur"]:
                fy2 = min(y1 + (y2-y1)//3, out.shape[0])
                x1c, x2c = max(x1,0), min(x2,out.shape[1])
                if fy2 > y1 and x2c > x1c:
                    k = 25|1
                    out[y1:fy2, x1c:x2c] = cv2.GaussianBlur(
                        out[y1:fy2, x1c:x2c], (k,k), 0)
                    cv2.putText(out, "SDP", (x1+4,y1+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,212,255), 1)

            # Bounding box
            cv2.rectangle(out, (x1,y1), (x2,y2), color, cmd["thickness"])

            # Top label
            lbl = cmd["top_lbl"]
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1,y1-th-10), (x1+tw+6,y1), color, -1)
            cv2.putText(out, lbl, (x1+3,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)

            # Bottom label
            cv2.putText(out, cmd["bot_lbl"], (x1,y2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        return out

    def clear(self):
        with self._lock:
            self._commands = []


# Singleton — shared between AI thread and stream
overlay = OverlayRenderer()
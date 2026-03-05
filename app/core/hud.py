import cv2
import numpy as np
from config.settings import DASHBOARD_PORT

def draw_hud(frame: np.ndarray, fps_mode: str, fps: int,
             cooldown: bool, frame_count: int) -> np.ndarray:
    """
    Renders semi-transparent HUD overlay onto preview window frame.
    Extracted from main so orchestrator stays thin.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0,0), (w,40), (10,17,40), -1)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    mc = (0,80,255)  if fps_mode == "EVIDENCE" else (0,200,100)
    cc = (0,80,255)  if cooldown              else (0,220,0)
    cv2.putText(out, "PROJECT RIO  AI-DVR", (8,27), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0,212,255), 1)
    cv2.putText(out, f"{fps_mode} @ {fps}fps", (w//2-85,27), cv2.FONT_HERSHEY_SIMPLEX, 0.54, mc, 1)
    cv2.putText(out, f"ALERT: {'COOLDOWN' if cooldown else 'READY'}", (w-175,27), cv2.FONT_HERSHEY_SIMPLEX, 0.54, cc, 1)
    cv2.putText(out, f"F#{frame_count}  Q=quit  S=snap  dashboard:localhost:{DASHBOARD_PORT}",
                (6,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,110,150), 1)
    return out

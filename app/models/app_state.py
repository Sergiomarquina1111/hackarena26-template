import threading
from typing import List, Optional
import numpy as np

class AppState:
    """
    Thread-safe singleton shared between hub loop, Flask routes, and MJPEG stream.
    Updated every frame by main.py. Read by the API and stream generator.
    """
    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self.latest_frame:  Optional[np.ndarray] = None
        self.fps_mode       = "MONITOR"
        self.current_fps    = 5
        self.alert_cooldown = False
        self.cooldown_secs  = 0
        self.total_alerts   = 0
        self.total_frames   = 0
        self.active_threats: List[dict] = []
        self.sdp_active     = False
        self.public_url     = ""

    def update(self, frame, detections, fps_mode, fps, cooldown, cooldown_secs) -> None:
        with self._lock:
            self.latest_frame   = frame
            self.active_threats = [d.to_dict() for d in detections]
            self.fps_mode       = fps_mode
            self.current_fps    = fps
            self.alert_cooldown = cooldown
            self.cooldown_secs  = cooldown_secs
            self.total_frames  += 1
            self.sdp_active     = any(d.get("threat") in ("LOW","NONE") for d in self.active_threats)

    def increment_alerts(self) -> None:
        with self._lock: self.total_alerts += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "fps_mode":       self.fps_mode,
                "current_fps":    self.current_fps,
                "alert_cooldown": self.alert_cooldown,
                "cooldown_secs":  self.cooldown_secs,
                "total_alerts":   self.total_alerts,
                "total_frames":   self.total_frames,
                "sdp_active":     self.sdp_active,
                "public_url":     self.public_url,
                "active_threats": list(self.active_threats),
            }

# Module-level singleton
app_state = AppState()

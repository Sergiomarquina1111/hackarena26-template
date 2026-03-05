"""
app/core/motion_analyzer.py
Tracks per-person:
  - Bounding box velocity  → is_running flag
  - Session visit count    → repeat offender flag
  - Ambient brightness     → night mode flag
"""
import time
from collections import defaultdict
from typing import Tuple
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class MotionAnalyzer:
    def __init__(self,
                 speed_threshold: float   = 18.0,   # px/frame to count as running
                 night_threshold: float   = 55.0,   # mean luminance 0-255
                 repeat_threshold: int    = 3,       # visits to flag repeat offender
                 cleanup_secs:    float   = 120.0) -> None:

        self._speed_thresh  = speed_threshold
        self._night_thresh  = night_threshold
        self._repeat_thresh = repeat_threshold
        self._cleanup_secs  = cleanup_secs

        # track_id → {cx, cy, last_seen, visit_count, first_seen}
        self._tracks: dict = defaultdict(lambda: {
            "cx": 0.0, "cy": 0.0,
            "last_seen": 0.0,
            "visit_count": 0,
            "first_seen": time.time(),
        })
        self._last_cleanup = time.time()

        # Cached brightness (updated every ~30 frames by caller)
        self._brightness: float = 128.0

    # ── Public API ────────────────────────────────────────
    def update(self, track_id: int, bbox: Tuple[int,int,int,int]) -> dict:
        """
        Call once per detected person per frame.
        Returns dict: {is_running, visit_count, speed}
        """
        self._prune()
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        now = time.time()

        t = self._tracks[track_id]

        # Speed calculation
        prev_cx, prev_cy = t["cx"], t["cy"]
        speed = ((cx - prev_cx)**2 + (cy - prev_cy)**2) ** 0.5

        # New visit? (track_id reappears after being gone > 30s)
        gap = now - t["last_seen"]
        if gap > 30 and t["visit_count"] > 0:
            t["visit_count"] += 1
            logger.info("Repeat visit detected — track #%d visit #%d", track_id, t["visit_count"])

        if t["visit_count"] == 0:
            t["visit_count"] = 1

        t["cx"]        = cx
        t["cy"]        = cy
        t["last_seen"] = now

        is_running     = speed > self._speed_thresh
        is_repeat      = t["visit_count"] >= self._repeat_thresh

        return {
            "is_running":   is_running,
            "visit_count":  t["visit_count"],
            "speed":        round(speed, 1),
            "is_repeat":    is_repeat,
        }

    def update_brightness(self, frame: np.ndarray) -> float:
        """Call every N frames. Returns mean luminance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._brightness = float(np.mean(gray))
        return self._brightness

    @property
    def is_night(self) -> bool:
        return self._brightness < self._night_thresh

    @property
    def brightness(self) -> float:
        return self._brightness

    def unknown_count(self, detections) -> int:
        """Count of non-known persons in current detections."""
        return sum(1 for d in detections if not d.is_known)

    # ── Private ───────────────────────────────────────────
    def _prune(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self._cleanup_secs:
            return
        stale = [k for k, v in self._tracks.items()
                 if now - v["last_seen"] > self._cleanup_secs]
        for k in stale:
            del self._tracks[k]
        self._last_cleanup = now
        if stale:
            logger.debug("Pruned %d stale motion tracks", len(stale))

"""
app/core/fps_controller.py
Switches between MONITOR (5fps) and EVIDENCE (20fps).

Change: 10-second confirmation delay before switching to EVIDENCE.
A fleeting detection doesn't immediately boost — threat must persist.
"""
import time
from typing import List
from config.settings import FPS_EVIDENCE, FPS_MONITOR, FPS_RELEASE_AFTER
from app.models.detection import Detection
from utils.logger import get_logger

logger = get_logger(__name__)

THREAT_CONFIRM_SECS = 10   # threat must persist this long before EVIDENCE mode


class FPSController:
    """
    Switches between MONITOR (5fps) and EVIDENCE (20fps)
    based on active HIGH/RED/YELLOW threat detections.
    10-second confirmation window prevents false triggers.
    """
    def __init__(self) -> None:
        self._fps            = FPS_MONITOR
        self._mode           = "MONITOR"
        self._last_high      = 0.0
        self._first_threat   = 0.0   # when current threat streak started
        self._in_threat      = False

    def update(self, detections: List[Detection]) -> int:
        has_threat = any(d.threat.requires_alert for d in detections)
        now        = time.time()

        if has_threat:
            if not self._in_threat:
                # Start of a new threat streak
                self._first_threat = now
                self._in_threat    = True

            # Only switch to EVIDENCE after threat persists 10 seconds
            if now - self._first_threat >= THREAT_CONFIRM_SECS:
                self._last_high = now
                self._set("EVIDENCE")

        else:
            self._in_threat = False
            if self._mode == "EVIDENCE":
                if now - self._last_high > FPS_RELEASE_AFTER:
                    self._set("MONITOR")

        return self._fps

    @property
    def current_fps(self) -> int: return self._fps

    @property
    def mode(self) -> str: return self._mode

    def _set(self, mode: str) -> None:
        if mode == self._mode: return
        self._mode = mode
        self._fps  = FPS_EVIDENCE if mode == "EVIDENCE" else FPS_MONITOR
        logger.info("FPS: %s @ %dfps", mode, self._fps)
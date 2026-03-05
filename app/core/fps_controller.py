import time
from typing import List
from config.settings import FPS_EVIDENCE, FPS_MONITOR, FPS_RELEASE_AFTER
from app.models.detection import Detection
from utils.logger import get_logger

logger = get_logger(__name__)

class FPSController:
    """
    Switches between MONITOR (5fps) and EVIDENCE (20fps)
    based on active HIGH threat detections.
    """
    def __init__(self) -> None:
        self._fps       = FPS_MONITOR
        self._mode      = "MONITOR"
        self._last_high = 0.0

    def update(self, detections: List[Detection]) -> int:
        if any(d.threat.requires_alert for d in detections):
            self._last_high = time.time()
            self._set("EVIDENCE")
        elif time.time() - self._last_high > FPS_RELEASE_AFTER:
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

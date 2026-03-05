import queue, cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class FrameSource:
    """
    Unified frame source abstraction.
    Supports webcam/video file AND ESP32-CAM HTTP POST.
    ESP32 frames are pushed onto esp32_queue by the /frame API route.
    """
    def __init__(self, source) -> None:
        self._cap         = cv2.VideoCapture(source)
        self.esp32_queue  = queue.Queue(maxsize=10)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        logger.info("Video source opened: %s", source)

    def read(self):
        """Returns next frame. ESP32 queue takes priority over webcam."""
        if not self.esp32_queue.empty():
            return True, self.esp32_queue.get_nowait()
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()

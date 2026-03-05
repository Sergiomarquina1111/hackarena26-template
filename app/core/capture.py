import queue, cv2
import numpy as np
import threading
import time
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
        self._latest_frame = None
        self._ret = False
        self._stopped = False
        self._lock = threading.Lock()

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        logger.info("Video source opened: %s", source)

        self._is_live = isinstance(source, int) or str(source).startswith("http")
        if self._is_live:
            # Read first frame to ensure it's ready
            self._ret, self._latest_frame = self._cap.read()
            self._thread = threading.Thread(target=self._update, daemon=True, name="CaptureThread")
            self._thread.start()

    def _update(self):
        """Background thread loop that constantly reads frames to prevent buffering."""
        while not self._stopped:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                if ret:
                    self._latest_frame = frame
            if not ret:
                break
            time.sleep(0.001)

    def read(self):
        """Returns next frame. ESP32 queue takes priority over webcam."""
        if not self.esp32_queue.empty():
            return True, self.esp32_queue.get_nowait()

        if self._is_live:
            with self._lock:
                if self._latest_frame is None:
                    return self._ret, None
                return self._ret, self._latest_frame.copy()
        else:
            return self._cap.read()

    def release(self) -> None:
        self._stopped = True
        if self._is_live and hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        self._cap.release()

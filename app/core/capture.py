"""
app/core/capture.py
Threaded frame source — camera reader runs in its own thread.
Latest frame always available instantly, no blocking.
"""
import queue
import threading
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class FrameSource:
    """
    Unified frame source abstraction.
    Camera is read in a background thread — latest frame always ready.
    ESP32 frames pushed via esp32_queue take priority over webcam.
    """
    def __init__(self, source) -> None:
        self._cap        = cv2.VideoCapture(source)
        self.esp32_queue = queue.Queue(maxsize=10)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Reduce internal buffer to 1 — always get freshest frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Latest raw frame — always overwritten, never queued
        self._latest_frame = None
        self._lock         = threading.Lock()
        self._running      = True

        # Start background reader thread
        self._thread = threading.Thread(
            target=self._reader, daemon=True, name="CameraReader"
        )
        self._thread.start()
        logger.info("Video source opened (threaded): %s", source)

    def _reader(self):
        """Continuously reads frames in background. Drops stale frames."""
        while self._running:
            # ESP32 frames take priority
            if not self.esp32_queue.empty():
                try:
                    frame = self.esp32_queue.get_nowait()
                    with self._lock:
                        self._latest_frame = frame
                    continue
                except queue.Empty:
                    pass

            ret, frame = self._cap.read()
            if not ret:
                self._running = False
                break

            with self._lock:
                self._latest_frame = frame  # always overwrite — no backlog

    def read(self):
        """Returns (True, latest_frame) or (False, None) if source ended."""
        if not self._running and self._latest_frame is None:
            return False, None
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return False, None
        return True, frame.copy()

    @property
    def is_running(self):
        return self._running

    def release(self) -> None:
        self._running = False
        self._cap.release()
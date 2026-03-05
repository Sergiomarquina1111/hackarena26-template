import time
import cv2
from app.models.app_state import app_state

def mjpeg_generator():
    """MJPEG frame generator for the /stream endpoint."""
    while True:
        frame = app_state.latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1.0 / max(app_state.current_fps, 1))

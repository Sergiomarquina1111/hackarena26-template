"""
app/api/stream.py
MJPEG streamer with dedicated JPEG encoder thread.

3 concerns fully separated:
  AI thread    → writes raw annotated frame to app_state
  Encoder thread → encodes JPEG in background, stores bytes
  MJPEG generator → just sends pre-encoded bytes, zero CPU
"""
import threading
import time
import cv2
from app.models.app_state import app_state

STREAM_FPS    = 30       # target stream FPS to browser
JPEG_QUALITY  = 70       # lower = faster encode + faster transfer
_encoded_lock = threading.Lock()
_encoded_buf  = None     # pre-encoded JPEG bytes
_encoder_started = False


def _encoder_thread():
    """
    Runs forever in background.
    Encodes latest frame to JPEG as fast as possible.
    Stream generator just reads the bytes — no encoding on hot path.
    """
    global _encoded_buf
    last_frame_id = None

    while True:
        frame = app_state.latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        # Only re-encode if frame actually changed
        frame_id = id(frame)
        if frame_id == last_frame_id:
            time.sleep(0.005)
            continue

        last_frame_id = frame_id

        # Resize to 720p max before encoding — big speed win
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(
                frame,
                (1280, int(h * scale)),
                interpolation=cv2.INTER_LINEAR
            )

        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if ok:
            with _encoded_lock:
                _encoded_buf = buf.tobytes()


def _ensure_encoder():
    global _encoder_started
    if not _encoder_started:
        t = threading.Thread(
            target=_encoder_thread,
            daemon=True,
            name="JPEGEncoder"
        )
        t.start()
        _encoder_started = True


def mjpeg_generator():
    """
    MJPEG frame generator — just sends pre-encoded bytes.
    Zero JPEG encoding on this path. Pure I/O only.
    """
    _ensure_encoder()
    interval  = 1.0 / STREAM_FPS
    last_sent = 0

    while True:
        now  = time.time()
        wait = interval - (now - last_sent)
        if wait > 0:
            time.sleep(wait)

        with _encoded_lock:
            buf = _encoded_buf

        if buf is None:
            time.sleep(0.01)
            continue

        last_sent = time.time()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf
            + b"\r\n"
        )
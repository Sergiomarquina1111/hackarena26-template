"""
app/api/stream.py — ULTRA SMOOTH

3 fully independent concerns:
  1. Raw frame reader  — camera speed (30fps)
  2. Overlay stamper   — stamps last AI boxes onto raw frame (<1ms)
  3. JPEG encoder      — dedicated thread, pre-encodes, stream just sends bytes

Stream latency = camera latency only. AI speed is irrelevant to smoothness.
"""
import threading
import time
import cv2
from app.models.app_state import app_state
from app.core.overlay     import overlay

STREAM_FPS   = 30
JPEG_QUALITY = 65    # 65 is good balance — lower = faster transfer

# Pre-encoded JPEG bytes — updated by encoder thread
_buf_lock = threading.Lock()
_buf      = None
_enc_started = False


def _encoder_thread():
    global _buf
    last_id = None

    while True:
        frame = app_state.raw_frame   # raw frame, no AI wait
        if frame is None:
            time.sleep(0.008)
            continue

        fid = id(frame)
        if fid == last_id:
            time.sleep(0.004)
            continue
        last_id = fid

        # Stamp AI overlay onto raw frame (< 1ms)
        rendered = overlay.render(frame)

        # Downscale to 960px wide max for faster browser transfer
        h, w = rendered.shape[:2]
        if w > 960:
            scale    = 960 / w
            rendered = cv2.resize(
                rendered,
                (960, int(h * scale)),
                interpolation=cv2.INTER_LINEAR
            )

        ok, buf = cv2.imencode(
            ".jpg", rendered,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if ok:
            with _buf_lock:
                _buf = buf.tobytes()


def _start_encoder():
    global _enc_started
    if not _enc_started:
        threading.Thread(
            target=_encoder_thread,
            daemon=True,
            name="JPEGEncoder"
        ).start()
        _enc_started = True


def mjpeg_generator():
    """Pure I/O — just sends pre-encoded bytes. Zero CPU here."""
    _start_encoder()
    interval  = 1.0 / STREAM_FPS
    last_sent = 0

    while True:
        now  = time.time()
        gap  = interval - (now - last_sent)
        if gap > 0:
            time.sleep(gap)

        with _buf_lock:
            buf = _buf

        if buf is None:
            time.sleep(0.005)
            continue

        last_sent = time.time()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf
            + b"\r\n"
        )
#!/usr/bin/env python3
"""
Project Rio | main.py — v2.2 (snapshot-analysis architecture)
Entry point. Wires all layers. No business logic here.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  CAPTURE THREAD  →  ring buffer  →  DISPLAY LOOP    │
  │                          ↓                          │
  │              every SNAPSHOT_INTERVAL secs           │
  │                          ↓                          │
  │           ANALYZER THREAD (background, async)       │
  │                          ↓                          │
  │           last_result (overlaid on live feed)       │
  └─────────────────────────────────────────────────────┘

  Camera NEVER blocks. Analyzer fires every 2 s on a frozen snapshot.
  Overlay from the last analysis is drawn on every live frame.

Usage:
  python main.py                              # webcam (default)
  python main.py --source 1                   # external webcam
  python main.py --source video.mp4           # video file
  python main.py --source http://IP:81/stream # ESP32-CAM MJPEG

Dashboard: http://localhost:5000
"""
import argparse, collections, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from config.settings import (
    DASHBOARD_HOST, DASHBOARD_PORT,
    FPS_EVIDENCE, LOG_FILE, LOG_LEVEL, TELEGRAM_BOT_TOKEN, VIDEO_SOURCE,
)
from app.core.analyzer          import ThreatAnalyzer
from app.core.alert_manager     import AlertManager
from app.core.fps_controller    import FPSController
from app.services.telegram_notifier import TelegramNotifier
from app.services.ngrok_tunnel       import start_tunnel, stop_tunnel
from app.core.capture                import FrameSource
from app.core.hud                    import draw_hud
from app.models.app_state            import app_state
from app.models.threat               import ThreatLevel
from app.api.server                  import create_app
from utils.logger import setup_logging, get_logger

setup_logging(LOG_FILE, LOG_LEVEL)
logger = get_logger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
BUFFER_SECONDS       = 15    # ring-buffer depth for clip saves
SNAPSHOT_INTERVAL    = 2.0   # seconds between analysis snapshots
IDLE_SUSPICIOUS_SECS = 5.0   # idle time before YELLOW upgrade
IDLE_GRID_PX         = 40    # centroid grid cell size for identity matching


# ── Idle tracker ──────────────────────────────────────────────────────────────
class IdleTracker:
    """Tracks how long each person has been roughly stationary."""

    def __init__(self, threshold: float = IDLE_SUSPICIOUS_SECS):
        self.threshold = threshold
        self._tracks: dict = {}   # key → (first_ts, last_ts)

    def _key(self, det) -> str:
        if getattr(det, "face_name", None):
            return f"face:{det.face_name}"
        cx = int(det.bbox[0] + det.bbox[2] / 2) // IDLE_GRID_PX
        cy = int(det.bbox[1] + det.bbox[3] / 2) // IDLE_GRID_PX
        return f"cell:{cx},{cy}"

    def update(self, detections: list) -> list:
        now  = time.monotonic()
        seen = set()
        for det in detections:
            k = self._key(det)
            seen.add(k)
            first, _ = self._tracks.get(k, (now, now))
            self._tracks[k] = (first, now)
            idle = now - first
            det.loiter_seconds = max(getattr(det, "loiter_seconds", 0.0), idle)

        # Evict tracks unseen for > 3 s
        for k in [k for k, (_, last) in list(self._tracks.items())
                  if now - last > 3.0 and k not in seen]:
            del self._tracks[k]

        return detections


# ── Snapshot-driven async analyzer ───────────────────────────────────────────
class SnapshotAnalyzer:
    """
    Fires ThreatAnalyzer on a frozen snapshot every SNAPSHOT_INTERVAL seconds.
    The camera loop is never touched — it runs at full speed independently.

    Public API:
        .submit(frame)   — called by display loop; internally rate-limited
        .latest()        → (annotated_frame | None, detections)
    """

    def __init__(self, analyzer: ThreatAnalyzer, interval: float = SNAPSHOT_INTERVAL):
        self._analyzer     = analyzer
        self._interval     = interval
        self._lock         = threading.Lock()
        self._pending      = None
        self._has_work     = threading.Event()
        self._last_submit  = 0.0
        self._annotated    = None
        self._detections   = []

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="SnapshotAnalyzer"
        )
        self._thread.start()
        logger.info("SnapshotAnalyzer ready — interval=%.1fs", interval)

    def submit(self, frame) -> None:
        """Rate-limited snapshot submission — non-blocking."""
        now = time.monotonic()
        if now - self._last_submit < self._interval:
            return
        self._last_submit = now
        snapshot = frame.copy()           # only copy that ever happens
        with self._lock:
            self._pending = snapshot
        self._has_work.set()
        logger.debug("Snapshot submitted at t=%.2f", now)

    def latest(self):
        """Returns (annotated, detections) instantly — never blocks."""
        with self._lock:
            return self._annotated, list(self._detections)

    def _worker(self):
        while True:
            self._has_work.wait()
            self._has_work.clear()
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            try:
                t0 = time.monotonic()
                annotated, detections = self._analyzer.analyze(frame)
                logger.debug("Analysis done in %.2fs | %d detections",
                             time.monotonic() - t0, len(detections))
                with self._lock:
                    self._annotated  = annotated
                    self._detections = detections
            except Exception:
                logger.exception("SnapshotAnalyzer worker error")

    def stop(self):
        self._has_work.set()   # unblock worker on shutdown


# ── Helpers ───────────────────────────────────────────────────────────────────
def _threat_priority(t: ThreatLevel) -> int:
    return {
        ThreatLevel.RED:    4,
        ThreatLevel.HIGH:   3,
        ThreatLevel.YELLOW: 2,
        ThreatLevel.LOW:    1,
        ThreatLevel.NONE:   0,
    }.get(t, 0)


def _parse_source(raw):
    if raw is None: return VIDEO_SOURCE
    try:    return int(raw)
    except: return raw


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(source) -> None:
    logger.info("Project Rio v2.2 starting | source=%s", source)

    if "YOUR_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram not configured — add tokens to .env")

    notifier     = TelegramNotifier()
    analyzer     = ThreatAnalyzer()
    snap_az      = SnapshotAnalyzer(analyzer, SNAPSHOT_INTERVAL)
    alert_mgr    = AlertManager(notifier)
    fps_ctrl     = FPSController()
    frame_src    = FrameSource(source)
    idle_tracker = IdleTracker(IDLE_SUSPICIOUS_SECS)

    # Ring buffer — shallow refs; deep-copy only when an alert fires
    frame_buf   = collections.deque(maxlen=FPS_EVIDENCE * BUFFER_SECONDS)
    frame_count = 0
    last_tick   = time.time()

    url = start_tunnel()
    if url:
        app_state.public_url = url

    flask_app = create_app(alert_mgr, frame_src)
    threading.Thread(
        target=lambda: flask_app.run(
            host=DASHBOARD_HOST, port=DASHBOARD_PORT,
            threaded=True, use_reloader=False,
        ),
        daemon=True, name="FlaskDashboard",
    ).start()
    logger.info("Dashboard ready: http://localhost:%d", DASHBOARD_PORT)

    last_annotated  = None
    last_detections = []

    logger.info(
        "Hub loop running — snapshot every %.1fs | idle flag after %.0fs.",
        SNAPSHOT_INTERVAL, IDLE_SUSPICIOUS_SECS,
    )

    while True:
        # 1. Pace display to target FPS — never blocked by analyzer
        elapsed = time.time() - last_tick
        wait    = (1.0 / fps_ctrl.current_fps) - elapsed
        if wait > 0:
            time.sleep(wait)
        last_tick = time.time()

        # 2. Grab live frame
        ret, frame = frame_src.read()
        if not ret:
            logger.info("Video source ended.")
            break

        frame_count += 1
        frame_buf.append(frame)           # shallow — fast

        # 3. Submit snapshot every 2 s (returns immediately)
        snap_az.submit(frame)

        # 4. Pull latest analysis result (never blocks)
        ann, dets = snap_az.latest()
        if ann is not None:
            last_annotated  = ann
            last_detections = dets

        display_base = last_annotated if last_annotated is not None else frame

        # 5. Idle tracking — cheap, runs every frame on cached detections
        last_detections = idle_tracker.update(last_detections)

        # 6. Upgrade idle LOW/NONE → YELLOW
        for det in last_detections:
            if (det.threat in (ThreatLevel.LOW, ThreatLevel.NONE)
                    and det.loiter_seconds >= IDLE_SUSPICIOUS_SECS):
                det.threat    = ThreatLevel.YELLOW
                det.loitering = True
                logger.info(
                    "IDLE → YELLOW | %.1fs idle | %s",
                    det.loiter_seconds,
                    getattr(det, "face_name", "unknown"),
                )

        fps_ctrl.update(last_detections)

        # 7. Alert logic
        threats = sorted(last_detections,
                         key=lambda d: _threat_priority(d.threat), reverse=True)

        if threats:
            top = threats[0]

            if top.threat == ThreatLevel.RED:
                if alert_mgr.trigger_alert(
                    "RED", [f.copy() for f in frame_buf],
                    masked      = top.masked,
                    loitering   = top.loitering,
                    loiter_secs = int(top.loiter_seconds),
                    face_name   = top.face_name,
                    is_known    = top.is_known,
                    is_running  = top.is_running,
                    visit_count = top.visit_count,
                    red_reasons = top.red_reasons,
                ):
                    app_state.increment_alerts()

            elif top.threat == ThreatLevel.HIGH:
                if alert_mgr.trigger_alert(
                    "HIGH", [f.copy() for f in frame_buf],
                    masked      = top.masked,
                    loitering   = top.loitering,
                    loiter_secs = int(top.loiter_seconds),
                    face_name   = top.face_name,
                    is_known    = top.is_known,
                ):
                    app_state.increment_alerts()

            elif top.threat == ThreatLevel.YELLOW:
                if alert_mgr.trigger_alert(
                    "YELLOW", [f.copy() for f in frame_buf],
                    face_name   = top.face_name,
                    is_known    = top.is_known,
                    loiter_secs = int(top.loiter_seconds),
                ):
                    app_state.increment_alerts()

        # 8. HUD + single state update
        display = draw_hud(
            display_base,
            fps_ctrl.mode,
            fps_ctrl.current_fps,
            alert_mgr.in_cooldown("HIGH"),
            frame_count,
        )
        app_state.update(
            frame         = display,
            detections    = last_detections,
            fps_mode      = fps_ctrl.mode,
            fps           = fps_ctrl.current_fps,
            cooldown      = alert_mgr.in_cooldown("HIGH"),
            cooldown_secs = alert_mgr.cooldown_remaining("HIGH"),
        )

    snap_az.stop()
    frame_src.release()
    cv2.destroyAllWindows()
    stop_tunnel()
    logger.info("Project Rio stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Rio AI-DVR v2.2")
    parser.add_argument("--source", default=None)
    run(_parse_source(parser.parse_args().source))
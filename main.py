#!/usr/bin/env python3
"""
Project Rio | main.py — v4 ULTRA SMOOTH

Stream = camera speed always (30fps).
AI     = runs as fast as CPU allows, completely independent.
Overlay= last known boxes stamped onto every raw frame in <1ms.

Thread layout:
  CameraReader  (capture.py) — reads raw frames, stores in app_state.raw_frame
  JPEGEncoder   (stream.py)  — raw_frame + overlay → JPEG bytes
  MJPEGStream   (stream.py)  — sends pre-encoded bytes to browser
  AIProcessor   (main.py)    — YOLO+pose+face, updates overlay commands
  AlertDispatch (alert_manager.py) — clips + Telegram, never blocks AI
  FlaskDashboard              — serves routes
"""
import argparse, collections, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from config.settings import (
    DASHBOARD_HOST, DASHBOARD_PORT,
    FPS_EVIDENCE, LOG_FILE, LOG_LEVEL,
    TELEGRAM_BOT_TOKEN, VIDEO_SOURCE,
)
from app.core.analyzer              import ThreatAnalyzer
from app.core.alert_manager         import AlertManager
from app.core.fps_controller        import FPSController
from app.core.overlay               import overlay
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

BUFFER_SECONDS = 15


def run(source) -> None:
    logger.info("Project Rio v4 (Ultra Smooth) | source=%s", source)

    if "YOUR_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram not configured — add tokens to .env")

    notifier  = TelegramNotifier()
    analyzer  = ThreatAnalyzer()
    alert_mgr = AlertManager(notifier)
    fps_ctrl  = FPSController()
    frame_src = FrameSource(source)

    frame_buf = collections.deque(maxlen=FPS_EVIDENCE * BUFFER_SECONDS)

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
    logger.info("Dashboard: http://localhost:%d", DASHBOARD_PORT)

    # ── AI queue — depth 1, always freshest frame ─────────
    ai_queue = collections.deque(maxlen=1)
    ai_stop  = threading.Event()
    frame_count = 0

    def ai_worker():
        """
        Runs as fast as CPU allows.
        Updates overlay commands + app_state detections.
        Never touches the stream — stream reads overlay independently.
        """
        while not ai_stop.is_set():
            if not ai_queue:
                time.sleep(0.001)
                continue

            frame = ai_queue[0]

            try:
                _, detections = analyzer.analyze(frame)
            except Exception as e:
                logger.error("AI error: %s", e)
                continue

            fps_ctrl.update(detections)

            # Update overlay draw commands (< 1ms)
            overlay.update(detections, {
                "fps_mode":     fps_ctrl.mode,
                "fps":          fps_ctrl.current_fps,
                "cooldown":     alert_mgr.in_cooldown("HIGH"),
                "cooldown_secs": alert_mgr.cooldown_remaining("HIGH"),
            })

            # Update dashboard state
            app_state.update(
                frame         = app_state.raw_frame,  # stream handles rendering
                detections    = detections,
                fps_mode      = fps_ctrl.mode,
                fps           = fps_ctrl.current_fps,
                cooldown      = alert_mgr.in_cooldown("HIGH"),
                cooldown_secs = alert_mgr.cooldown_remaining("HIGH"),
            )

            # Alerts
            threats = sorted(
                detections,
                key=lambda d: _threat_priority(d.threat),
                reverse=True,
            )

            if threats:
                top = threats[0]

                if top.threat == ThreatLevel.RED:
                    if alert_mgr.trigger_alert(
                        "RED", list(frame_buf),
                        masked=top.masked, loitering=top.loitering,
                        loiter_secs=int(top.loiter_seconds),
                        face_name=top.face_name, is_known=top.is_known,
                        is_running=top.is_running, visit_count=top.visit_count,
                        red_reasons=top.red_reasons,
                    ):
                        app_state.increment_alerts()

                elif top.threat == ThreatLevel.HIGH:
                    if alert_mgr.trigger_alert(
                        "HIGH", list(frame_buf),
                        masked=top.masked, loitering=top.loitering,
                        loiter_secs=int(top.loiter_seconds),
                        face_name=top.face_name, is_known=top.is_known,
                    ):
                        app_state.increment_alerts()

                elif top.threat == ThreatLevel.YELLOW:
                    if alert_mgr.trigger_alert(
                        "YELLOW", list(frame_buf),
                        face_name=top.face_name, is_known=top.is_known,
                        loiter_secs=int(top.loiter_seconds),
                    ):
                        app_state.increment_alerts()

    threading.Thread(
        target=ai_worker, daemon=True, name="AIProcessor"
    ).start()
    logger.info("AI processor started.")

    # ── Main loop — feed raw frames everywhere ────────────
    logger.info("Streaming. Ctrl+C to stop.")
    try:
        while True:
            ret, frame = frame_src.read()
            if not ret:
                time.sleep(0.005)
                if not frame_src.is_running:
                    break
                continue

            frame_count += 1
            frame_buf.append(frame.copy())

            # ★ KEY: raw frame goes to stream IMMEDIATELY
            # Stream reads this and stamps overlay — no AI wait
            app_state.raw_frame = frame

            # Also feed AI queue (depth 1 = always fresh, drops stale)
            ai_queue.append(frame)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        ai_stop.set()
        frame_src.release()
        stop_tunnel()
        logger.info("Project Rio stopped.")


def _threat_priority(t: ThreatLevel) -> int:
    return {
        ThreatLevel.RED: 4, ThreatLevel.HIGH: 3,
        ThreatLevel.YELLOW: 2, ThreatLevel.LOW: 1, ThreatLevel.NONE: 0,
    }.get(t, 0)


def _parse_source(raw):
    if raw is None: return VIDEO_SOURCE
    try:    return int(raw)
    except: return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    run(_parse_source(parser.parse_args().source))
#!/usr/bin/env python3
"""
Project Rio | main.py — v2
Entry point. Wires all layers. No business logic here.

Usage:
  python main.py                         # webcam (default)
  python main.py --source 1              # external webcam
  python main.py --source video.mp4      # video file
  python main.py --source http://IP:81/stream  # ESP32-CAM MJPEG

Dashboard: http://localhost:5000
"""
import argparse, collections, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from config.settings import (
    CLIP_DURATION_SECONDS, DASHBOARD_HOST, DASHBOARD_PORT,
    FPS_EVIDENCE, LOG_FILE, LOG_LEVEL, TELEGRAM_BOT_TOKEN, VIDEO_SOURCE,
)
from app.core.analyzer        import ThreatAnalyzer
from app.core.alert_manager   import AlertManager
from app.core.fps_controller  import FPSController
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

# How many frames to buffer (at evidence FPS)
BUFFER_SECONDS = 15   # enough for RED 15s clip


def run(source) -> None:
    logger.info("Project Rio v2 starting | source=%s", source)

    if "YOUR_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram not configured — add tokens to .env")

    # ── Initialise layers ────────────────────────────────
    notifier   = TelegramNotifier()
    analyzer   = ThreatAnalyzer()
    alert_mgr  = AlertManager(notifier)
    fps_ctrl   = FPSController()
    frame_src  = FrameSource(source)

    # Frame buffer — large enough for 15s RED clips at evidence FPS
    frame_buf  = collections.deque(maxlen=FPS_EVIDENCE * BUFFER_SECONDS)
    frame_count = 0
    last_tick   = time.time()

    # ── ngrok ────────────────────────────────────────────
    url = start_tunnel()
    if url:
        app_state.public_url = url

    # ── Flask dashboard ──────────────────────────────────
    flask_app = create_app(alert_mgr, frame_src)
    threading.Thread(
        target=lambda: flask_app.run(
            host=DASHBOARD_HOST, port=DASHBOARD_PORT,
            threaded=True, use_reloader=False,
        ),
        daemon=True, name="FlaskDashboard",
    ).start()
    logger.info("Dashboard ready: http://localhost:%d", DASHBOARD_PORT)

    # ── Hub loop ─────────────────────────────────────────
    logger.info("Hub loop running. Q = quit, S = snapshot.")

    while True:
        # ── Pace to target FPS ──
        elapsed = time.time() - last_tick
        target  = 1.0 / fps_ctrl.current_fps
        if elapsed < target:
            time.sleep(target - elapsed)
        last_tick = time.time()

        ret, frame = frame_src.read()
        if not ret:
            logger.info("Video source ended.")
            break

        frame_count += 1
        frame_buf.append(frame.copy())

        annotated, detections = analyzer.analyze(frame)
        fps_ctrl.update(detections)

        # ── Determine highest threat this frame ──
        threats = sorted(detections, key=lambda d: _threat_priority(d.threat), reverse=True)

        if threats:
            top = threats[0]

            # RED alert
            if top.threat == ThreatLevel.RED:
                if alert_mgr.trigger_alert(
                    "RED", list(frame_buf),
                    masked       = top.masked,
                    loitering    = top.loitering,
                    loiter_secs  = int(top.loiter_seconds),
                    face_name    = top.face_name,
                    is_known     = top.is_known,
                    is_running   = top.is_running,
                    visit_count  = top.visit_count,
                    red_reasons  = top.red_reasons,
                ):
                    app_state.increment_alerts()

            # HIGH alert
            elif top.threat == ThreatLevel.HIGH:
                if alert_mgr.trigger_alert(
                    "HIGH", list(frame_buf),
                    masked       = top.masked,
                    loitering    = top.loitering,
                    loiter_secs  = int(top.loiter_seconds),
                    face_name    = top.face_name,
                    is_known     = top.is_known,
                ):
                    app_state.increment_alerts()

            # YELLOW alert — trespasser
            elif top.threat == ThreatLevel.YELLOW:
                if alert_mgr.trigger_alert(
                    "YELLOW", list(frame_buf),
                    face_name    = top.face_name,
                    is_known     = top.is_known,
                    loiter_secs  = int(top.loiter_seconds),
                ):
                    app_state.increment_alerts()

        # ── Update shared app state ──
        app_state.update(
            frame       = annotated,
            detections  = detections,
            fps_mode    = fps_ctrl.mode,
            fps         = fps_ctrl.current_fps,
            cooldown    = alert_mgr.in_cooldown("HIGH"),
            cooldown_secs = alert_mgr.cooldown_remaining("HIGH"),
        )

        # ── HUD (for /stream endpoint) ──
        display = draw_hud(annotated, fps_ctrl.mode, fps_ctrl.current_fps,
                           alert_mgr.in_cooldown("HIGH"), frame_count)
        app_state.update(
            frame         = display,
            detections    = detections,
            fps_mode      = fps_ctrl.mode,
            fps           = fps_ctrl.current_fps,
            cooldown      = alert_mgr.in_cooldown("HIGH"),
            cooldown_secs = alert_mgr.cooldown_remaining("HIGH"),
        )

    frame_src.release()
    cv2.destroyAllWindows()
    stop_tunnel()
    logger.info("Project Rio stopped.")


def _threat_priority(t: ThreatLevel) -> int:
    return {ThreatLevel.RED: 4, ThreatLevel.HIGH: 3,
            ThreatLevel.YELLOW: 2, ThreatLevel.LOW: 1, ThreatLevel.NONE: 0}.get(t, 0)


def _parse_source(raw):
    if raw is None: return VIDEO_SOURCE
    try:    return int(raw)
    except: return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Rio AI-DVR v2")
    parser.add_argument("--source", default=None)
    run(_parse_source(parser.parse_args().source))

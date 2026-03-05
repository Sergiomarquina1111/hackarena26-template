#!/usr/bin/env python3
"""
Project Rio | main.py — v3.1 (Latency-Optimised)

Architecture:
  Thread 1 — CameraReader  : reads raw frames as fast as possible (in capture.py)
  Thread 2 — AIProcessor   : runs YOLO+face on frames, never throttled
  Thread 3 — FlaskDashboard: serves MJPEG stream + API independently

Key fix: FPSController no longer gates the main loop.
It only controls EVIDENCE mode labelling for the dashboard.
AI processes every frame it can — stream stays live and smooth.
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
from app.services.telegram_notifier import TelegramNotifier
from app.services.ngrok_tunnel      import start_tunnel, stop_tunnel
from app.core.capture               import FrameSource
from app.core.hud                   import draw_hud
from app.models.app_state           import app_state
from app.models.threat              import ThreatLevel
from app.api.server                 import create_app
from utils.logger import setup_logging, get_logger

setup_logging(LOG_FILE, LOG_LEVEL)
logger = get_logger(__name__)

BUFFER_SECONDS = 15   # frame buffer depth for RED 15s clips


def run(source) -> None:
    logger.info("Project Rio v3.1 starting | source=%s", source)

    if "YOUR_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram not configured — add tokens to .env")

    # ── Initialise layers ──────────────────────────────────
    notifier  = TelegramNotifier()
    analyzer  = ThreatAnalyzer()
    alert_mgr = AlertManager(notifier)
    fps_ctrl  = FPSController()
    frame_src = FrameSource(source)

    frame_buf   = collections.deque(maxlen=FPS_EVIDENCE * BUFFER_SECONDS)
    frame_count = 0

    # ── ngrok ──────────────────────────────────────────────
    url = start_tunnel()
    if url:
        app_state.public_url = url

    # ── Flask dashboard (Thread 3) ─────────────────────────
    flask_app = create_app(alert_mgr, frame_src)
    threading.Thread(
        target=lambda: flask_app.run(
            host=DASHBOARD_HOST, port=DASHBOARD_PORT,
            threaded=True, use_reloader=False,
        ),
        daemon=True, name="FlaskDashboard",
    ).start()
    logger.info("Dashboard ready: http://localhost:%d", DASHBOARD_PORT)

    # ── AI queue — maxlen=1 means AI always gets the LATEST frame ──
    # If AI is slow, old frames are silently dropped. Stream never waits.
    ai_queue = collections.deque(maxlen=1)
    ai_stop  = threading.Event()

    def ai_worker():
        """
        Thread 2 — AI Processor.
        Runs as fast as CPU allows. No sleep, no throttle.
        Always grabs the freshest frame from ai_queue.
        """
        while not ai_stop.is_set():
            if not ai_queue:
                time.sleep(0.001)
                continue

            frame = ai_queue[0]

            try:
                annotated, detections = analyzer.analyze(frame)
            except Exception as e:
                logger.error("AI error: %s", e)
                continue

            fps_ctrl.update(detections)

            # ── Threat alerts ──────────────────────────────
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

            # ── Push annotated frame to stream ─────────────
            display = draw_hud(
                annotated,
                fps_ctrl.mode,
                fps_ctrl.current_fps,
                alert_mgr.in_cooldown("HIGH"),
                frame_count,
            )
            app_state.update(
                frame         = display,
                detections    = detections,
                fps_mode      = fps_ctrl.mode,
                fps           = fps_ctrl.current_fps,
                cooldown      = alert_mgr.in_cooldown("HIGH"),
                cooldown_secs = alert_mgr.cooldown_remaining("HIGH"),
            )

    ai_thread = threading.Thread(
        target=ai_worker, daemon=True, name="AIProcessor"
    )
    ai_thread.start()
    logger.info("AI processor thread started.")

    # ── Main loop — feed frames into pipeline ──────────────
    # No sleep. No FPS throttle. Read → buffer → push. That's it.
    logger.info("Hub loop running.")

    try:
        while True:
            ret, frame = frame_src.read()
            if not ret:
                time.sleep(0.005)
                if not frame_src.is_running:
                    logger.info("Video source ended.")
                    break
                continue

            frame_count += 1

            # Always buffer raw frame for clip recording
            frame_buf.append(frame.copy())

            # Push to AI — maxlen=1 auto-drops stale frames
            ai_queue.append(frame)

            # Push raw frame to stream immediately so browser
            # always has something even before AI processes it
            if app_state.latest_frame is None:
                app_state.latest_frame = frame

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        ai_stop.set()
        frame_src.release()
        stop_tunnel()
        logger.info("Project Rio stopped.")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Rio AI-DVR v3.1")
    parser.add_argument("--source", default=None)
    run(_parse_source(parser.parse_args().source))
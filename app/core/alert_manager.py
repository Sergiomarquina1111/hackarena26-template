"""
app/core/alert_manager.py
Manages Intelligence Buffer, clip recording, alert dispatch.
- RED    → 15s clip, immediate Telegram with red siren message
- HIGH   → 10s clip, Telegram alert
- YELLOW → 10s clip, Telegram warning (trespasser)
- Per-level cooldowns
"""
import os, queue, subprocess, threading, time
from datetime import datetime
from typing import List
import cv2

from config.settings import (
    ALERT_COOLDOWN_SECONDS, FPS_EVIDENCE, CLIPS_DIR,
)
from app.models.threat import ThreatLevel
from app.services.notifier import AbstractNotifier
from utils.logger import get_logger

logger = get_logger(__name__)

# Clip durations per threat level (seconds)
CLIP_DURATIONS = {
    "RED":    15,
    "HIGH":   10,
    "YELLOW": 10,
    "LOW":    5,
}


class AlertManager:
    def __init__(self, notifier: AbstractNotifier) -> None:
        os.makedirs(CLIPS_DIR, exist_ok=True)
        self._notifier       = notifier
        self._last_alert:    dict       = {}
        self._recent_events: List[dict] = []
        self._queue:         queue.Queue = queue.Queue(maxsize=10)
        threading.Thread(target=self._worker, daemon=True, name="AlertDispatch").start()

    # ── Public ────────────────────────────────────────────
    def trigger_alert(self, threat_level: str, frame_buffer: list,
                      masked:      bool = False,
                      loitering:   bool = False,
                      loiter_secs: int  = 0,
                      face_name:   str  = "UNKNOWN",
                      is_known:    bool = False,
                      is_running:  bool = False,
                      visit_count: int  = 1,
                      red_reasons: list = None) -> bool:

        if self.in_cooldown(threat_level):
            return False

        self._last_alert[threat_level] = time.time()
        clip_secs = CLIP_DURATIONS.get(threat_level, 5)
        clip      = self._save_clip(frame_buffer, threat_level, clip_secs)

        ts    = datetime.now().strftime("%H:%M:%S")
        event = {
            "threat":    threat_level,
            "clip":      clip,
            "timestamp": ts,
            "date":      datetime.now().strftime("%Y-%m-%d"),
            "info": {
                "masked":        masked,
                "loitering":     loitering,
                "loiter_seconds": loiter_secs,
                "face_name":     face_name,
                "is_known":      is_known,
                "is_running":    is_running,
                "visit_count":   visit_count,
                "red_reasons":   red_reasons or [],
            }
        }
        self._recent_events.insert(0, event)
        self._recent_events = self._recent_events[:50]

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Alert queue full — dropping event")

        reasons_str = ", ".join(red_reasons) if red_reasons else ""
        logger.warning("ALERT [%s] face=%s known=%s running=%s reasons=%s",
                        threat_level, face_name, is_known, is_running, reasons_str)
        return True

    def in_cooldown(self, level: str) -> bool:
        return (time.time() - self._last_alert.get(level, 0)) < ALERT_COOLDOWN_SECONDS

    def cooldown_remaining(self, level: str) -> int:
        return max(0, int(ALERT_COOLDOWN_SECONDS -
                          (time.time() - self._last_alert.get(level, 0))))

    def get_recent_events(self) -> List[dict]:
        return self._recent_events

    # ── Clip recording ────────────────────────────────────
    def _save_clip(self, frames: list, level: str, duration_secs: int) -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        avi  = os.path.join(CLIPS_DIR, f"threat_{level}_{ts}.avi")
        mp4  = avi.replace(".avi", ".mp4")

        if not frames:
            return mp4

        h, w  = frames[0].shape[:2]
        n_cap = FPS_EVIDENCE * duration_secs
        writer = cv2.VideoWriter(
            avi,
            cv2.VideoWriter_fourcc(*"XVID"),
            FPS_EVIDENCE, (w, h)
        )
        for f in frames[-n_cap:]:
            writer.write(f)
        writer.release()
        logger.info("Clip saved (avi): %s", avi)

        # Convert to mp4 for Telegram inline playback
        converted = self._to_mp4(avi, mp4)
        return mp4 if converted else avi

    @staticmethod
    def _to_mp4(src: str, dst: str) -> bool:
        """Convert avi → mp4 using ffmpeg if available, else OpenCV fallback."""
        # Try ffmpeg first (best quality)
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", src, "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "28", dst],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                os.remove(src)
                logger.info("Converted to mp4: %s", dst)
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: OpenCV re-encode
        try:
            cap = cv2.VideoCapture(src)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
            cap.release(); out.release()
            os.remove(src)
            logger.info("Converted to mp4 (cv2 fallback): %s", dst)
            return True
        except Exception as e:
            logger.error("mp4 conversion failed: %s", e)
            return False

    # ── Dispatch worker ───────────────────────────────────
    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                info  = job["info"]
                level = job["threat"]

                # Build rich message
                caption = self._build_caption(level, job["date"], job["timestamp"], info)

                self._notifier.send_alert(
                    threat_level = level,
                    timestamp    = f"{job['date']} {job['timestamp']}",
                    clip_path    = job["clip"],
                    masked       = info.get("masked", False),
                    loitering    = info.get("loitering", False),
                    loiter_secs  = info.get("loiter_seconds", 0),
                    caption      = caption,
                )
            except Exception as e:
                logger.error("Dispatch failed: %s", e)
            finally:
                self._queue.task_done()

    @staticmethod
    def _build_caption(level, date, ts, info) -> str:
        icons = {"RED": "🚨🚨🚨", "HIGH": "🚨", "YELLOW": "⚠️", "LOW": "✅"}
        icon  = icons.get(level, "⚠️")

        lines = [
            f"{icon} *PROJECT RIO — {level} ALERT*",
            f"🕐 `{date} {ts}`",
        ]

        face = info.get("face_name", "UNKNOWN")
        if info.get("is_known"):
            lines.append(f"👤 Known: *{face}*")
        else:
            lines.append(f"👤 *TRESPASSER* (unrecognised)")

        if info.get("masked"):      lines.append("🎭 Face concealed")
        if info.get("loitering"):   lines.append(f"⏱ Loitering {info.get('loiter_seconds',0)}s")
        if info.get("is_running"):  lines.append("🏃 Subject running")
        if info.get("visit_count", 1) > 1:
            lines.append(f"🔁 Repeat visit #{info['visit_count']}")

        reasons = info.get("red_reasons", [])
        if reasons:
            lines.append(f"⚡ Triggers: {', '.join(reasons)}")

        if level == "RED":
            lines.append("\n🔴 *IMMEDIATE ACTION REQUIRED*")

        return "\n".join(lines)

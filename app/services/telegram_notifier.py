"""
app/services/telegram_notifier.py
Sends rich Telegram alerts with video clips.
Supports caption parameter from AlertManager.
"""
import os
from app.services.notifier import AbstractNotifier
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import get_logger

logger = get_logger(__name__)

_PLACEHOLDER = {"YOUR_BOT_TOKEN_HERE", "YOUR_CHAT_ID_HERE", ""}

class TelegramNotifier(AbstractNotifier):
    def __init__(self) -> None:
        self._token   = TELEGRAM_BOT_TOKEN
        self._chat_id = TELEGRAM_CHAT_ID
        self._ok      = (self._token  not in _PLACEHOLDER and
                         self._chat_id not in _PLACEHOLDER)
        if not self._ok:
            logger.warning("Telegram not configured — add tokens to .env")

    def send_alert(self, threat_level: str, timestamp: str,
                   clip_path: str = None, masked: bool = False,
                   loitering: bool = False, loiter_secs: int = 0,
                   caption: str = None) -> None:

        if not self._ok:
            logger.info("[Telegram] Not configured — skipping alert: %s @ %s",
                        threat_level, timestamp)
            return

        try:
            import requests
        except ImportError:
            logger.error("requests not installed")
            return

        base = f"https://api.telegram.org/bot{self._token}"

        # Send text message first
        msg = caption or self._default_caption(threat_level, timestamp, masked, loitering, loiter_secs)
        try:
            requests.post(f"{base}/sendMessage", data={
                "chat_id":    self._chat_id,
                "text":       msg,
                "parse_mode": "Markdown",
            }, timeout=10)
        except Exception as e:
            logger.error("[Telegram] sendMessage failed: %s", e)

        # Send video clip
        if clip_path and os.path.exists(clip_path):
            try:
                with open(clip_path, "rb") as vid:
                    ext = os.path.splitext(clip_path)[1].lower()
                    endpoint = "sendVideo" if ext == ".mp4" else "sendDocument"
                    requests.post(f"{base}/{endpoint}", data={
                        "chat_id": self._chat_id,
                        "caption": f"{threat_level} clip",
                    }, files={"video" if ext == ".mp4" else "document": vid}, timeout=60)
                logger.info("[Telegram] Clip sent: %s", clip_path)
            except Exception as e:
                logger.error("[Telegram] sendVideo failed: %s", e)
        else:
            logger.warning("[Telegram] Clip not found: %s", clip_path)

    @staticmethod
    def _default_caption(level, ts, masked, loitering, loiter_secs) -> str:
        lines = [f"🚨 *{level} ALERT* — `{ts}`"]
        if masked:    lines.append("🎭 Face concealed")
        if loitering: lines.append(f"⏱ Loitering {loiter_secs}s")
        return "\n".join(lines)

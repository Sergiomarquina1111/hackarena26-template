import os
from pathlib import Path

_env = Path(__file__).resolve().parents[2] / ".env"
if _env.exists():
    for line in open(_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

def _s(k, d=""):     return os.environ.get(k, d)
def _i(k, d=0):
    try:    return int(os.environ.get(k, d))
    except: return d
def _b(k, d=False):  return os.environ.get(k, str(d)).lower() in ("true","1","yes")
def _f(k, d=0.0):
    try:    return float(os.environ.get(k, d))
    except: return d

# ── Telegram ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN     = _s("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID       = _s("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID_HERE")

# ── ngrok ─────────────────────────────────────────────────
NGROK_AUTHTOKEN        = _s("NGROK_AUTHTOKEN", "YOUR_NGROK_AUTHTOKEN_HERE")
NGROK_ENABLED          = _b("NGROK_ENABLED", True)

# ── Video source ──────────────────────────────────────────
_raw = _s("VIDEO_SOURCE", "0")
try:    VIDEO_SOURCE   = int(_raw)
except: VIDEO_SOURCE   = _raw

# ── Dashboard ─────────────────────────────────────────────
DASHBOARD_HOST         = _s("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT         = _i("DASHBOARD_PORT", 5000)

# ── Detection thresholds ──────────────────────────────────
THREAT_CONFIDENCE      = _f("THREAT_CONFIDENCE", 0.50)
LOITER_SECONDS         = _i("LOITER_SECONDS", 3)
ALERT_COOLDOWN_SECONDS = _i("ALERT_COOLDOWN_SECONDS", 60)
TRIGGER_DISTANCE_CM    = _i("TRIGGER_DISTANCE_CM", 150)

# ── Face recognition ──────────────────────────────────────
KNOWN_FACES_DIR        = _s("KNOWN_FACES_DIR", "known_faces/")

# ── Motion / Red alert thresholds ────────────────────────
SPEED_THRESHOLD             = _f("SPEED_THRESHOLD", 18.0)       # px/frame
NIGHT_BRIGHTNESS_THRESHOLD  = _f("NIGHT_BRIGHTNESS_THRESHOLD", 55.0)  # luminance 0-255
REPEAT_VISIT_THRESHOLD      = _i("REPEAT_VISIT_THRESHOLD", 3)   # visits to flag repeat

# ── FPS & performance ─────────────────────────────────────
FPS_MONITOR            = 5
FPS_EVIDENCE           = 20
FPS_RELEASE_AFTER      = 10
INFER_WIDTH            = 480     # resize longest side to this before YOLO (latency fix)

# ── Models ────────────────────────────────────────────────
YOLO_DETECT_MODEL      = "yolov8n.pt"
YOLO_POSE_MODEL        = "yolov8n-pose.pt"

# ── Privacy / SDP ─────────────────────────────────────────
BLUR_NON_THREATS       = True
GAUSSIAN_BLUR_STRENGTH = 25

# ── Clips ─────────────────────────────────────────────────
CLIP_DURATION_SECONDS  = 5      # default (overridden per threat level)
CLIPS_DIR              = "clips/"

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL              = _s("LOG_LEVEL", "INFO")
LOG_FILE               = "logs/project_rio.log"

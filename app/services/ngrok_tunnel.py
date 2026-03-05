from config.settings import DASHBOARD_PORT, NGROK_AUTHTOKEN, NGROK_ENABLED
from utils.logger import get_logger

logger = get_logger(__name__)

def start_tunnel() -> str:
    """Start ngrok tunnel. Returns public HTTPS URL or empty string."""
    if not NGROK_ENABLED:
        logger.info("ngrok disabled")
        return ""
    if "YOUR_NGROK_AUTHTOKEN" in NGROK_AUTHTOKEN:
        logger.warning("ngrok authtoken not set — add to .env")
        return ""
    try:
        from pyngrok import conf, ngrok
        conf.get_default().auth_token = NGROK_AUTHTOKEN
        tunnel = ngrok.connect(DASHBOARD_PORT, "http")
        url    = tunnel.public_url.replace("http://","https://",1)
        logger.info("=" * 55)
        logger.info("  PUBLIC DASHBOARD: %s", url)
        logger.info("  Share this with judges!")
        logger.info("=" * 55)
        return url
    except ImportError:
        logger.error("pyngrok not installed: pip install pyngrok")
        return ""
    except Exception as e:
        logger.error("ngrok failed: %s", e)
        return ""

def stop_tunnel() -> None:
    try:
        from pyngrok import ngrok
        ngrok.kill()
        logger.info("ngrok tunnel closed")
    except Exception: pass

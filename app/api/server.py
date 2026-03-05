from pathlib import Path
from flask import Flask, Response
from app.api.dashboard import dashboard_bp
from app.api.status       import api_bp
from app.api.ingest    import ingest_bp
from app.api.stream           import mjpeg_generator
from utils.logger import get_logger

logger       = get_logger(__name__)
TEMPLATE_DIR = str(Path(__file__).resolve().parents[1] / "templates")

def create_app(alert_manager, frame_source) -> Flask:
    """
    Flask application factory.
    Registers all blueprints. Injects dependencies via app.config.
    """
    app = Flask(__name__, template_folder=TEMPLATE_DIR)
    app.config["alert_manager"] = alert_manager
    app.config["frame_source"]  = frame_source

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(ingest_bp)

    @app.route("/stream")
    def stream():
        return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    logger.info("Flask app created | templates: %s", TEMPLATE_DIR)
    return app

from flask import Blueprint, jsonify, current_app
from app.models.app_state import app_state
api_bp = Blueprint("api", __name__)

@api_bp.route("/api/status")
def status():
    """Live system snapshot polled by dashboard JS every 800ms."""
    return jsonify(app_state.snapshot())

@api_bp.route("/api/events")
def events():
    """Recent alert log for dashboard event feed."""
    return jsonify(current_app.config["alert_manager"].get_recent_events())

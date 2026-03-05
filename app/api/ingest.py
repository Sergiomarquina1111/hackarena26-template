import queue as Q
import cv2, numpy as np
from flask import Blueprint, current_app, request
from utils.logger import get_logger

logger    = get_logger(__name__)
ingest_bp = Blueprint("ingest", __name__)

@ingest_bp.route("/frame", methods=["POST"])
def receive_frame():
    """
    ESP32-CAM HTTP POST endpoint.
    Accepts raw JPEG bytes + X-Distance-CM header.
    Decoded frame pushed to FrameSource.esp32_queue in main loop.
    """
    data = request.data
    if not data: return "No data", 400
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None: return "Invalid image", 400
    dist = request.headers.get("X-Distance-CM","?")
    cv2.putText(frame, f"HC-SR04: {dist}cm", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,255), 1)
    try: current_app.config["frame_source"].esp32_queue.put_nowait(frame)
    except Q.Full: logger.debug("ESP32 queue full — frame dropped")
    return "OK", 200

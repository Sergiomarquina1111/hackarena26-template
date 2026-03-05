"""
app/core/analyzer.py
Stateful AI pipeline. Optimised for CPU-only systems.

Speed improvements:
  - YOLO on every frame (fast — it's the skip that was causing stutter)
  - Face recognition only on NEW track IDs or every FACE_RECHECK frames
  - Pose only when needed (threat classification)
  - No unnecessary frame copies
"""
import time
from collections import defaultdict
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import (
    BLUR_NON_THREATS, GAUSSIAN_BLUR_STRENGTH,
    LOITER_SECONDS, THREAT_CONFIDENCE,
    YOLO_DETECT_MODEL, YOLO_POSE_MODEL,
    KNOWN_FACES_DIR, INFER_WIDTH,
    SPEED_THRESHOLD, NIGHT_BRIGHTNESS_THRESHOLD, REPEAT_VISIT_THRESHOLD,
)
from app.models.detection   import Detection
from app.models.threat       import ThreatLevel
from app.core.face_recognizer import FaceRecognizer
from app.core.motion_analyzer import MotionAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

IGNORED          = {"cat","dog","bird","horse","sheep","cow","elephant","bear","zebra","giraffe"}
POSE_R           = 150
CLEANUP_S        = 30
FACE_RECHECK     = 30    # re-run face_recognition every N frames for known tracks
                         # (new tracks always get immediate check)
BRIGHTNESS_SKIP  = 20    # update ambient brightness every N frames


class ThreatAnalyzer:
    def __init__(self) -> None:
        logger.info("Loading models: %s + %s", YOLO_DETECT_MODEL, YOLO_POSE_MODEL)
        self._det  = YOLO(YOLO_DETECT_MODEL)
        self._pose = YOLO(YOLO_POSE_MODEL)

        self._hist = defaultdict(lambda: {
            "first": time.time(), "last": time.time(), "frames": 0
        })
        self._last_cleanup = time.time()

        self._face_rec = FaceRecognizer(KNOWN_FACES_DIR)
        # track_id → (name, is_known, last_checked_frame)
        self._face_cache: dict = {}
        # track IDs seen before — new ones get immediate face check
        self._seen_tracks: set = set()

        self._motion = MotionAnalyzer(
            speed_threshold   = SPEED_THRESHOLD,
            night_threshold   = NIGHT_BRIGHTNESS_THRESHOLD,
            repeat_threshold  = REPEAT_VISIT_THRESHOLD,
        )

        self._frame_num = 0
        # Cache last infer scale so boxes stay correct
        self._last_scale = (1.0, 1.0)

    # ── Public ────────────────────────────────────────────
    def analyze(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        self._prune()
        self._frame_num += 1
        annotated  = frame.copy()
        detections: List[Detection] = []

        # Brightness update (cheap, infrequent)
        if self._frame_num % BRIGHTNESS_SKIP == 0:
            self._motion.update_brightness(frame)

        # ── YOLO inference on resized frame ───────────────
        small, sx, sy = self._resize_for_infer(frame)
        self._last_scale = (sx, sy)

        det  = self._det.track(small, persist=True, conf=THREAT_CONFIDENCE, verbose=False)
        pose = self._pose(small, verbose=False)
        poses = self._extract_poses(pose, sx, sy)

        if det[0].boxes is None or len(det[0].boxes) == 0:
            return annotated, detections

        boxes = det[0].boxes
        unknown_count = 0

        for box in boxes:
            label = self._det.names[int(box.cls[0])]
            if label in IGNORED or label != "person":
                continue

            conf = float(box.conf[0])
            tid  = int(box.id[0]) if box.id is not None else -1

            # Scale bbox back to original resolution
            sx2, sy2 = self._last_scale
            coords = box.xyxy[0].tolist()
            x1 = int(coords[0] / sx2)
            y1 = int(coords[1] / sy2)
            x2 = int(coords[2] / sx2)
            y2 = int(coords[3] / sy2)

            # Clamp
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)

            # Loiter tracking
            h = self._hist[tid]
            h["last"] = time.time()
            h["frames"] += 1
            loiter  = h["last"] - h["first"]
            is_loit = loiter > LOITER_SECONDS

            # Pose / mask
            cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
            p        = self._nearest(cx, cy, poses)
            is_masked = (p is not None) and not self._face_visible(p)

            # Motion
            motion      = self._motion.update(tid, (x1, y1, x2, y2))
            is_running  = motion["is_running"]
            visit_count = motion["visit_count"]
            is_repeat   = motion["is_repeat"]

            # ── Smart face recognition ────────────────────
            # New track → immediate check
            # Known track → recheck every FACE_RECHECK frames
            # Unknown track → recheck every FACE_RECHECK frames
            face_name, is_known = self._get_face_smart(frame, tid, (x1, y1, x2, y2))
            if not is_known:
                unknown_count += 1

            # ── Threat classification ─────────────────────
            red_reasons = []
            if not is_known:
                if self._motion.is_night:
                    red_reasons.append("NIGHT INTRUDER")
                if is_running:
                    red_reasons.append("RUNNING")
                if is_repeat:
                    red_reasons.append(f"REPEAT VISIT #{visit_count}")

            threat = self._classify(is_known, is_loit, is_masked, conf, red_reasons)

            if BLUR_NON_THREATS and threat.requires_blur:
                annotated = self._sdp_blur(annotated, x1, y1, x2, y2)

            self._draw(annotated, x1, y1, x2, y2,
                       threat, tid, loiter, conf,
                       face_name, is_known, is_running,
                       motion["speed"], red_reasons)

            detections.append(Detection(
                track_id       = tid,
                threat         = threat,
                loitering      = is_loit,
                masked         = is_masked,
                loiter_seconds = loiter,
                confidence     = conf,
                bbox           = (x1, y1, x2, y2),
                face_name      = face_name,
                is_known       = is_known,
                is_running     = is_running,
                visit_count    = visit_count,
                red_reasons    = red_reasons,
            ))

        # ── Gang detection post-pass ──────────────────────
        if unknown_count >= 2:
            for d in detections:
                if not d.is_known and "GANG DETECTED" not in d.red_reasons:
                    d.red_reasons.append("GANG DETECTED")
                    d.threat = ThreatLevel.RED

        return annotated, detections

    # ── Smart face recognition ────────────────────────────
    def _get_face_smart(self, frame, tid, bbox) -> Tuple[str, bool]:
        cached = self._face_cache.get(tid)
        is_new = tid not in self._seen_tracks
        self._seen_tracks.add(tid)

        if cached:
            name, is_known, last_frame = cached
            # Known person: recheck infrequently
            # Unknown person: recheck more often to catch recognition
            recheck = FACE_RECHECK if is_known else FACE_RECHECK // 2
            if (self._frame_num - last_frame) < recheck:
                return name, is_known

        # Run face recognition
        name, is_known = self._face_rec.recognize(frame, bbox)
        self._face_cache[tid] = (name, is_known, self._frame_num)
        return name, is_known

    # ── Classification ────────────────────────────────────
    @staticmethod
    def _classify(is_known, is_loit, is_masked, conf, red_reasons) -> ThreatLevel:
        if red_reasons:
            return ThreatLevel.RED
        if is_known:
            return ThreatLevel.LOW
        if is_masked or is_loit:
            return ThreatLevel.HIGH
        if conf > 0.5:
            return ThreatLevel.YELLOW
        return ThreatLevel.LOW

    # ── Drawing ───────────────────────────────────────────
    @staticmethod
    def _draw(frame, x1, y1, x2, y2, threat, tid, loiter, conf,
              face_name, is_known, is_running, speed, red_reasons):
        color     = threat.color_bgr
        thickness = 3 if threat == ThreatLevel.RED else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        name_lbl = face_name if is_known else "TRESPASSER"
        top_lbl  = f"{name_lbl}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(top_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, top_lbl, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        parts = [f"#{tid}", f"loiter:{int(loiter)}s"]
        if is_running:   parts.append(f"RUN {speed:.0f}px")
        if red_reasons:  parts.append(" | ".join(red_reasons))
        cv2.putText(frame, "  ".join(parts), (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # ── Pose helpers ──────────────────────────────────────
    @staticmethod
    def _extract_poses(results, sx=1.0, sy=1.0):
        out = []
        for r in results:
            if r.keypoints is None:
                continue
            for kp in r.keypoints.xy:
                pts = kp.cpu().numpy()
                if not len(pts):
                    continue
                scaled = pts.copy()
                scaled[:, 0] /= sx
                scaled[:, 1] /= sy
                out.append({
                    "cx":  float(np.mean(scaled[:, 0])),
                    "cy":  float(np.mean(scaled[:, 1])),
                    "pts": scaled,
                })
        return out

    @staticmethod
    def _nearest(cx, cy, poses):
        if not poses:
            return None
        best, bd = None, float("inf")
        for p in poses:
            d = ((p["cx"] - cx) ** 2 + (p["cy"] - cy) ** 2) ** 0.5
            if d < bd:
                bd, best = d, p
        return best if bd < POSE_R else None

    @staticmethod
    def _face_visible(pose) -> bool:
        pts = pose["pts"]
        return sum(
            1 for i in range(5)
            if i < len(pts) and (pts[i][0] > 0 or pts[i][1] > 0)
        ) >= 3

    @staticmethod
    def _sdp_blur(frame, x1, y1, x2, y2):
        fy2  = min(y1 + (y2 - y1) // 3, frame.shape[0])
        x1c, x2c = max(x1, 0), min(x2, frame.shape[1])
        if fy2 <= y1 or x2c <= x1c:
            return frame
        k = GAUSSIAN_BLUR_STRENGTH | 1
        frame[y1:fy2, x1c:x2c] = cv2.GaussianBlur(
            frame[y1:fy2, x1c:x2c], (k, k), 0
        )
        cv2.putText(frame, "SDP", (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 212, 255), 1)
        return frame

    @staticmethod
    def _resize_for_infer(frame):
        h, w  = frame.shape[:2]
        scale = INFER_WIDTH / max(w, h, 1)
        if scale >= 1.0:
            return frame, 1.0, 1.0
        nw, nh = int(w * scale), int(h * scale)
        small  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return small, nw / w, nh / h

    def _prune(self):
        now = time.time()
        if now - self._last_cleanup < CLEANUP_S:
            return
        stale = [k for k, h in self._hist.items() if now - h["last"] > CLEANUP_S]
        for k in stale:
            del self._hist[k]
            self._face_cache.pop(k, None)
            self._seen_tracks.discard(k)
        self._last_cleanup = now
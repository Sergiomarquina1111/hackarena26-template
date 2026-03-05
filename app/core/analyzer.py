"""
app/core/analyzer.py
Stateful AI pipeline. One instance, call analyze() per frame.
Latency optimisations:
  - YOLO runs on every Nth frame (SKIP_FRAMES), cached otherwise
  - Inference on resized frame, boxes scaled back
  - Face recognition only on unknown persons, every FACE_SKIP frames
"""
import time
from collections import defaultdict
from typing import List, Tuple
import cv2, numpy as np
from ultralytics import YOLO

from config.settings import (
    BLUR_NON_THREATS, GAUSSIAN_BLUR_STRENGTH,
    LOITER_SECONDS, THREAT_CONFIDENCE,
    YOLO_DETECT_MODEL, YOLO_POSE_MODEL,
    KNOWN_FACES_DIR, INFER_WIDTH,
    SPEED_THRESHOLD, NIGHT_BRIGHTNESS_THRESHOLD, REPEAT_VISIT_THRESHOLD,
)
from app.models.detection import Detection
from app.models.threat    import ThreatLevel
from app.core.face_recognizer import FaceRecognizer
from app.core.motion_analyzer import MotionAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

IGNORED        = {"cat","dog","bird","horse","sheep","cow","elephant","bear","zebra","giraffe"}
POSE_R         = 150
CLEANUP_S      = 30
SKIP_FRAMES    = 2      # run YOLO every N frames (latency fix)
FACE_SKIP      = 4      # run face_recognition every N frames per track
BRIGHTNESS_SKIP = 15    # update brightness every N frames


class ThreatAnalyzer:
    def __init__(self) -> None:
        logger.info("Loading models: %s + %s", YOLO_DETECT_MODEL, YOLO_POSE_MODEL)
        self._det  = YOLO(YOLO_DETECT_MODEL)
        self._pose = YOLO(YOLO_POSE_MODEL)

        # Per-track history
        self._hist = defaultdict(lambda: {
            "first": time.time(), "last": time.time(), "frames": 0
        })
        self._last_cleanup = time.time()

        # Face recognition
        self._face_rec  = FaceRecognizer(KNOWN_FACES_DIR)
        self._face_cache: dict = {}     # track_id → (name, is_known, frame_num)

        # Motion analysis
        self._motion = MotionAnalyzer(
            speed_threshold=SPEED_THRESHOLD,
            night_threshold=NIGHT_BRIGHTNESS_THRESHOLD,
            repeat_threshold=REPEAT_VISIT_THRESHOLD,
        )

        # Frame counters
        self._frame_num     = 0
        self._cached_boxes  = []        # last YOLO result
        self._cached_poses  = []        # last pose result

    # ── Public ────────────────────────────────────────────
    def analyze(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        self._prune()
        self._frame_num += 1
        annotated  = frame.copy()
        detections: List[Detection] = []

        # Update brightness periodically
        if self._frame_num % BRIGHTNESS_SKIP == 0:
            self._motion.update_brightness(frame)

        # ── YOLO inference (skipped on intermediate frames) ──
        if self._frame_num % SKIP_FRAMES == 0 or not self._cached_boxes:
            small, sx, sy = self._resize_for_infer(frame)
            det  = self._det.track(small, persist=True, conf=THREAT_CONFIDENCE, verbose=False)
            pose = self._pose(small, verbose=False)
            self._cached_poses = self._extract_poses(pose, sx, sy)
            self._cached_scale = (sx, sy)
            if det[0].boxes is not None:
                self._cached_boxes = det[0].boxes
            else:
                self._cached_boxes = []
                
        boxes = self._cached_boxes
        poses = self._cached_poses

        if not len(boxes):
            return annotated, detections

        # ── Per-detection processing ──
        unknown_this_frame = 0

        for box in boxes:
            label = self._det.names[int(box.cls[0])]
            if label in IGNORED or label != "person":
                continue

            conf     = float(box.conf[0])
            tid      = int(box.id[0]) if box.id is not None else -1
            sx, sy = getattr(self, '_cached_scale', (1.0, 1.0))
            coords = box.xyxy[0].tolist()
            x1 = int(coords[0] / sx)
            y1 = int(coords[1] / sy)
            x2 = int(coords[2] / sx)
            y2 = int(coords[3] / sy)

            # Clamp to frame
            fh, fw = frame.shape[:2]
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(fw,x2), min(fh,y2)

            # Loiter tracking
            h = self._hist[tid]
            h["last"] = time.time(); h["frames"] += 1
            loiter    = h["last"] - h["first"]
            is_loit   = loiter > LOITER_SECONDS

            # Pose / mask
            cx, cy   = (x1+x2)//2, (y1+y2)//2
            p        = self._nearest(cx, cy, poses)
            is_masked = (p is not None) and not self._face_visible(p)

            # Motion
            motion = self._motion.update(tid, (x1,y1,x2,y2))
            is_running  = motion["is_running"]
            visit_count = motion["visit_count"]
            is_repeat   = motion["is_repeat"]

            # Face recognition (cached, runs every FACE_SKIP frames)
            face_name, is_known = self._get_face(frame, tid, (x1,y1,x2,y2))
            if not is_known:
                unknown_this_frame += 1

            # ── Threat classification ──
            red_reasons = []

            if not is_known:
                # Night intruder
                if self._motion.is_night:
                    red_reasons.append("NIGHT INTRUDER")
                # Running trespasser
                if is_running:
                    red_reasons.append("RUNNING")
                # Repeat offender
                if is_repeat:
                    red_reasons.append(f"REPEAT VISIT #{visit_count}")

            # Gang — checked after all boxes processed (see below)
            # We flag it per-detection if unknown_this_frame ≥ 2

            threat = self._classify(
                is_known, is_loit, is_masked, conf, red_reasons
            )

            # SDP blur for known/safe persons
            if BLUR_NON_THREATS and threat.requires_blur:
                annotated = self._sdp_blur(annotated, x1, y1, x2, y2)

            self._draw(annotated, x1, y1, x2, y2,
                       threat, tid, loiter, conf, face_name, is_known,
                       is_running, motion["speed"], red_reasons)

            detections.append(Detection(
                track_id       = tid,
                threat         = threat,
                loitering      = is_loit,
                masked         = is_masked,
                loiter_seconds = loiter,
                confidence     = conf,
                bbox           = (x1,y1,x2,y2),
                face_name      = face_name,
                is_known       = is_known,
                is_running     = is_running,
                visit_count    = visit_count,
                red_reasons    = red_reasons,
            ))

        # ── Gang detection post-pass ──
        # If 2+ unknowns in same frame, upgrade them all to RED
        if unknown_this_frame >= 2:
            for d in detections:
                if not d.is_known and "GANG DETECTED" not in d.red_reasons:
                    d.red_reasons.append("GANG DETECTED")
                    d.threat = ThreatLevel.RED

        return annotated, detections

    # ── Classification ────────────────────────────────────
    @staticmethod
    def _classify(is_known, is_loit, is_masked, conf, red_reasons) -> ThreatLevel:
        if red_reasons:
            return ThreatLevel.RED
        if is_known:
            return ThreatLevel.LOW     # green box
        if is_masked or is_loit:
            return ThreatLevel.HIGH    # red box — masked or lingering unknown
        if conf > 0.5:
            return ThreatLevel.YELLOW  # yellow box — trespasser, just appeared
        return ThreatLevel.LOW

    # ── Face recognition helpers ──────────────────────────
    def _get_face(self, frame, tid, bbox) -> Tuple[str, bool]:
        cached = self._face_cache.get(tid)
        if cached and (self._frame_num - cached[2]) < FACE_SKIP:
            return cached[0], cached[1]
        name, is_known = self._face_rec.recognize(frame, bbox)
        self._face_cache[tid] = (name, is_known, self._frame_num)
        return name, is_known

    # ── Drawing ───────────────────────────────────────────
    @staticmethod
    def _draw(frame, x1,y1,x2,y2, threat, tid, loiter, conf,
              face_name, is_known, is_running, speed, red_reasons):
        color = threat.color_bgr
        thickness = 3 if threat == ThreatLevel.RED else 2

        # Box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

        # Name / status label
        if is_known:
            name_lbl = face_name
        else:
            name_lbl = "TRESPASSER"

        top_lbl = f"{name_lbl}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(top_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, top_lbl, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)

        # Bottom sub-label
        parts = [f"#{tid}", f"loiter:{int(loiter)}s"]
        if is_running: parts.append(f"RUN {speed:.0f}px")
        if red_reasons: parts.append(" | ".join(red_reasons))
        bot_lbl = "  ".join(parts)
        cv2.putText(frame, bot_lbl, (x1, y2+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # ── Pose helpers ──────────────────────────────────────
    @staticmethod
    def _extract_poses(results, sx=1.0, sy=1.0):
        out = []
        for r in results:
            if r.keypoints is None: continue
            for kp in r.keypoints.xy:
                pts = kp.cpu().numpy()
                if not len(pts): continue
                # Scale back to original frame coords
                scaled = pts.copy()
                scaled[:,0] /= sx
                scaled[:,1] /= sy
                out.append({
                    "cx": float(np.mean(scaled[:,0])),
                    "cy": float(np.mean(scaled[:,1])),
                    "pts": scaled
                })
        return out

    @staticmethod
    def _nearest(cx, cy, poses):
        if not poses: return None
        best, bd = None, float("inf")
        for p in poses:
            d = ((p["cx"]-cx)**2+(p["cy"]-cy)**2)**0.5
            if d < bd: bd, best = d, p
        return best if bd < POSE_R else None

    @staticmethod
    def _face_visible(pose) -> bool:
        pts = pose["pts"]
        return sum(1 for i in range(5)
                   if i < len(pts) and (pts[i][0] > 0 or pts[i][1] > 0)) >= 3

    @staticmethod
    def _sdp_blur(frame, x1, y1, x2, y2):
        fy2  = min(y1 + (y2-y1)//3, frame.shape[0])
        x1c, x2c = max(x1,0), min(x2, frame.shape[1])
        if fy2 <= y1 or x2c <= x1c: return frame
        k = GAUSSIAN_BLUR_STRENGTH | 1
        frame[y1:fy2, x1c:x2c] = cv2.GaussianBlur(frame[y1:fy2, x1c:x2c], (k,k), 0)
        cv2.putText(frame, "SDP", (x1+4, y1+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,212,255), 1)
        return frame

    # ── Resize for inference ──────────────────────────────
    @staticmethod
    def _resize_for_infer(frame):
        """Resize frame so longest side = INFER_WIDTH. Returns (small, sx, sy)."""
        h, w = frame.shape[:2]
        scale = INFER_WIDTH / max(w, h, 1)
        if scale >= 1.0:
            return frame, 1.0, 1.0
        nw, nh = int(w*scale), int(h*scale)
        small  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return small, nw/w, nh/h

    def _prune(self):
        now = time.time()
        if now - self._last_cleanup < CLEANUP_S: return
        for t in [k for k,h in self._hist.items() if now-h["last"]>CLEANUP_S]:
            del self._hist[t]
            self._face_cache.pop(t, None)
        self._last_cleanup = now

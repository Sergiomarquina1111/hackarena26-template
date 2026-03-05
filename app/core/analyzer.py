"""
app/core/analyzer.py — v5 MAXIMUM PERFORMANCE

Every possible CPU saving applied:
  1. INFER_WIDTH = 256px  — fastest YOLO inference on CPU
  2. YOLO predict() not track() — predict is faster, we do our own IoU tracker
  3. Pose runs every POSE_EVERY frames only, result cached between runs  
  4. Face recognition fully async in own thread — YOLO never waits for dlib
  5. No unnecessary frame.copy() — annotated drawn on single copy only
  6. Pose skipped entirely when no persons detected
"""
import queue
import threading
import time
from collections import defaultdict
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import (
    BLUR_NON_THREATS, GAUSSIAN_BLUR_STRENGTH,
    LOITER_SECONDS, THREAT_CONFIDENCE,
    YOLO_DETECT_MODEL, YOLO_POSE_MODEL,
    KNOWN_FACES_DIR,
    SPEED_THRESHOLD, NIGHT_BRIGHTNESS_THRESHOLD, REPEAT_VISIT_THRESHOLD,
)
from app.models.detection    import Detection
from app.models.threat        import ThreatLevel
from app.core.face_recognizer import FaceRecognizer
from app.core.motion_analyzer import MotionAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

IGNORED         = {"cat","dog","bird","horse","sheep","cow",
                   "elephant","bear","zebra","giraffe"}
POSE_R          = 150
CLEANUP_S       = 30
INFER_WIDTH     = 256    # ← 256 is the sweet spot for speed vs accuracy on CPU
POSE_EVERY      = 4      # run pose model every N frames, cache in between
FACE_RECHECK    = 20
BRIGHTNESS_SKIP = 25
IOU_THRESHOLD   = 0.3    # IoU threshold for our lightweight tracker


class _SimpleTracker:
    """
    Lightweight IoU-based tracker.
    Replaces YOLO's built-in tracker (which adds ~15ms overhead per frame).
    Assigns stable IDs by matching boxes frame-to-frame via IoU.
    """
    def __init__(self):
        self._tracks: dict = {}   # id → bbox (x1,y1,x2,y2)
        self._next_id = 1

    def update(self, boxes: list) -> list:
        """
        boxes: list of (x1,y1,x2,y2,conf)
        returns: list of (x1,y1,x2,y2,conf,track_id)
        """
        if not self._tracks:
            result = []
            for b in boxes:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = b[:4]
                result.append((*b, tid))
            return result

        assigned   = {}   # tid → box
        unmatched  = list(boxes)
        used_tids  = set()

        for box in boxes:
            best_iou, best_tid = 0, None
            for tid, prev in self._tracks.items():
                if tid in used_tids: continue
                iou = _iou(box[:4], prev)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_tid is not None and best_iou >= IOU_THRESHOLD:
                assigned[best_tid] = box
                used_tids.add(best_tid)
            else:
                tid = self._next_id; self._next_id += 1
                assigned[tid] = box

        self._tracks = {tid: b[:4] for tid, b in assigned.items()}
        return [(*b, tid) for tid, b in assigned.items()]


def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


class ThreatAnalyzer:
    def __init__(self) -> None:
        logger.info("Loading models: %s + %s", YOLO_DETECT_MODEL, YOLO_POSE_MODEL)
        self._det  = YOLO(YOLO_DETECT_MODEL)
        self._pose = YOLO(YOLO_POSE_MODEL)

        # Warm up models so first-frame cost is paid at startup
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        self._det.predict(dummy, verbose=False)
        self._pose.predict(dummy, verbose=False)
        logger.info("Models warmed up")

        self._tracker = _SimpleTracker()

        self._hist = defaultdict(lambda: {
            "first": time.time(), "last": time.time(), "frames": 0
        })
        self._last_cleanup = time.time()

        self._motion = MotionAnalyzer(
            speed_threshold  = SPEED_THRESHOLD,
            night_threshold  = NIGHT_BRIGHTNESS_THRESHOLD,
            repeat_threshold = REPEAT_VISIT_THRESHOLD,
        )

        self._frame_num  = 0
        self._last_scale = (1.0, 1.0)

        # Cached pose results — reused between POSE_EVERY frames
        self._cached_poses: list = []
        self._last_pose_frame = -99

        # Async face recognition
        self._face_rec      = FaceRecognizer(KNOWN_FACES_DIR)
        self._face_cache:   dict = {}
        self._seen_tracks:  set  = set()
        self._face_job_q    = queue.Queue(maxsize=3)
        self._face_result_q = queue.Queue(maxsize=20)

        threading.Thread(
            target=self._face_worker, daemon=True, name="FaceRecognition"
        ).start()
        logger.info("Async face recognition thread started")

    # ── Face worker ───────────────────────────────────────
    def _face_worker(self):
        while True:
            try:
                tid, frame, bbox = self._face_job_q.get(timeout=1.0)
                name, is_known   = self._face_rec.recognize(frame, bbox)
                try:
                    self._face_result_q.put_nowait((tid, name, is_known))
                except queue.Full:
                    pass
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Face worker: %s", e)

    def _flush_face_results(self):
        while not self._face_result_q.empty():
            try:
                tid, name, is_known = self._face_result_q.get_nowait()
                self._face_cache[tid] = (name, is_known, self._frame_num)
            except queue.Empty:
                break

    def _request_face(self, tid, frame, bbox):
        cached = self._face_cache.get(tid)
        self._seen_tracks.add(tid)
        if cached:
            _, is_known, last_f = cached
            recheck = FACE_RECHECK if is_known else FACE_RECHECK // 2
            if (self._frame_num - last_f) < recheck:
                return
        try:
            self._face_job_q.put_nowait((tid, frame, bbox))
        except queue.Full:
            pass

    def _get_cached_face(self, tid) -> Tuple[str, bool]:
        c = self._face_cache.get(tid)
        return (c[0], c[1]) if c else ("TRESPASSER", False)

    # ── Main analyze ──────────────────────────────────────
    def analyze(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        self._prune()
        self._frame_num += 1
        self._flush_face_results()

        if self._frame_num % BRIGHTNESS_SKIP == 0:
            self._motion.update_brightness(frame)

        # Single frame copy for drawing
        annotated  = frame.copy()
        detections: List[Detection] = []
        fh, fw = frame.shape[:2]

        # ── Resize for inference ──────────────────────────
        scale = INFER_WIDTH / max(fw, fh, 1)
        if scale < 1.0:
            nw, nh = int(fw * scale), int(fh * scale)
            small  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            sx, sy = nw / fw, nh / fh
        else:
            small, sx, sy = frame, 1.0, 1.0
        self._last_scale = (sx, sy)

        # ── YOLO detect (predict is faster than track) ────
        det = self._det.predict(small, conf=THREAT_CONFIDENCE, verbose=False)

        if det[0].boxes is None or len(det[0].boxes) == 0:
            self._cached_poses = []
            return annotated, detections

        # Build raw box list for our tracker
        raw_boxes = []
        for box in det[0].boxes:
            label = self._det.names[int(box.cls[0])]
            if label in IGNORED or label != "person":
                continue
            conf   = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            # Scale back to original resolution immediately
            x1 = int(coords[0] / sx); y1 = int(coords[1] / sy)
            x2 = int(coords[2] / sx); y2 = int(coords[3] / sy)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(fw,x2), min(fh,y2)
            raw_boxes.append((x1, y1, x2, y2, conf))

        if not raw_boxes:
            return annotated, detections

        tracked = self._tracker.update(raw_boxes)

        # ── Pose — only every POSE_EVERY frames ──────────
        if (self._frame_num - self._last_pose_frame) >= POSE_EVERY:
            pose_res = self._pose.predict(small, verbose=False)
            self._cached_poses = self._extract_poses(pose_res, sx, sy)
            self._last_pose_frame = self._frame_num
        poses = self._cached_poses

        unknown_count = 0

        for (x1, y1, x2, y2, conf, tid) in tracked:
            # Loiter
            h = self._hist[tid]
            h["last"] = time.time(); h["frames"] += 1
            loiter  = h["last"] - h["first"]
            is_loit = loiter > LOITER_SECONDS

            # Pose / mask
            cx, cy    = (x1+x2)//2, (y1+y2)//2
            p         = self._nearest(cx, cy, poses)
            is_masked = (p is not None) and not self._face_visible(p)

            # Motion
            motion      = self._motion.update(tid, (x1,y1,x2,y2))
            is_running  = motion["is_running"]
            visit_count = motion["visit_count"]
            is_repeat   = motion["is_repeat"]

            # Face (async)
            self._request_face(tid, frame, (x1,y1,x2,y2))
            face_name, is_known = self._get_cached_face(tid)
            if not is_known:
                unknown_count += 1

            # Classify
            red_reasons = []
            if not is_known:
                if self._motion.is_night:   red_reasons.append("NIGHT INTRUDER")
                if is_running:              red_reasons.append("RUNNING")
                if is_repeat:               red_reasons.append(f"REPEAT VISIT #{visit_count}")

            threat = self._classify(is_known, is_loit, is_masked, conf, red_reasons)

            if BLUR_NON_THREATS and threat.requires_blur:
                annotated = self._sdp_blur(annotated, x1,y1,x2,y2)

            self._draw(annotated, x1,y1,x2,y2,
                       threat, tid, loiter, conf,
                       face_name, is_known, is_running,
                       motion["speed"], red_reasons)

            detections.append(Detection(
                track_id=tid, threat=threat,
                loitering=is_loit, masked=is_masked,
                loiter_seconds=loiter, confidence=conf,
                bbox=(x1,y1,x2,y2),
                face_name=face_name, is_known=is_known,
                is_running=is_running, visit_count=visit_count,
                red_reasons=red_reasons,
            ))

        # Gang check
        if unknown_count >= 2:
            for d in detections:
                if not d.is_known and "GANG DETECTED" not in d.red_reasons:
                    d.red_reasons.append("GANG DETECTED")
                    d.threat = ThreatLevel.RED

        return annotated, detections

    # ── Static helpers ────────────────────────────────────
    @staticmethod
    def _classify(is_known, is_loit, is_masked, conf, red_reasons) -> ThreatLevel:
        if red_reasons:          return ThreatLevel.RED
        if is_known:             return ThreatLevel.LOW
        if is_masked or is_loit: return ThreatLevel.HIGH
        if conf > 0.5:           return ThreatLevel.YELLOW
        return ThreatLevel.LOW

    @staticmethod
    def _draw(frame, x1,y1,x2,y2, threat, tid, loiter, conf,
              face_name, is_known, is_running, speed, red_reasons):
        color = threat.color_bgr
        cv2.rectangle(frame, (x1,y1), (x2,y2), color,
                      3 if threat == ThreatLevel.RED else 2)
        lbl = f"{'TRESPASSER' if not is_known else face_name}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1,y1-th-10), (x1+tw+6,y1), color, -1)
        cv2.putText(frame, lbl, (x1+3,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
        parts = [f"#{tid}", f"loiter:{int(loiter)}s"]
        if is_running:  parts.append(f"RUN {speed:.0f}px")
        if red_reasons: parts.append(" | ".join(red_reasons))
        cv2.putText(frame, "  ".join(parts), (x1,y2+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    @staticmethod
    def _extract_poses(results, sx=1.0, sy=1.0):
        out = []
        for r in results:
            if r.keypoints is None: continue
            for kp in r.keypoints.xy:
                pts = kp.cpu().numpy()
                if not len(pts): continue
                sc = pts.copy()
                sc[:,0] /= sx; sc[:,1] /= sy
                out.append({"cx": float(np.mean(sc[:,0])),
                            "cy": float(np.mean(sc[:,1])), "pts": sc})
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
                   if i < len(pts) and (pts[i][0]>0 or pts[i][1]>0)) >= 3

    @staticmethod
    def _sdp_blur(frame, x1,y1,x2,y2):
        fy2 = min(y1+(y2-y1)//3, frame.shape[0])
        x1c,x2c = max(x1,0), min(x2,frame.shape[1])
        if fy2<=y1 or x2c<=x1c: return frame
        k = GAUSSIAN_BLUR_STRENGTH|1
        frame[y1:fy2,x1c:x2c] = cv2.GaussianBlur(
            frame[y1:fy2,x1c:x2c],(k,k),0)
        cv2.putText(frame,"SDP",(x1+4,y1+16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,212,255),1)
        return frame

    def _prune(self):
        now = time.time()
        if now-self._last_cleanup < CLEANUP_S: return
        stale = [k for k,h in self._hist.items() if now-h["last"]>CLEANUP_S]
        for k in stale:
            del self._hist[k]
            self._face_cache.pop(k,None)
            self._seen_tracks.discard(k)
        self._last_cleanup = now
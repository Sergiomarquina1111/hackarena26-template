"""
app/core/analyzer.py — v5.1 Behavioural Refinements

Subtle changes only:
  1. Walking-through detection  — person steadily moving = no threat
  2. Weapon detection           — YOLO knife/scissors/bat = immediate RED
  3. Staring detection          — stationary + face visible > STARE_SECS = HIGH
  4. Masks remain HIGH
  5. Normal bags/walking = ignored
"""
import queue
import threading
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
    KNOWN_FACES_DIR,
    SPEED_THRESHOLD, NIGHT_BRIGHTNESS_THRESHOLD, REPEAT_VISIT_THRESHOLD,
)
from app.models.detection    import Detection
from app.models.threat        import ThreatLevel
from app.core.face_recognizer import FaceRecognizer
from app.core.motion_analyzer import MotionAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────
IGNORED         = {"cat","dog","bird","horse","sheep","cow",
                   "elephant","bear","zebra","giraffe"}

# YOLO classes that are immediately suspicious
WEAPONS         = {"knife", "scissors", "baseball bat"}

# These objects alone are NOT suspicious — person carrying them is normal
BENIGN_OBJECTS  = {"handbag", "backpack", "suitcase", "umbrella",
                   "bottle", "cup", "cell phone", "book"}

POSE_R          = 150
CLEANUP_S       = 30
INFER_WIDTH     = 256
POSE_EVERY      = 4
FACE_RECHECK    = 20
BRIGHTNESS_SKIP = 25
IOU_THRESHOLD   = 0.3

# Staring: face visible + barely moving for this many seconds
STARE_SECS      = 6.0
# Walking through: if average speed > this, person is just passing
WALK_SPEED_PX   = 8.0    # px/frame — below this = standing/loitering
# Loiter only triggers if person is slow/stationary
LOITER_MIN_SPEED = 6.0   # must be slower than this to count as loitering


class _SimpleTracker:
    """Lightweight IoU tracker — replaces YOLO built-in tracker."""
    def __init__(self):
        self._tracks  = {}
        self._next_id = 1

    def update(self, boxes):
        if not self._tracks:
            result = []
            for b in boxes:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = b[:4]
                result.append((*b, tid))
            return result

        assigned  = {}
        used_tids = set()

        for box in boxes:
            best_iou, best_tid = 0, None
            for tid, prev in self._tracks.items():
                if tid in used_tids: continue
                iou = _iou(box[:4], prev)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid and best_iou >= IOU_THRESHOLD:
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
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0


class ThreatAnalyzer:
    def __init__(self) -> None:
        logger.info("Loading models: %s + %s", YOLO_DETECT_MODEL, YOLO_POSE_MODEL)
        self._det  = YOLO(YOLO_DETECT_MODEL)
        self._pose = YOLO(YOLO_POSE_MODEL)

        # Warm up
        dummy = np.zeros((256,256,3), dtype=np.uint8)
        self._det.predict(dummy, verbose=False)
        self._pose.predict(dummy, verbose=False)
        logger.info("Models warmed up")

        self._tracker = _SimpleTracker()

        self._hist = defaultdict(lambda: {
            "first": time.time(), "last": time.time(), "frames": 0,
            "speed_sum": 0.0,    # cumulative speed for average
            "stare_start": None, # when staring began
        })
        self._last_cleanup = time.time()

        self._motion = MotionAnalyzer(
            speed_threshold  = SPEED_THRESHOLD,
            night_threshold  = NIGHT_BRIGHTNESS_THRESHOLD,
            repeat_threshold = REPEAT_VISIT_THRESHOLD,
        )

        self._frame_num   = 0
        self._last_scale  = (1.0, 1.0)
        self._cached_poses    = []
        self._last_pose_frame = -99

        # Weapon detections this frame (set of labels found near persons)
        self._weapon_labels: set = set()

        # Async face recognition
        self._face_rec      = FaceRecognizer(KNOWN_FACES_DIR)
        self._face_cache:   dict = {}
        self._seen_tracks:  set  = set()
        self._face_job_q    = queue.Queue(maxsize=3)
        self._face_result_q = queue.Queue(maxsize=20)
        threading.Thread(target=self._face_worker, daemon=True,
                         name="FaceRecognition").start()
        logger.info("Async face recognition started")

    # ── Face worker ───────────────────────────────────────
    def _face_worker(self):
        while True:
            try:
                tid, frame, bbox = self._face_job_q.get(timeout=1.0)
                name, is_known   = self._face_rec.recognize(frame, bbox)
                try: self._face_result_q.put_nowait((tid, name, is_known))
                except queue.Full: pass
            except queue.Empty: continue
            except Exception as e: logger.error("Face worker: %s", e)

    def _flush_face_results(self):
        while not self._face_result_q.empty():
            try:
                tid, name, is_known = self._face_result_q.get_nowait()
                self._face_cache[tid] = (name, is_known, self._frame_num)
            except queue.Empty: break

    def _request_face(self, tid, frame, bbox):
        cached = self._face_cache.get(tid)
        self._seen_tracks.add(tid)
        if cached:
            _, is_known, last_f = cached
            recheck = FACE_RECHECK if is_known else FACE_RECHECK//2
            if (self._frame_num - last_f) < recheck: return
        try: self._face_job_q.put_nowait((tid, frame, bbox))
        except queue.Full: pass

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

        annotated  = frame.copy()
        detections: List[Detection] = []
        fh, fw = frame.shape[:2]

        # Resize for inference
        scale = INFER_WIDTH / max(fw, fh, 1)
        if scale < 1.0:
            nw, nh = int(fw*scale), int(fh*scale)
            small  = cv2.resize(frame, (nw,nh), interpolation=cv2.INTER_LINEAR)
            sx, sy = nw/fw, nh/fh
        else:
            small, sx, sy = frame, 1.0, 1.0
        self._last_scale = (sx, sy)

        det = self._det.predict(small, conf=THREAT_CONFIDENCE, verbose=False)

        if det[0].boxes is None or len(det[0].boxes) == 0:
            self._cached_poses  = []
            self._weapon_labels = set()
            return annotated, detections

        boxes = det[0].boxes

        # ── Collect weapons in scene ──────────────────────
        self._weapon_labels = set()
        for box in boxes:
            lbl = self._det.names[int(box.cls[0])]
            if lbl in WEAPONS:
                self._weapon_labels.add(lbl)

        # ── Person boxes only ─────────────────────────────
        person_raw = []
        for box in boxes:
            lbl = self._det.names[int(box.cls[0])]
            if lbl in IGNORED or lbl != "person": continue
            conf   = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            x1 = int(coords[0]/sx); y1 = int(coords[1]/sy)
            x2 = int(coords[2]/sx); y2 = int(coords[3]/sy)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(fw,x2), min(fh,y2)
            person_raw.append((x1,y1,x2,y2,conf))

        if not person_raw:
            return annotated, detections

        tracked = self._tracker.update(person_raw)

        # Pose every POSE_EVERY frames
        if (self._frame_num - self._last_pose_frame) >= POSE_EVERY:
            pose_res = self._pose.predict(small, verbose=False)
            self._cached_poses    = self._extract_poses(pose_res, sx, sy)
            self._last_pose_frame = self._frame_num
        poses = self._cached_poses

        unknown_count = 0

        for (x1, y1, x2, y2, conf, tid) in tracked:
            h = self._hist[tid]
            h["last"] = time.time(); h["frames"] += 1
            loiter    = h["last"] - h["first"]

            # Motion
            motion      = self._motion.update(tid, (x1,y1,x2,y2))
            is_running  = motion["is_running"]
            visit_count = motion["visit_count"]
            is_repeat   = motion["is_repeat"]
            speed       = motion["speed"]

            # ── Walking-through check ─────────────────────
            # Accumulate speed average
            h["speed_sum"] += speed
            avg_speed = h["speed_sum"] / h["frames"]

            # Person is "just walking" if:
            #   - moving steadily (avg speed above walk threshold)
            #   - hasn't been here long enough to be suspicious
            just_walking = (avg_speed > WALK_SPEED_PX
                            and loiter < LOITER_SECONDS * 1.5)

            # Loiter only counts if person is slow/stationary
            is_loit = (loiter > LOITER_SECONDS
                       and avg_speed < LOITER_MIN_SPEED)

            # Pose / mask
            cx, cy    = (x1+x2)//2, (y1+y2)//2
            p         = self._nearest(cx, cy, poses)
            is_masked = (p is not None) and not self._face_visible(p)
            face_vis  = (p is not None) and self._face_visible(p)

            # ── Staring detection ─────────────────────────
            # Face clearly visible + person barely moving for STARE_SECS
            is_staring = False
            if face_vis and speed < WALK_SPEED_PX:
                if h["stare_start"] is None:
                    h["stare_start"] = time.time()
                elif time.time() - h["stare_start"] >= STARE_SECS:
                    is_staring = True
            else:
                h["stare_start"] = None   # reset if they move

            # Face recognition
            self._request_face(tid, frame, (x1,y1,x2,y2))
            face_name, is_known = self._get_cached_face(tid)
            if not is_known:
                unknown_count += 1

            # ── Classify ──────────────────────────────────
            red_reasons = []

            if self._weapon_labels:
                # Weapon in scene — flag nearest person
                red_reasons.append(
                    f"WEAPON: {', '.join(self._weapon_labels).upper()}"
                )

            if not is_known:
                if self._motion.is_night:
                    red_reasons.append("NIGHT INTRUDER")
                if is_running:
                    red_reasons.append("RUNNING")
                if is_repeat:
                    red_reasons.append(f"REPEAT VISIT #{visit_count}")

            threat = self._classify(
                is_known, is_loit, is_masked, is_staring,
                conf, red_reasons, just_walking
            )

            if BLUR_NON_THREATS and threat.requires_blur:
                annotated = self._sdp_blur(annotated, x1,y1,x2,y2)

            # Build label — show reason for flagging
            reasons_display = []
            if is_masked:   reasons_display.append("MASKED")
            if is_staring:  reasons_display.append("STARING")
            if is_loit:     reasons_display.append(f"LOITER {int(loiter)}s")
            if red_reasons: reasons_display.extend(red_reasons)

            self._draw(annotated, x1,y1,x2,y2,
                       threat, tid, conf,
                       face_name, is_known, is_running,
                       speed, reasons_display)

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

    # ── Classification ────────────────────────────────────
    @staticmethod
    def _classify(is_known, is_loit, is_masked, is_staring,
                  conf, red_reasons, just_walking) -> ThreatLevel:
        # Weapons always RED regardless of who it is
        if any("WEAPON" in r for r in red_reasons):
            return ThreatLevel.RED

        # Known person — safe regardless of behaviour
        if is_known:
            return ThreatLevel.LOW

        # Person just walking through — no alert
        if just_walking and not is_masked and not is_staring:
            return ThreatLevel.NONE

        # Red escalations
        if red_reasons:
            return ThreatLevel.RED

        # Masked face or staring — HIGH
        if is_masked or is_staring:
            return ThreatLevel.HIGH

        # Loitering unknown
        if is_loit:
            return ThreatLevel.HIGH

        # Unrecognised face, present but not yet suspicious
        if conf > 0.5:
            return ThreatLevel.YELLOW

        return ThreatLevel.LOW

    # ── Drawing ───────────────────────────────────────────
    @staticmethod
    def _draw(frame, x1,y1,x2,y2, threat, tid, conf,
              face_name, is_known, is_running, speed, reasons):
        if threat == ThreatLevel.NONE:
            # Just walking through — minimal marker, no scary box
            cv2.rectangle(frame, (x1,y1),(x2,y2),(80,80,80),1)
            cv2.putText(frame, "PASSING", (x1,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)
            return

        color = threat.color_bgr
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,
                      3 if threat==ThreatLevel.RED else 2)

        name_lbl = face_name if is_known else "TRESPASSER"
        top_lbl  = f"{name_lbl}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(top_lbl,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
        cv2.putText(frame,top_lbl,(x1+3,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1)

        if reasons:
            cv2.putText(frame, " | ".join(reasons), (x1,y2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # ── Pose helpers ──────────────────────────────────────
    @staticmethod
    def _extract_poses(results, sx=1.0, sy=1.0):
        out = []
        for r in results:
            if r.keypoints is None: continue
            for kp in r.keypoints.xy:
                pts = kp.cpu().numpy()
                if not len(pts): continue
                sc = pts.copy()
                sc[:,0]/=sx; sc[:,1]/=sy
                out.append({"cx":float(np.mean(sc[:,0])),
                            "cy":float(np.mean(sc[:,1])),"pts":sc})
        return out

    @staticmethod
    def _nearest(cx, cy, poses):
        if not poses: return None
        best, bd = None, float("inf")
        for p in poses:
            d=((p["cx"]-cx)**2+(p["cy"]-cy)**2)**0.5
            if d<bd: bd,best=d,p
        return best if bd<POSE_R else None

    @staticmethod
    def _face_visible(pose) -> bool:
        pts = pose["pts"]
        return sum(1 for i in range(5)
                   if i<len(pts) and (pts[i][0]>0 or pts[i][1]>0)) >= 3

    @staticmethod
    def _sdp_blur(frame, x1,y1,x2,y2):
        fy2=min(y1+(y2-y1)//3,frame.shape[0])
        x1c,x2c=max(x1,0),min(x2,frame.shape[1])
        if fy2<=y1 or x2c<=x1c: return frame
        k=GAUSSIAN_BLUR_STRENGTH|1
        frame[y1:fy2,x1c:x2c]=cv2.GaussianBlur(
            frame[y1:fy2,x1c:x2c],(k,k),0)
        cv2.putText(frame,"SDP",(x1+4,y1+16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,212,255),1)
        return frame

    def _prune(self):
        now=time.time()
        if now-self._last_cleanup<CLEANUP_S: return
        stale=[k for k,h in self._hist.items() if now-h["last"]>CLEANUP_S]
        for k in stale:
            del self._hist[k]
            self._face_cache.pop(k,None)
            self._seen_tracks.discard(k)
        self._last_cleanup=now
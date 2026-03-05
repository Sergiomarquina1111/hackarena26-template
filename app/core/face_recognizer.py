"""
app/core/face_recognizer.py
Loads known faces from KNOWN_FACES_DIR on startup.
Call recognize(frame, bbox) per detection to get name or UNKNOWN.
CPU-optimised: uses small face model, skips frames via caller.
"""
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import face_recognition
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False
    logger.warning("face_recognition not installed — face ID disabled")


class FaceRecognizer:
    def __init__(self, known_faces_dir: str) -> None:
        self._encodings = []   # list of np arrays
        self._names:    list = []
        self._enabled = FR_AVAILABLE
        if not self._enabled:
            return
        self._load(known_faces_dir)

    # ── Public ────────────────────────────────────────────
    def recognize(self, frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[str, bool]:
        """
        Returns (name, is_known).
        name is person's name if known, else "TRESPASSER".
        Crops bbox region, runs recognition on it only (fast).
        """
        if not self._enabled or not self._encodings:
            return "TRESPASSER", False

        x1, y1, x2, y2 = bbox
        # face_recognition uses RGB
        rgb = frame[max(0,y1):y2, max(0,x1):x2, ::-1]
        if rgb.size == 0:
            return "TRESPASSER", False

        # Resize crop to speed up encoding on CPU
        h, w = rgb.shape[:2]
        scale = min(1.0, 320 / max(h, w, 1))
        if scale < 1.0:
            rgb = _resize(rgb, scale)

        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            return "TRESPASSER", False

        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            return "TRESPASSER", False

        matches = face_recognition.compare_faces(self._encodings, encs[0], tolerance=0.5)
        dists   = face_recognition.face_distance(self._encodings, encs[0])

        if True in matches:
            best = int(np.argmin(dists))
            return self._names[best], True

        return "TRESPASSER", False

    def add_face(self, image_path: str, name: str) -> bool:
        """Dynamically add a face (for dashboard upload feature)."""
        if not self._enabled:
            return False
        try:
            img  = face_recognition.load_image_file(image_path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                logger.warning("No face found in %s", image_path)
                return False
            self._encodings.append(encs[0])
            self._names.append(name)
            logger.info("Added face: %s", name)
            return True
        except Exception as e:
            logger.error("add_face failed: %s", e)
            return False

    @property
    def known_names(self) -> list:
        return list(self._names)

    # ── Private ───────────────────────────────────────────
    def _load(self, directory: str) -> None:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Created known_faces dir: %s", directory)
            return

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        loaded = 0
        for fp in path.iterdir():
            if fp.suffix.lower() not in exts:
                continue
            try:
                img  = face_recognition.load_image_file(str(fp))
                encs = face_recognition.face_encodings(img)
                if not encs:
                    logger.warning("No face found in %s — skipping", fp.name)
                    continue
                name = fp.stem.replace("_", " ").replace("-", " ").title()
                self._encodings.append(encs[0])
                self._names.append(name)
                loaded += 1
            except Exception as e:
                logger.error("Failed to load %s: %s", fp.name, e)

        logger.info("Face DB: %d known person(s) loaded from %s", loaded, directory)


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))))

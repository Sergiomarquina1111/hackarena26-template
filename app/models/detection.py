from dataclasses import dataclass, field
from app.models.threat import ThreatLevel

@dataclass
class Detection:
    track_id:       int
    threat:         ThreatLevel
    loitering:      bool
    masked:         bool
    loiter_seconds: float
    confidence:     float
    bbox:           tuple           # (x1, y1, x2, y2)

    # Face recognition
    face_name:      str   = "UNKNOWN"
    is_known:       bool  = False

    # Motion / behaviour
    is_running:     bool  = False
    visit_count:    int   = 1

    # Red alert sub-reasons
    red_reasons:    list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id":       self.track_id,
            "threat":         self.threat.value,
            "loitering":      self.loitering,
            "masked":         self.masked,
            "loiter_seconds": round(self.loiter_seconds, 1),
            "confidence":     round(self.confidence, 2),
            "face_name":      self.face_name,
            "is_known":       self.is_known,
            "is_running":     self.is_running,
            "visit_count":    self.visit_count,
            "red_reasons":    self.red_reasons,
        }

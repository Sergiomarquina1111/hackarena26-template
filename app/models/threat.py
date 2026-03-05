from enum import Enum

class ThreatLevel(str, Enum):
    RED    = "RED"       # Extreme — gang, night, repeat, running
    HIGH   = "HIGH"      # Masked or loitering unknown
    YELLOW = "YELLOW"    # Unrecognised face / trespasser
    LOW    = "LOW"       # Known person
    NONE   = "NONE"

    @property
    def color_bgr(self) -> tuple:
        return {
            ThreatLevel.RED:    (0, 0, 220),
            ThreatLevel.HIGH:   (0, 0, 255),
            ThreatLevel.YELLOW: (0, 200, 255),
            ThreatLevel.LOW:    (0, 220, 0),
            ThreatLevel.NONE:   (180, 180, 180),
        }[self]

    @property
    def requires_alert(self) -> bool:
        return self in (ThreatLevel.RED, ThreatLevel.HIGH, ThreatLevel.YELLOW)

    @property
    def requires_blur(self) -> bool:
        return self in (ThreatLevel.LOW, ThreatLevel.NONE)

    @property
    def clip_seconds(self) -> int:
        return {
            ThreatLevel.RED:    15,
            ThreatLevel.HIGH:   10,
            ThreatLevel.YELLOW: 10,
            ThreatLevel.LOW:    5,
            ThreatLevel.NONE:   5,
        }[self]

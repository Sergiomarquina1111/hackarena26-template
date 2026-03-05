from abc import ABC, abstractmethod

class AbstractNotifier(ABC):
    """
    Abstract base for all notification channels.
    Implement in app/infrastructure/notifiers/.
    """
    @abstractmethod
    def send_alert(self, threat_level: str, timestamp: str, clip_path: str,
                   masked: bool = False, loitering: bool = False, loiter_secs: int = 0) -> None: ...

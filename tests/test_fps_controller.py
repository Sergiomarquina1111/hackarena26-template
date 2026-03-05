"""Unit tests for adaptive FPS controller."""
class TestFPSController:
    def test_default_monitor_mode(self):
        from app.core.fps_controller import FPSController
        c = FPSController()
        assert c.mode == "MONITOR" and c.current_fps == 5

    def test_switches_to_evidence_on_high(self):
        from app.core.fps_controller import FPSController
        from app.models.threat import ThreatLevel
        c = FPSController()
        c.update([{"threat": ThreatLevel.HIGH}])
        assert c.mode == "EVIDENCE" and c.current_fps == 20

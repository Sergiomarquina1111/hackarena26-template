"""Unit tests for threat classification logic."""
import pytest

class TestThreatClassification:
    def test_masked_face_triggers_high(self):
        from app.core.analyzer import ThreatAnalyzer
        assert ThreatAnalyzer._classify(False, True,  0.5).value == "HIGH"

    def test_loitering_triggers_high(self):
        from app.core.analyzer import ThreatAnalyzer
        assert ThreatAnalyzer._classify(True,  False, 0.5).value == "HIGH"

    def test_high_confidence_is_medium(self):
        from app.core.analyzer import ThreatAnalyzer
        assert ThreatAnalyzer._classify(False, False, 0.8).value == "MEDIUM"

    def test_low_confidence_is_low(self):
        from app.core.analyzer import ThreatAnalyzer
        assert ThreatAnalyzer._classify(False, False, 0.5).value == "LOW"

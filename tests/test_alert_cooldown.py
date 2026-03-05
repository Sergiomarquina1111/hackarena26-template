"""Unit tests for alert cooldown / Intelligence Buffer."""
class TestAlertCooldown:
    def _make_mgr(self):
        from app.core.alert_manager import AlertManager
        from app.services.telegram_notifier import TelegramNotifier
        return AlertManager(TelegramNotifier())

    def test_first_alert_fires(self):
        mgr = self._make_mgr()
        assert mgr.trigger_alert("HIGH", [], masked=False, loitering=False, loiter_secs=0) is True

    def test_duplicate_suppressed(self):
        mgr = self._make_mgr()
        mgr.trigger_alert("HIGH", [], masked=False, loitering=False, loiter_secs=0)
        assert mgr.trigger_alert("HIGH", [], masked=False, loitering=False, loiter_secs=0) is False

    def test_cooldown_countdown_decreases(self):
        import time
        mgr = self._make_mgr()
        mgr.trigger_alert("HIGH", [], masked=False, loitering=False, loiter_secs=0)
        r1 = mgr.cooldown_remaining("HIGH")
        time.sleep(1)
        r2 = mgr.cooldown_remaining("HIGH")
        assert r2 < r1

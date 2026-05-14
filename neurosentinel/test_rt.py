"""
Tests for neurosentinel/rt

Run with:  pytest tests/rt/ -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_eeg(n_channels: int, n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples)).astype(np.float64)


def inject_line_noise(
    eeg: np.ndarray,
    sfreq: float,
    freq: float = 60.0,
    amplitude: float = 5.0,
) -> np.ndarray:
    t = np.arange(eeg.shape[1]) / sfreq
    sinusoid = amplitude * np.sin(2 * np.pi * freq * t)
    noisy = eeg.copy()
    noisy += sinusoid[np.newaxis, :]
    return noisy


# ---------------------------------------------------------------------------
# buffers
# ---------------------------------------------------------------------------

class TestRollingEEGBuffer:
    def test_append_and_get(self):
        from neurosentinel.rt.buffers import RollingEEGBuffer

        buf = RollingEEGBuffer(n_channels=4, capacity=100)
        chunk = make_eeg(4, 60)
        buf.append(chunk)
        assert buf.n_samples_available == 60

        window = buf.get_latest(60)
        np.testing.assert_array_equal(window, chunk)

    def test_not_enough_data_returns_none(self):
        from neurosentinel.rt.buffers import RollingEEGBuffer

        buf = RollingEEGBuffer(n_channels=4, capacity=100)
        buf.append(make_eeg(4, 30))
        assert buf.get_latest(50) is None

    def test_ring_wraps_correctly(self):
        from neurosentinel.rt.buffers import RollingEEGBuffer

        buf = RollingEEGBuffer(n_channels=2, capacity=10)
        # Fill past capacity in two chunks
        chunk1 = make_eeg(2, 7, seed=1)
        chunk2 = make_eeg(2, 7, seed=2)
        buf.append(chunk1)
        buf.append(chunk2)

        # Buffer should hold last 10 samples (3 from chunk1, 7 from chunk2)
        window = buf.get_latest(10)
        assert window.shape == (2, 10)
        np.testing.assert_array_equal(window[:, 3:], chunk2)

    def test_wrong_channel_count_raises(self):
        from neurosentinel.rt.buffers import RollingEEGBuffer

        buf = RollingEEGBuffer(n_channels=4, capacity=100)
        with pytest.raises(ValueError, match="4 channels"):
            buf.append(make_eeg(3, 10))


class TestImpedanceStore:
    def test_update_and_retrieve(self):
        from neurosentinel.rt.buffers import ImpedanceStore

        store = ImpedanceStore(n_channels=8, history_length=3)
        vals = np.ones(8) * 25.0
        store.update(vals, timestamp=1.0)

        assert store.has_data
        np.testing.assert_array_equal(store.latest, vals)

    def test_history_bounded(self):
        from neurosentinel.rt.buffers import ImpedanceStore

        store = ImpedanceStore(n_channels=4, history_length=2)
        for i in range(5):
            store.update(np.ones(4) * i, timestamp=float(i))
        assert len(store.history()) == 2


# ---------------------------------------------------------------------------
# ewma
# ---------------------------------------------------------------------------

class TestOnlineEWMA:
    def test_cold_start_seeds_from_first_value(self):
        from neurosentinel.rt.ewma import OnlineEWMA

        ewma = OnlineEWMA(beta=0.9)
        val = ewma.update(10.0)
        assert val == pytest.approx(10.0)

    def test_smoothing(self):
        from neurosentinel.rt.ewma import OnlineEWMA

        ewma = OnlineEWMA(beta=0.9, initial_value=0.0)
        # After many updates with value=10, EWMA should converge toward 10
        for _ in range(200):
            ewma.update(10.0)
        assert ewma.current == pytest.approx(10.0, abs=0.01)

    def test_reset(self):
        from neurosentinel.rt.ewma import OnlineEWMA

        ewma = OnlineEWMA(beta=0.9)
        ewma.update(5.0)
        ewma.reset()
        assert not ewma.is_initialized
        assert ewma.n_updates == 0

    def test_invalid_beta_raises(self):
        from neurosentinel.rt.ewma import OnlineEWMA

        with pytest.raises(ValueError):
            OnlineEWMA(beta=1.5)


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

class TestThresholdRule:
    def test_enters_bad_after_persistence(self):
        from neurosentinel.rt.rules import ThresholdRule

        rule = ThresholdRule(
            enter_threshold=3.0,
            exit_threshold=2.0,
            min_steps_to_enter=3,
            min_steps_to_clear=2,
            enter_message="Bad!",
        )
        from neurosentinel.rt.schemas import SignalState

        for i in range(2):
            result = rule.update(4.0, timestamp=float(i))
            assert result.state == SignalState.GOOD

        result = rule.update(4.0, timestamp=2.0)
        assert result.state == SignalState.BAD
        assert result.transitioned is True
        assert result.message == "Bad!"

    def test_clears_bad_after_persistence(self):
        from neurosentinel.rt.rules import ThresholdRule
        from neurosentinel.rt.schemas import SignalState

        rule = ThresholdRule(
            enter_threshold=3.0,
            exit_threshold=2.0,
            min_steps_to_enter=1,
            min_steps_to_clear=2,
        )
        rule.update(4.0, 0.0)  # → BAD

        rule.update(1.0, 1.0)  # good, but not enough
        assert rule.current_state == SignalState.BAD

        rule.update(1.0, 2.0)  # second good step → clears
        assert rule.current_state == SignalState.GOOD

    def test_hysteresis_prevents_chattering(self):
        """Value oscillating between 2.5 and 3.5 should not cause repeated transitions."""
        from neurosentinel.rt.rules import ThresholdRule
        from neurosentinel.rt.schemas import SignalState

        rule = ThresholdRule(
            enter_threshold=3.0,
            exit_threshold=2.0,
            min_steps_to_enter=1,
            min_steps_to_clear=1,
        )
        transitions = []
        for v in [3.5, 2.5, 3.5, 2.5, 3.5]:
            r = rule.update(v, 0.0)
            if r.transitioned:
                transitions.append(r.state)

        # Value never drops below exit_threshold (2.0), so it should not clear
        assert SignalState.GOOD not in transitions

    def test_exit_above_enter_raises(self):
        from neurosentinel.rt.rules import ThresholdRule

        with pytest.raises(ValueError, match="hysteresis"):
            ThresholdRule(enter_threshold=2.0, exit_threshold=3.0)


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------

class TestAlertManager:
    def test_emits_on_transition(self):
        from neurosentinel.rt.alerts import AlertManager
        from neurosentinel.rt.schemas import AlertSeverity, RuleResult, SignalState

        mgr = AlertManager(cooldowns={"ln": 0.0})
        result = RuleResult(
            state=SignalState.BAD,
            transitioned=True,
            severity=AlertSeverity.WARNING,
            message="Line noise!",
        )
        alerts = mgr.process("ln", result, timestamp=1.0)
        assert len(alerts) == 1
        assert alerts[0].message == "Line noise!"

    def test_cooldown_suppresses_repeat(self):
        from neurosentinel.rt.alerts import AlertManager
        from neurosentinel.rt.schemas import AlertSeverity, RuleResult, SignalState

        mgr = AlertManager(cooldowns={"ln": 30.0})
        result = RuleResult(
            state=SignalState.BAD,
            transitioned=True,
            severity=AlertSeverity.WARNING,
            message="Noise!",
        )
        mgr.process("ln", result, timestamp=0.0)
        alerts = mgr.process("ln", result, timestamp=10.0)  # within cooldown
        assert len(alerts) == 0

    def test_flush_clears_pending(self):
        from neurosentinel.rt.alerts import AlertManager
        from neurosentinel.rt.schemas import AlertSeverity, RuleResult, SignalState

        mgr = AlertManager()
        result = RuleResult(
            state=SignalState.BAD,
            transitioned=True,
            severity=AlertSeverity.WARNING,
            message="X",
        )
        mgr.process("m", result, timestamp=1.0)
        assert len(mgr.flush()) == 1
        assert len(mgr.flush()) == 0  # cleared


# ---------------------------------------------------------------------------
# engine (integration)
# ---------------------------------------------------------------------------

class TestRealtimeMonitor:
    def _make_monitor(self, sfreq: float = 256.0, n_channels: int = 8):
        from neurosentinel.rt import RTConfig, RealtimeMonitor
        from neurosentinel.rt.config import EWMAConfig, ImpedanceConfig, LineNoiseConfig

        cfg = RTConfig(
            sfreq=sfreq,
            n_channels=n_channels,
            line_noise=LineNoiseConfig(
                window_sec=2.0,
                step_sec=0.5,
                enter_threshold_db=3.0,
                exit_threshold_db=2.0,
                min_steps_to_enter=2,
                min_steps_to_clear=2,
                alert_cooldown_sec=0.0,
                ewma=EWMAConfig(beta=0.8),
            ),
            impedance=ImpedanceConfig(
                bad_threshold_kohm=30.0,
                min_steps_to_enter=1,
                min_steps_to_clear=1,
                alert_cooldown_sec=0.0,
            ),
        )
        return RealtimeMonitor(cfg)

    def test_no_alert_on_clean_signal(self):
        monitor = self._make_monitor(sfreq=256, n_channels=8)
        rng = np.random.default_rng(42)
        t = 0.0
        for _ in range(20):
            chunk = rng.standard_normal((8, 64)).astype(np.float64) * 0.1
            result = monitor.update_eeg(chunk, timestamp=t)
            t += 64 / 256
        # Clean signal should not trigger line noise alert
        assert all(
            not a.get("metric") == "line_noise"
            for a in result.alerts
        )

    def test_line_noise_alert_on_contaminated_signal(self):
        monitor = self._make_monitor(sfreq=256, n_channels=8)
        sfreq = 256.0
        alerts_seen = []
        t = 0.0
        for step in range(30):
            chunk = make_eeg(8, 128, seed=step)
            chunk = inject_line_noise(chunk, sfreq, amplitude=20.0)
            result = monitor.update_eeg(chunk, timestamp=t)
            alerts_seen.extend(result.alerts)
            t += 128 / sfreq

        line_noise_alerts = [a for a in alerts_seen if a["metric"] == "line_noise"]
        assert len(line_noise_alerts) >= 1, "Expected at least one line noise alert"

    def test_impedance_alert_above_threshold(self):
        monitor = self._make_monitor()
        vals = np.ones(8) * 50.0  # 50 kΩ — above 30 kΩ threshold
        result = monitor.update_impedance(vals, timestamp=5.0)
        # First call: enters bad after min_steps_to_enter=1
        assert result.metrics.get("impedance") is not None

        result2 = monitor.update_impedance(vals, timestamp=6.0)
        imp_alerts = [a for a in result2.alerts if a["metric"] == "impedance"]
        # May alert on first or second step depending on persistence config
        all_alerts = [a for a in (result.alerts + result2.alerts) if a["metric"] == "impedance"]
        assert len(all_alerts) >= 1

    def test_impedance_good_below_threshold(self):
        monitor = self._make_monitor()
        vals = np.ones(8) * 10.0  # 10 kΩ — well below threshold
        result = monitor.update_impedance(vals, timestamp=1.0)
        imp_alerts = [a for a in result.alerts if a["metric"] == "impedance"]
        assert len(imp_alerts) == 0

    def test_get_current_state_returns_dict(self):
        monitor = self._make_monitor()
        monitor.update_impedance(np.ones(8) * 10.0, timestamp=1.0)
        state = monitor.get_current_state()
        assert "impedance" in state
        assert state["impedance"] in ("good", "bad", "warning", "unknown")

    def test_reset_clears_state(self):
        monitor = self._make_monitor()
        monitor.update_impedance(np.ones(8) * 50.0, timestamp=1.0)
        monitor.reset()
        assert monitor.get_current_state() == {}
        assert monitor.get_new_alerts() == []


# ---------------------------------------------------------------------------
# metrics (unit)
# ---------------------------------------------------------------------------

class TestLineNoiseMetric:
    def test_returns_float(self):
        from neurosentinel.rt.metrics import line_noise_db

        eeg = make_eeg(8, 512)
        val, meta = line_noise_db(eeg, sfreq=256.0)
        assert isinstance(val, float)
        assert "db_per_channel" in meta

    def test_noisy_signal_higher_than_clean(self):
        from neurosentinel.rt.metrics import line_noise_db

        clean = make_eeg(8, 1024, seed=0) * 0.1
        noisy = inject_line_noise(clean, sfreq=256.0, amplitude=10.0)

        db_clean, _ = line_noise_db(clean, sfreq=256.0)
        db_noisy, _ = line_noise_db(noisy, sfreq=256.0)
        assert db_noisy > db_clean


class TestImpedanceSummary:
    def test_median_and_proportion(self):
        from neurosentinel.rt.metrics import impedance_summary

        vals = np.array([10.0, 20.0, 40.0, 50.0])
        median, meta = impedance_summary(vals, bad_threshold_kohm=30.0)

        assert median == pytest.approx(30.0)  # median of [10,20,40,50]
        assert meta["prop_below_threshold"] == pytest.approx(0.5)

    def test_worst_channels_sorted(self):
        from neurosentinel.rt.metrics import impedance_summary

        vals = np.array([5.0, 100.0, 50.0, 20.0])
        _, meta = impedance_summary(vals, bad_threshold_kohm=30.0)
        worst = meta["worst_channels"]
        assert worst[0]["channel"] == 1  # 100 kΩ is worst
        assert worst[0]["impedance_kohm"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# bad_channel_features
# ---------------------------------------------------------------------------

class TestBadChannelFeatures:
    def test_returns_proportion_and_metadata(self):
        from neurosentinel.rt.metrics import bad_channel_features

        eeg = make_eeg(8, 512)
        prop, meta = bad_channel_features(eeg, n_neighbours=2)
        assert 0.0 <= prop <= 1.0
        assert "local_corr" in meta
        assert "excess_kurt" in meta
        assert "max_amp_uv" in meta
        assert len(meta["bad_mask"]) == 8

    def test_flat_channel_flagged_by_correlation(self):
        """A zeroed-out channel should have near-zero correlation with neighbours."""
        from neurosentinel.rt.metrics import bad_channel_features

        eeg = make_eeg(8, 512)
        eeg[3, :] = 0.0   # flat channel
        _, meta = bad_channel_features(eeg, n_neighbours=2)
        # Channel 3 should be corr_bad
        assert meta["corr_bad"][3] is True

    def test_high_amplitude_channel_flagged(self):
        from neurosentinel.rt.metrics import bad_channel_features

        eeg = make_eeg(8, 512) * 0.1   # small amplitude baseline
        eeg[5, :] += 500.0             # one channel with huge offset + PtP
        eeg[5, 0]  = -500.0
        _, meta = bad_channel_features(eeg, n_neighbours=2)
        assert meta["amp_bad"][5] is True

    def test_high_kurtosis_channel_flagged(self):
        from neurosentinel.rt.metrics import bad_channel_features

        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((8, 1024)).astype(np.float64) * 0.5
        # Inject spike train on channel 2 → very high kurtosis
        eeg[2, ::20] += 200.0
        _, meta = bad_channel_features(eeg, n_neighbours=2)
        assert meta["kurt_bad"][2] is True


# ---------------------------------------------------------------------------
# anomaly_score
# ---------------------------------------------------------------------------

class TestAnomalyScore:
    def test_returns_float_and_metadata(self):
        from neurosentinel.rt.metrics import anomaly_score

        eeg = make_eeg(8, 512)
        val, meta = anomaly_score(eeg, reference_stats=None)
        assert isinstance(val, float)
        assert "raw_score" in meta
        assert meta["normalised_score"] == val   # same when no ref

    def test_normalised_higher_for_artefact(self):
        """After warm-up, a highly anomalous window should score above a clean one."""
        from neurosentinel.rt.metrics import anomaly_score

        rng = np.random.default_rng(42)
        clean = rng.standard_normal((8, 512)).astype(np.float64)
        # Build reference from clean windows
        raw_scores = []
        for _ in range(20):
            w = rng.standard_normal((8, 512)).astype(np.float64)
            _, m = anomaly_score(w, reference_stats=None)
            raw_scores.append(m["raw_score"])
        hist = np.array(raw_scores)
        ref = {"median": float(np.median(hist)), "mad": float(np.median(np.abs(hist - np.median(hist))))}

        # Clean window normalised
        norm_clean, _ = anomaly_score(clean, reference_stats=ref)

        # Artefact window: massive broadband burst on two channels
        artefact = clean.copy()
        artefact[:2, :] *= 50.0
        norm_artefact, _ = anomaly_score(artefact, reference_stats=ref)

        assert norm_artefact > norm_clean


# ---------------------------------------------------------------------------
# Engine — bad_channel and anomaly (integration)
# ---------------------------------------------------------------------------

class TestRealtimeMonitorExtended:
    def _make_monitor(self, sfreq=256.0, n_ch=16):
        from neurosentinel.rt import RTConfig, RealtimeMonitor
        from neurosentinel.rt.config import (
            AnomalyConfig, BadChannelConfig, EWMAConfig,
            ImpedanceConfig, LineNoiseConfig,
        )
        cfg = RTConfig(
            sfreq=sfreq,
            n_channels=n_ch,
            line_noise=LineNoiseConfig(
                window_sec=2.0, step_sec=0.5,
                min_steps_to_enter=2, min_steps_to_clear=2,
                alert_cooldown_sec=0.0, ewma=EWMAConfig(beta=0.8),
            ),
            bad_channel=BadChannelConfig(
                window_sec=2.0, step_sec=0.5,
                min_steps_to_enter=2, min_steps_to_clear=2,
                alert_cooldown_sec=0.0,
                bad_channel_proportion_threshold=0.05,  # low: flag if >5% bad
                bad_channel_proportion_exit=0.03,
                ewma=EWMAConfig(beta=0.8),
            ),
            anomaly=AnomalyConfig(
                window_sec=2.0, step_sec=0.5,
                reference_steps=10,
                enter_threshold=3.0, exit_threshold=2.0,
                min_steps_to_enter=2, min_steps_to_clear=2,
                alert_cooldown_sec=0.0, ewma=EWMAConfig(beta=0.8),
            ),
            impedance=ImpedanceConfig(
                bad_threshold_kohm=30.0, min_steps_to_enter=1, min_steps_to_clear=1,
                alert_cooldown_sec=0.0,
            ),
        )
        return RealtimeMonitor(cfg)

    def test_all_four_metrics_appear_in_state(self):
        monitor = self._make_monitor()
        sfreq = 256.0
        t = 0.0
        # Feed enough data for all pipelines to fire at least once
        for step in range(40):
            chunk = np.random.randn(16, 128).astype(np.float64)
            monitor.update_eeg(chunk, timestamp=t)
            t += 128 / sfreq
        monitor.update_impedance(np.ones(16) * 10.0, timestamp=t)
        state = monitor.get_current_state()
        assert set(state.keys()) >= {"line_noise", "bad_channel", "anomaly", "impedance"}

    def test_bad_channel_alert_on_persistent_flat_channels(self):
        monitor = self._make_monitor(n_ch=16)
        sfreq = 256.0
        t = 0.0
        alerts_seen = []
        for step in range(40):
            rng = np.random.default_rng(step)
            chunk = rng.standard_normal((16, 128)).astype(np.float64)
            # Keep 3 of 16 channels flat (18.75% > 5% threshold)
            chunk[0, :] = 0.0
            chunk[1, :] = 0.0
            chunk[2, :] = 0.0
            result = monitor.update_eeg(chunk, timestamp=t)
            alerts_seen.extend(result.alerts)
            t += 128 / sfreq
        bc_alerts = [a for a in alerts_seen if a["metric"] == "bad_channel"]
        assert len(bc_alerts) >= 1, "Expected bad_channel alert with 3 flat channels"

    def test_anomaly_alert_on_broadband_burst(self):
        monitor = self._make_monitor(n_ch=16)
        sfreq = 256.0
        t = 0.0
        alerts_seen = []

        # Warm-up with clean signal to build anomaly reference
        for step in range(20):
            chunk = np.random.default_rng(step).standard_normal((16, 128)).astype(np.float64)
            monitor.update_eeg(chunk, timestamp=t)
            t += 128 / sfreq

        # Inject broadband artefact (channels scaled up 20x)
        for step in range(20):
            rng = np.random.default_rng(step + 100)
            chunk = rng.standard_normal((16, 128)).astype(np.float64) * 20.0
            result = monitor.update_eeg(chunk, timestamp=t)
            alerts_seen.extend(result.alerts)
            t += 128 / sfreq

        an_alerts = [a for a in alerts_seen if a["metric"] == "anomaly"]
        assert len(an_alerts) >= 1, "Expected anomaly alert during broadband burst"

    def test_monitor_reset_clears_all_metrics(self):
        monitor = self._make_monitor()
        t = 0.0
        for step in range(20):
            chunk = np.random.randn(16, 128).astype(np.float64)
            monitor.update_eeg(chunk, timestamp=t)
            t += 128 / 256.0
        monitor.update_impedance(np.ones(16) * 50.0, timestamp=t)
        monitor.reset()
        assert monitor.get_current_state() == {}
        assert monitor.get_new_alerts() == []

    def test_enriched_bad_channel_message_contains_count(self):
        """The alert message should contain the number of bad channels."""
        monitor = self._make_monitor(n_ch=16)
        sfreq = 256.0
        t = 0.0
        alerts_seen = []
        for step in range(40):
            rng = np.random.default_rng(step)
            chunk = rng.standard_normal((16, 128)).astype(np.float64)
            chunk[0, :] = chunk[1, :] = chunk[2, :] = 0.0  # 3 flat channels
            result = monitor.update_eeg(chunk, timestamp=t)
            alerts_seen.extend(result.alerts)
            t += 128 / sfreq
        bc_alerts = [a for a in alerts_seen if a["metric"] == "bad_channel"]
        if bc_alerts:
            msg = bc_alerts[0]["message"]
            # Message should mention the channel count or percentage
            assert any(c.isdigit() for c in msg), f"Expected digits in message: {msg!r}"

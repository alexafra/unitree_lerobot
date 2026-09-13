from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from unitree_lerobot.eval_robot import probe_dex3_ranges as probe


def test_parser_defaults_to_dex3_and_accepts_inspire_ftp() -> None:
    parser = probe.build_parser()
    default = parser.parse_args(["--network-interface", "eth0"])
    ftp = parser.parse_args(
        ["--network-interface", "eth0", "--end-effector", "inspire-ftp"]
    )

    assert default.end_effector == "dex3"
    assert ftp.end_effector == "inspire-ftp"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--network-interface", "eth0", "--end-effector", "inspire-dfx"]
        )


def test_dex3_decoder_preserves_seven_motor_radians() -> None:
    message = SimpleNamespace(
        motor_state=[SimpleNamespace(q=value) for value in np.linspace(-0.6, 0.6, 7)]
    )

    values = probe._decode_dex3_state(message)

    assert values is not None
    np.testing.assert_allclose(values, np.linspace(-0.6, 0.6, 7))


def test_inspire_ftp_decoder_normalizes_six_angle_codes() -> None:
    message = SimpleNamespace(angle_act=[0, 200, 400, 600, 800, 1000])

    values = probe._decode_inspire_ftp_state(message)

    assert values is not None
    np.testing.assert_allclose(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(),
        SimpleNamespace(angle_act=[0, 1, 2, 3, 4]),
        SimpleNamespace(angle_act=[0, 1, 2, 3, 4, 5, 6]),
        SimpleNamespace(angle_act=[0, 1, 2, 3, 4, np.nan]),
        SimpleNamespace(angle_act=[0, 1, 2, 3, 4, -1]),
        SimpleNamespace(angle_act=[0, 1, 2, 3, 4, 1001]),
        SimpleNamespace(angle_act=["0", "1", "2", "3", "4", "5"]),
    ],
)
def test_inspire_ftp_decoder_rejects_malformed_values(message: object) -> None:
    assert probe._decode_inspire_ftp_state(message) is None


def test_dynamic_hand_range_reports_rate_gaps_and_rejections() -> None:
    hand = probe.HandRange("left", 6)
    first = np.linspace(0.0, 0.5, 6)
    second = np.linspace(0.1, 0.6, 6)

    hand.update(first, 10.0)
    hand.update(None, 10.01)
    hand.update(second, 10.1)

    assert hand.callback_count == 3
    assert hand.count == 2
    assert hand.rejected_count == 1
    assert hand.mean_rate_hz == pytest.approx(10.0)
    assert hand.latest_rate_hz == pytest.approx(10.0)
    assert hand.maximum_gap_s == pytest.approx(0.1)
    assert hand.gaps_over_freshness == pytest.approx([0.1])


def test_snapshot_copy_keeps_six_dof_state_independent() -> None:
    hand = probe.HandRange("left", 6)
    hand.update(np.arange(6, dtype=np.float64), 1.0)

    copied = hand.copy()
    copied.current[0] = 99.0

    assert hand.current.shape == (6,)
    assert hand.current[0] == 0.0


def test_inspire_ftp_probe_constructs_only_expected_state_subscribers(monkeypatch) -> None:
    subscribers = []

    class FakeState:
        pass

    class FakeSubscriber:
        def __init__(self, topic, message_type):
            self.topic = topic
            self.message_type = message_type
            self.handler = None
            self.closed = False
            subscribers.append(self)

        def Init(self, handler):
            self.handler = handler

        def Close(self):
            self.closed = True

    channel_module = ModuleType("unitree_sdk2py.core.channel")
    channel_module.ChannelSubscriber = FakeSubscriber
    inspire_package = ModuleType("inspire_sdkpy")
    inspire_package.inspire_dds = SimpleNamespace(inspire_hand_state=FakeState)
    initialized = []
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel_module)
    monkeypatch.setitem(sys.modules, "inspire_sdkpy", inspire_package)
    monkeypatch.setattr(
        probe,
        "initialize_dds",
        lambda simulation, interface: initialized.append((simulation, interface)),
    )
    monkeypatch.setattr(probe.time, "monotonic", lambda: 10.0)

    hand_probe = probe.Dex3RangeProbe("eth-test", "inspire-ftp")

    assert initialized == [(False, "eth-test")]
    assert [(item.topic, item.message_type) for item in subscribers] == [
        ("rt/inspire_hand/state/l", FakeState),
        ("rt/inspire_hand/state/r", FakeState),
    ]
    assert all(item.handler is not None for item in subscribers)
    subscribers[0].handler(SimpleNamespace(angle_act=[0, 200, 400, 600, 800, 1000]))
    np.testing.assert_allclose(
        hand_probe.snapshot()["left"].current,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    hand_probe.close()
    assert all(item.closed for item in subscribers)


def test_probe_source_never_constructs_a_command_publisher() -> None:
    source = probe.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "ChannelPublisher" not in text

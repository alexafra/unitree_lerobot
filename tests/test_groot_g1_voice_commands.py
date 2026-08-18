from types import SimpleNamespace
import unittest
from unittest import mock
from uuid import uuid4

from unitree_lerobot.eval_robot.eval_groot_g1 import (
    RETURN_TO_START,
    VOICE_STAY_HOLDING,
    _select_next_goal_while_holding,
    _take_held_voice_command,
    _take_initial_voice_command,
    build_parser,
    validate_args,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.voice_command_server import VoiceCommand


class FakeVoiceServer:
    def __init__(self, goal: str):
        self.command = VoiceCommand(str(uuid4()), "confirmation", goal, "phone")
        self.accepted = []
        self.rejected = []

    @property
    def command_pending(self):
        return self.command is not None

    def poll_command(self):
        command, self.command = self.command, None
        return command

    def accept_command(self, command, **_kwargs):
        self.accepted.append(command.goal)
        return True

    def reject_command(self, command, code, message):
        self.rejected.append((command.goal, code, message))
        return True


def held_actuator():
    return SimpleNamespace(
        heartbeat=mock.Mock(),
        assert_healthy=mock.Mock(),
        immediate_control_requested=lambda: None,
        hold=mock.Mock(),
    )


class GrootVoiceCommandIntegrationTest(unittest.TestCase):
    def test_voice_flags_parse_and_confirm_text_requires_voice(self):
        parser = build_parser()
        args = parser.parse_args(["--voice", "--confirm-text"])
        validate_args(args)
        self.assertTrue(args.voice)
        self.assertTrue(args.confirm_text)
        with self.assertRaisesRegex(DeploymentError, "requires --voice"):
            validate_args(parser.parse_args(["--confirm-text"]))

    def test_confirmed_voice_stop_or_pause_stays_in_powered_hold(self):
        for goal in ("stop", "STOP", "pause"):
            with self.subTest(goal=goal):
                server = FakeVoiceServer(goal)
                actuator = held_actuator()
                result = _take_held_voice_command(actuator, server, confirm_voice_text=False)
                self.assertEqual(result, VOICE_STAY_HOLDING)
                actuator.hold.assert_called_once_with()
                self.assertEqual(server.accepted, [goal])

    def test_confirmed_voice_quit_releases_and_custom_text_becomes_goal(self):
        actuator = held_actuator()
        quit_server = FakeVoiceServer("quit")
        self.assertEqual(
            _take_held_voice_command(actuator, quit_server, confirm_voice_text=False),
            "release",
        )
        self.assertEqual(quit_server.accepted, ["quit"])

        goal_server = FakeVoiceServer("stack six cups away from me")
        self.assertEqual(
            _take_held_voice_command(actuator, goal_server, confirm_voice_text=False),
            ("custom-goal", "stack six cups away from me"),
        )
        self.assertEqual(goal_server.accepted, ["stack six cups away from me"])

    def test_return_to_start_keyword_requires_the_existing_run_option(self):
        actuator = held_actuator()
        disabled = FakeVoiceServer("return to start")
        self.assertEqual(
            _take_held_voice_command(
                actuator,
                disabled,
                confirm_voice_text=False,
                return_to_start=False,
            ),
            VOICE_STAY_HOLDING,
        )
        self.assertEqual(disabled.rejected[0][1], "return_to_start_disabled")

        enabled = FakeVoiceServer("return to start")
        self.assertEqual(
            _take_held_voice_command(
                actuator,
                enabled,
                confirm_voice_text=False,
                return_to_start=True,
            ),
            RETURN_TO_START,
        )
        self.assertEqual(enabled.accepted, ["return to start"])

    def test_optional_local_text_confirmation_can_reject_phone_command(self):
        server = FakeVoiceServer("pick up the red cup")
        actuator = held_actuator()
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with mock.patch(f"{module}._readline_while_armed", return_value="NO"):
            result = _take_held_voice_command(actuator, server, confirm_voice_text=True)
        self.assertEqual(result, VOICE_STAY_HOLDING)
        self.assertFalse(server.accepted)
        self.assertEqual(server.rejected[0][1], "local_confirmation_rejected")

    def test_voice_only_initial_selection_and_terminal_selector_remain_supported(self):
        initial_server = FakeVoiceServer("pick up the wooden block")
        self.assertEqual(
            _take_initial_voice_command(initial_server, confirm_voice_text=False),
            ("custom-goal", "pick up the wooden block"),
        )
        self.assertEqual(initial_server.accepted, ["pick up the wooden block"])

        held_server = FakeVoiceServer("place the cup on the table")
        actuator = held_actuator()
        selected = _select_next_goal_while_holding(
            actuator,
            voice_server=held_server,
            confirm_voice_text=False,
        )
        self.assertEqual(selected, ("custom-goal", "place the cup on the table"))
        self.assertEqual(held_server.accepted, ["place the cup on the table"])


if __name__ == "__main__":
    unittest.main()

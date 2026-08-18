import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
from uuid import uuid4

from unitree_lerobot.eval_robot.voice_command_server import (
    JsonLineDecoder,
    VoiceCommandServer,
    VoiceProtocolError,
    load_voice_session_token,
    normalize_voice_goal,
)


TOKEN = "local-development-token-123"


def read_json_line(connection: socket.socket) -> dict[str, object]:
    payload = bytearray()
    while b"\n" not in payload:
        data = connection.recv(4096)
        if not data:
            raise AssertionError("connection closed before a JSON line arrived")
        payload.extend(data)
    return json.loads(bytes(payload.split(b"\n", 1)[0]).decode("utf-8"))


def wait_for_command(server: VoiceCommandServer):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        command = server.poll_command()
        if command is not None:
            return command
        time.sleep(0.01)
    raise AssertionError("voice command did not reach the controller")


class VoiceCommandServerTest(unittest.TestCase):
    def test_json_line_decoder_handles_partial_and_combined_reads(self):
        decoder = JsonLineDecoder()
        self.assertEqual(decoder.feed(b'{"a":'), [])
        self.assertEqual(
            decoder.feed(b'1}\n{"b":2}\n{"c"'),
            [{"a": 1}, {"b": 2}],
        )
        self.assertEqual(decoder.feed(b':3}\n'), [{"c": 3}])

    def test_goal_normalization_is_trimmed_printable_and_bounded(self):
        self.assertEqual(normalize_voice_goal("  pick   up the cup  "), "pick up the cup")
        with self.assertRaises(VoiceProtocolError):
            normalize_voice_goal("bad\ncommand")
        with self.assertRaises(VoiceProtocolError):
            normalize_voice_goal("x" * 257)

    def test_token_loads_from_file_without_logging_or_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text(f"{TOKEN}\n", encoding="utf-8")
            self.assertEqual(load_voice_session_token(path), TOKEN)

    def test_two_phase_round_trip_requires_controller_acceptance(self):
        holds = []
        server = VoiceCommandServer(
            "127.0.0.1",
            0,
            TOKEN,
            on_proposal=lambda: holds.append("hold"),
            confirmation_timeout_s=2.0,
        )
        server.start()
        connection = socket.create_connection(server.address, timeout=2.0)
        connection.settimeout(2.0)
        request_id = str(uuid4())
        proposal = {
            "protocol": 1,
            "type": "propose_goal",
            "request_id": request_id,
            "session_token": TOKEN,
            "goal": "  pick   up the red cup  ",
        }
        encoded = (json.dumps(proposal) + "\n").encode("utf-8")
        connection.sendall(encoded[:7])
        connection.sendall(encoded[7:])
        confirmation_required = read_json_line(connection)
        self.assertEqual(confirmation_required["type"], "confirmation_required")
        self.assertEqual(confirmation_required["goal"], "pick up the red cup")
        self.assertEqual(confirmation_required["robot_state"], "HOLD")
        self.assertEqual(holds, ["hold"])

        connection.sendall(
            (
                json.dumps(
                    {
                        "protocol": 1,
                        "type": "confirm_goal",
                        "request_id": request_id,
                        "confirmation_id": confirmation_required["confirmation_id"],
                        "session_token": TOKEN,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        command = wait_for_command(server)
        self.assertEqual(holds, ["hold", "hold"])
        self.assertEqual(command.goal, "pick up the red cup")
        self.assertTrue(server.accept_command(command))
        self.assertEqual(read_json_line(connection)["type"], "goal_accepted")
        connection.close()
        server.close()

    def test_invalid_token_and_mismatched_confirmation_fail_closed(self):
        server = VoiceCommandServer("127.0.0.1", 0, TOKEN, confirmation_timeout_s=2.0)
        server.start()
        connection = socket.create_connection(server.address, timeout=2.0)
        connection.settimeout(2.0)
        request_id = str(uuid4())
        connection.sendall(
            (
                json.dumps(
                    {
                        "protocol": 1,
                        "type": "propose_goal",
                        "request_id": request_id,
                        "session_token": "wrong-token-value",
                        "goal": "pick up the cup",
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        error = read_json_line(connection)
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["code"], "authentication_failed")
        connection.close()

        connection = socket.create_connection(server.address, timeout=2.0)
        connection.settimeout(2.0)
        connection.sendall(
            (
                json.dumps(
                    {
                        "protocol": 1,
                        "type": "propose_goal",
                        "request_id": request_id,
                        "session_token": TOKEN,
                        "goal": "pick up the cup",
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        read_json_line(connection)
        connection.sendall(
            (
                json.dumps(
                    {
                        "protocol": 1,
                        "type": "confirm_goal",
                        "request_id": request_id,
                        "confirmation_id": "not-the-server-value",
                        "session_token": TOKEN,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        error = read_json_line(connection)
        self.assertEqual(error["code"], "confirmation_mismatch")
        self.assertIsNone(server.poll_command())
        connection.close()
        server.close()

    def test_disconnect_after_confirmation_cannot_execute_queued_goal(self):
        server = VoiceCommandServer("127.0.0.1", 0, TOKEN, confirmation_timeout_s=2.0)
        server.start()
        connection = socket.create_connection(server.address, timeout=2.0)
        connection.settimeout(2.0)
        request_id = str(uuid4())
        connection.sendall(
            (
                json.dumps(
                    {
                        "protocol": 1,
                        "type": "propose_goal",
                        "request_id": request_id,
                        "session_token": TOKEN,
                        "goal": "pick up the wooden block",
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        required = read_json_line(connection)
        connection.sendall(
            (
                json.dumps(
                    {
                        "protocol": 1,
                        "type": "confirm_goal",
                        "request_id": request_id,
                        "confirmation_id": required["confirmation_id"],
                        "session_token": TOKEN,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        command = wait_for_command(server)
        connection.close()
        self.assertFalse(server.accept_command(command))
        self.assertTrue(command.cancelled)
        server.close()


if __name__ == "__main__":
    unittest.main()

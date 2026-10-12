import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock
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
        data = connection.recv(1)
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
    def make_server(self, on_proposal=None):
        server = VoiceCommandServer(
            "127.0.0.1", 0, TOKEN, on_proposal=on_proposal, confirmation_timeout_s=2.0,
        )
        server.start()
        self.addCleanup(server.close)
        return server

    def connect(self, server):
        connection = socket.create_connection(server.address, timeout=2.0)
        connection.settimeout(2.0)
        self.addCleanup(connection.close)
        return connection

    def send_message(self, connection, message_type, **fields):
        message = {
            "protocol": 1, "type": message_type, "request_id": str(uuid4()),
            "session_token": TOKEN, **fields,
        }
        connection.sendall((json.dumps(message) + "\n").encode())
        return message

    def propose(self, connection):
        self.send_message(connection, "propose_goal", goal="pick up the red cup")
        required = read_json_line(connection)
        self.assertEqual(required["type"], "confirmation_required")
        return required

    def confirm(self, connection, required):
        return self.send_message(
            connection, "confirm_goal", request_id=required["request_id"],
            confirmation_id=required["confirmation_id"],
        )

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
        self.assertEqual(confirmation_required["robot_state"], "HOLD_REQUESTED")
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

    def test_immediate_hold_is_authenticated_receipt_and_connection_is_reusable(self):
        holds = []
        server = self.make_server(lambda: holds.append("hold"))
        connection = self.connect(server)
        for _ in range(2):
            request = self.send_message(connection, "request_hold")
            reply = read_json_line(connection)
            self.assertEqual(reply, {
                "protocol": 1, "type": "hold_requested", "request_id": request["request_id"],
            })
        self.assertEqual(holds, ["hold", "hold"])
        required = self.propose(connection)
        self.confirm(connection, required)
        command = wait_for_command(server)
        self.assertTrue(server.accept_command(command))
        self.assertEqual(read_json_line(connection)["type"], "goal_accepted")
        self.send_message(connection, "request_hold")
        self.assertEqual(read_json_line(connection)["type"], "hold_requested")
        self.assertEqual(len(holds), 5)

    def test_unauthenticated_hold_does_not_invoke_controller(self):
        holds = []
        server = self.make_server(lambda: holds.append("hold"))
        connection = self.connect(server)
        self.send_message(connection, "request_hold", session_token="wrong-token-value")
        self.assertEqual(read_json_line(connection)["code"], "authentication_failed")
        self.assertEqual(holds, [])

    def test_hold_requires_a_controller_and_callback_failure_is_not_acknowledged(self):
        server = self.make_server()
        connection = self.connect(server)
        self.send_message(connection, "request_hold")
        self.assertEqual(read_json_line(connection)["code"], "controller_unavailable")
        server.set_on_proposal(mock.Mock(side_effect=RuntimeError("controller failed")))
        self.send_message(connection, "request_hold")
        self.assertEqual(read_json_line(connection)["code"], "controller_unavailable")
        callback = mock.Mock()
        server.set_on_proposal(callback)
        self.send_message(connection, "request_hold")
        self.assertEqual(read_json_line(connection)["type"], "hold_requested")
        callback.assert_called_once_with()

    def test_hold_cancels_preview_and_rejects_stale_confirmation_without_disconnect(self):
        holds = []
        server = self.make_server(lambda: holds.append("hold"))
        connection = self.connect(server)
        required = self.propose(connection)
        self.send_message(connection, "request_hold")
        self.assertEqual(read_json_line(connection)["type"], "hold_requested")
        self.confirm(connection, required)
        self.assertEqual(read_json_line(connection)["code"], "goal_cancelled")
        self.assertIsNone(server.poll_command())
        self.propose(connection)
        self.assertEqual(len(holds), 3)

    def test_hold_cancels_queued_and_runner_owned_commands(self):
        for runner_polled in (False, True):
            with self.subTest(runner_polled=runner_polled):
                server = self.make_server(lambda: None)
                connection = self.connect(server)
                required = self.propose(connection)
                self.confirm(connection, required)
                if runner_polled:
                    command = wait_for_command(server)
                else:
                    deadline = time.monotonic() + 2.0
                    while not server.command_pending and time.monotonic() < deadline:
                        time.sleep(0.005)
                    self.assertTrue(server.command_pending)
                self.send_message(connection, "request_hold")
                self.assertEqual(read_json_line(connection)["type"], "hold_requested")
                self.assertIsNone(server.poll_command())
                if runner_polled:
                    self.assertTrue(command.cancelled)
                    self.assertFalse(server.accept_command(command))

    def test_hold_takes_priority_over_co_buffered_confirmation(self):
        server = self.make_server(lambda: None)
        connection = self.connect(server)
        required = self.propose(connection)
        confirmation = {"protocol": 1, "type": "confirm_goal", "session_token": TOKEN,
                        "request_id": required["request_id"],
                        "confirmation_id": required["confirmation_id"]}
        hold = {"protocol": 1, "type": "request_hold", "session_token": TOKEN,
                "request_id": str(uuid4())}
        # One decoder feed models the confirm/HOLD combination, regardless of
        # the platform's TCP packet splitting. The callback also cancels a
        # confirmation processed in a preceding read before runner acceptance.
        connection.sendall((json.dumps(confirmation) + "\n" + json.dumps(hold) + "\n").encode())
        self.assertEqual(read_json_line(connection)["type"], "hold_requested")
        self.assertIsNone(server.poll_command())

    def test_second_connection_hold_invalidates_another_connections_preview(self):
        server = self.make_server(lambda: None)
        reviewing = self.connect(server)
        required = self.propose(reviewing)
        control = self.connect(server)
        self.send_message(control, "request_hold")
        self.assertEqual(read_json_line(control)["type"], "hold_requested")
        self.confirm(reviewing, required)
        self.assertEqual(read_json_line(reviewing)["code"], "goal_cancelled")
        self.assertIsNone(server.poll_command())
        self.propose(reviewing)

    def test_hold_cancels_acceptance_while_response_delivery_is_in_flight(self):
        server = self.make_server(lambda: None)
        connection = self.connect(server)
        control = self.connect(server)
        required = self.propose(connection)
        self.confirm(connection, required)
        command = wait_for_command(server)
        sending = threading.Event()
        resume_send = threading.Event()
        self.addCleanup(resume_send.set)
        original_send = server._send

        def delayed_send(sock, payload):
            if payload["type"] == "goal_accepted":
                sending.set()
                if not resume_send.wait(2.0):
                    raise AssertionError("test did not release response delivery")
            original_send(sock, payload)

        accepted = []
        with mock.patch.object(server, "_send", side_effect=delayed_send):
            runner = threading.Thread(target=lambda: accepted.append(server.accept_command(command)))
            runner.start()
            self.assertTrue(sending.wait(1.0))
            self.send_message(control, "request_hold")
            self.assertEqual(read_json_line(control)["type"], "hold_requested")
            runner.join(1.0)
            self.assertEqual(accepted, [False])
            self.assertTrue(command.cancelled)
            resume_send.set()
            read_json_line(connection)
        self.assertIsNone(server.poll_command())

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

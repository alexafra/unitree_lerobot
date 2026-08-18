"""Authenticated local TCP bridge for reviewed iOS voice goals.

The wire contract is deliberately tiny: newline-delimited UTF-8 JSON, one
proposal followed by one explicit confirmation.  This module never talks to
the policy or robot directly.  It hands a confirmed command to the runner and
waits for the runner to accept or reject it before replying to the phone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import json
import logging
import os
from pathlib import Path
import queue
import secrets
import select
import socket
import threading
import time
from typing import Callable
from uuid import UUID


LOGGER = logging.getLogger("eval_groot_g1.voice")
PROTOCOL_VERSION = 1
DEFAULT_VOICE_PORT = 8765
MAX_GOAL_CHARACTERS = 256
MAX_LINE_BYTES = 4096
CONFIRMATION_TIMEOUT_S = 180.0
SESSION_TOKEN_ENV = "GROOT_VOICE_SESSION_TOKEN"


class VoiceProtocolError(Exception):
    """A machine-readable protocol failure safe to report to the app."""

    def __init__(self, code: str, message: str, request_id: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id


class VoiceConnectionClosed(Exception):
    """The phone connection disappeared before the transaction completed."""


@dataclass
class VoiceCommand:
    request_id: str
    confirmation_id: str
    goal: str
    peer: str
    _decision: threading.Event = field(default_factory=threading.Event, repr=False)
    _accepted: bool = field(default=False, repr=False)
    _error_code: str = field(default="rejected", repr=False)
    _error_message: str = field(default="Voice goal was rejected by the controller", repr=False)
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    _delivery: threading.Event = field(default_factory=threading.Event, repr=False)
    _delivery_succeeded: bool = field(default=False, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()


class JsonLineDecoder:
    """Incrementally decode bounded newline-delimited JSON objects."""

    def __init__(self, *, max_line_bytes: int = MAX_LINE_BYTES):
        self._buffer = bytearray()
        self._max_line_bytes = max_line_bytes

    def feed(self, data: bytes) -> list[dict[str, object]]:
        self._buffer.extend(data)
        if len(self._buffer) > self._max_line_bytes and b"\n" not in self._buffer:
            raise VoiceProtocolError("line_too_large", "JSON message exceeds the size limit")
        messages: list[dict[str, object]] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            if newline > self._max_line_bytes:
                raise VoiceProtocolError("line_too_large", "JSON message exceeds the size limit")
            raw = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if not raw:
                continue
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise VoiceProtocolError("invalid_utf8", "Message is not valid UTF-8") from exc
            try:
                value = json.loads(decoded)
            except json.JSONDecodeError as exc:
                raise VoiceProtocolError("invalid_json", "Message is not valid JSON") from exc
            if not isinstance(value, dict):
                raise VoiceProtocolError("invalid_message", "JSON message must be an object")
            messages.append(value)
        if len(self._buffer) > self._max_line_bytes:
            raise VoiceProtocolError("line_too_large", "JSON message exceeds the size limit")
        return messages


def normalize_voice_goal(value: object) -> str:
    if not isinstance(value, str):
        raise VoiceProtocolError("invalid_goal", "goal must be text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise VoiceProtocolError("invalid_goal", "goal must contain only printable text")
    goal = " ".join(value.split())
    if not goal or len(goal) > MAX_GOAL_CHARACTERS:
        raise VoiceProtocolError("invalid_goal", "goal must be 1..256 printable characters")
    return goal


def load_voice_session_token(path: Path | None) -> str:
    if path is not None:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"Could not read voice session-token file {path}: {exc}") from exc
    else:
        token = os.environ.get(SESSION_TOKEN_ENV, "").strip()
    if not 16 <= len(token) <= 256 or any(ord(character) < 33 or ord(character) > 126 for character in token):
        source = str(path) if path is not None else SESSION_TOKEN_ENV
        raise ValueError(f"Voice session token from {source} must be 16..256 non-whitespace printable characters")
    return token


def _request_id(value: object) -> str:
    if not isinstance(value, str):
        raise VoiceProtocolError("invalid_request_id", "request_id must be a UUID string")
    try:
        UUID(value)
    except (ValueError, AttributeError) as exc:
        raise VoiceProtocolError("invalid_request_id", "request_id must be a UUID string", value) from exc
    return value


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class VoiceCommandServer:
    """Single-transaction-at-a-time TCP server for the iOS protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        session_token: str,
        *,
        warning_factory: Callable[[str], str] | None = None,
        on_proposal: Callable[[], None] | None = None,
        confirmation_timeout_s: float = CONFIRMATION_TIMEOUT_S,
    ):
        self.host = host
        self.port = port
        self._session_token = session_token
        self._warning_factory = warning_factory or (
            lambda _goal: "Review this goal carefully. Confirmation authorizes a fresh GR00T request."
        )
        self._on_proposal = on_proposal
        self._confirmation_timeout_s = confirmation_timeout_s
        self._commands: queue.Queue[VoiceCommand] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._active_lock = threading.Lock()
        self._active: VoiceCommand | None = None

    @property
    def address(self) -> tuple[str, int]:
        listener = self._listener
        if listener is None:
            return self.host, self.port
        host, port = listener.getsockname()[:2]
        return str(host), int(port)

    @property
    def command_pending(self) -> bool:
        return not self._commands.empty()

    def set_on_proposal(self, callback: Callable[[], None] | None) -> None:
        self._on_proposal = callback

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Voice command server is already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(2)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="groot-voice-server", daemon=True)
        self._thread.start()

    def poll_command(self) -> VoiceCommand | None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return None
            if not command.cancelled:
                return command

    def accept_command(
        self,
        command: VoiceCommand,
        *,
        service_callback: Callable[[], None] | None = None,
    ) -> bool:
        if command.cancelled or command._decision.is_set():
            return False
        command._accepted = True
        command._decision.set()
        deadline = time.monotonic() + 2.0
        try:
            while not command._delivery.wait(timeout=0.02):
                if service_callback is not None:
                    service_callback()
                if time.monotonic() >= deadline:
                    command._cancelled.set()
                    return False
        except Exception:
            command._cancelled.set()
            raise
        return command._delivery_succeeded and not command.cancelled

    def reject_command(self, command: VoiceCommand, code: str, message: str) -> bool:
        if command.cancelled or command._decision.is_set():
            return False
        command._error_code = code
        command._error_message = message
        command._decision.set()
        return True

    def close(self) -> None:
        self._stop.set()
        with self._active_lock:
            active = self._active
            if active is not None:
                active._cancelled.set()
                active._decision.set()
                active._delivery.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                connection, address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                LOGGER.exception("Voice listener failed")
                return
            peer = f"{address[0]}:{address[1]}"
            try:
                self._handle_connection(connection, peer)
            except VoiceConnectionClosed:
                LOGGER.info("Voice connection %s closed; pending proposal discarded", peer)
            except Exception:
                LOGGER.exception("Voice connection %s failed", peer)
            finally:
                try:
                    connection.close()
                except OSError:
                    pass

    def _handle_connection(self, connection: socket.socket, peer: str) -> None:
        connection.settimeout(0.2)
        decoder = JsonLineDecoder()
        buffered: list[dict[str, object]] = []
        while not self._stop.is_set():
            try:
                proposal = self._read_message(
                    connection,
                    decoder,
                    buffered,
                    time.monotonic() + self._confirmation_timeout_s,
                )
            except VoiceProtocolError as exc:
                self._send_error(connection, exc)
                return
            request_id = ""
            try:
                request_id, goal = self._validate_proposal(proposal)
                confirmation_id = secrets.token_urlsafe(24)
                self._request_powered_hold(request_id)
                self._send(
                    connection,
                    {
                        "protocol": PROTOCOL_VERSION,
                        "type": "confirmation_required",
                        "request_id": request_id,
                        "confirmation_id": confirmation_id,
                        "goal": goal,
                        "robot_state": "HOLD",
                        "warning": self._warning_factory(goal),
                    },
                )
                deadline = time.monotonic() + self._confirmation_timeout_s
                confirmation = self._read_message(connection, decoder, buffered, deadline)
                self._validate_confirmation(confirmation, request_id, confirmation_id)
                # A terminal command may have started while the phone was
                # reviewing the proposal. Reassert HOLD when confirmation
                # arrives so a late confirmation cannot leave motion active.
                self._request_powered_hold(request_id)
                command = VoiceCommand(request_id, confirmation_id, goal, peer)
                with self._active_lock:
                    if self._active is not None:
                        raise VoiceProtocolError("controller_busy", "Another voice command is pending", request_id)
                    self._active = command
                try:
                    self._commands.put_nowait(command)
                except queue.Full as exc:
                    with self._active_lock:
                        self._active = None
                    raise VoiceProtocolError("controller_busy", "Another voice command is pending", request_id) from exc
                self._wait_for_decision(connection, command, deadline)
                if command.cancelled:
                    raise VoiceConnectionClosed
                if command._accepted:
                    try:
                        self._send(
                            connection,
                            {
                                "protocol": PROTOCOL_VERSION,
                                "type": "goal_accepted",
                                "request_id": request_id,
                            },
                        )
                    except VoiceConnectionClosed:
                        command._cancelled.set()
                        raise
                    else:
                        command._delivery_succeeded = True
                    finally:
                        command._delivery.set()
                else:
                    self._send_error(
                        connection,
                        VoiceProtocolError(command._error_code, command._error_message, request_id),
                    )
                with self._active_lock:
                    if self._active is command:
                        self._active = None
            except VoiceConnectionClosed:
                with self._active_lock:
                    active = self._active
                    if active is not None and active.request_id == request_id:
                        active._cancelled.set()
                        active._decision.set()
                        active._delivery.set()
                        self._active = None
                raise
            except VoiceProtocolError as exc:
                with self._active_lock:
                    active = self._active
                    if active is not None and active.request_id == request_id:
                        active._cancelled.set()
                        active._decision.set()
                        active._delivery.set()
                        self._active = None
                self._send_error(connection, exc)
                return

    def _request_powered_hold(self, request_id: str) -> None:
        callback = self._on_proposal
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            raise VoiceProtocolError(
                "controller_unavailable",
                "Controller could not enter powered HOLD",
                request_id,
            ) from exc

    def _read_message(
        self,
        connection: socket.socket,
        decoder: JsonLineDecoder,
        buffered: list[dict[str, object]],
        deadline: float | None,
    ) -> dict[str, object]:
        while not self._stop.is_set():
            if buffered:
                return buffered.pop(0)
            if deadline is not None and time.monotonic() >= deadline:
                raise VoiceProtocolError("confirmation_timeout", "Confirmation timed out")
            try:
                data = connection.recv(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                raise VoiceConnectionClosed from exc
            if not data:
                raise VoiceConnectionClosed
            buffered.extend(decoder.feed(data))
        raise VoiceConnectionClosed

    def _wait_for_decision(self, connection: socket.socket, command: VoiceCommand, deadline: float) -> None:
        while not self._stop.is_set() and not command._decision.wait(0.05):
            if time.monotonic() >= deadline:
                command._cancelled.set()
                raise VoiceProtocolError("controller_timeout", "Controller did not accept the goal in time", command.request_id)
            try:
                readable, _, _ = select.select([connection], [], [], 0.0)
                if readable and connection.recv(1, socket.MSG_PEEK) == b"":
                    command._cancelled.set()
                    raise VoiceConnectionClosed
            except OSError as exc:
                command._cancelled.set()
                raise VoiceConnectionClosed from exc
        # The wire protocol has no client ACK after confirm_goal. Give a local
        # TCP FIN a short opportunity to surface before reporting acceptance;
        # this closes the common confirm-then-disconnect race without adding a
        # third app message or delaying robot execution materially.
        try:
            readable, _, _ = select.select([connection], [], [], 0.05)
            if readable and connection.recv(1, socket.MSG_PEEK) == b"":
                command._cancelled.set()
                raise VoiceConnectionClosed
        except OSError as exc:
            command._cancelled.set()
            raise VoiceConnectionClosed from exc

    def _validate_common(self, value: dict[str, object], expected_type: str) -> str:
        request_id = _request_id(value.get("request_id"))
        if value.get("protocol") != PROTOCOL_VERSION or value.get("type") != expected_type:
            raise VoiceProtocolError("invalid_message", f"Expected protocol 1 {expected_type}", request_id)
        token = value.get("session_token")
        if not isinstance(token, str) or not hmac.compare_digest(token, self._session_token):
            raise VoiceProtocolError("authentication_failed", "Invalid session token", request_id)
        return request_id

    def _validate_proposal(self, value: dict[str, object]) -> tuple[str, str]:
        request_id = self._validate_common(value, "propose_goal")
        return request_id, normalize_voice_goal(value.get("goal"))

    def _validate_confirmation(
        self,
        value: dict[str, object],
        request_id: str,
        confirmation_id: str,
    ) -> None:
        confirmed_request_id = self._validate_common(value, "confirm_goal")
        if confirmed_request_id != request_id or value.get("confirmation_id") != confirmation_id:
            raise VoiceProtocolError("confirmation_mismatch", "Confirmation does not match the proposal", request_id)

    @staticmethod
    def _send(connection: socket.socket, value: dict[str, object]) -> None:
        try:
            connection.sendall(_json_bytes(value))
        except OSError as exc:
            raise VoiceConnectionClosed from exc

    def _send_error(self, connection: socket.socket, error: VoiceProtocolError) -> None:
        try:
            self._send(
                connection,
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "error",
                    "request_id": error.request_id,
                    "code": error.code,
                    "message": error.message,
                },
            )
        except VoiceConnectionClosed:
            pass

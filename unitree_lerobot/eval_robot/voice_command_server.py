"""Authenticated local TCP bridge for reviewed iOS voice goals.

The wire contract is deliberately tiny: newline-delimited UTF-8 JSON, one
proposal followed by one explicit confirmation, or an authenticated immediate
HOLD request. This module never talks to
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
    _hold_generation: int = field(default=0, repr=False)
    _decision_at: float | None = field(default=None, repr=False)

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
    """One queued goal at a time, with HOLD available on every connection."""

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
        self._hold_generation = 0
        self._connections: set[socket.socket] = set()
        self._client_threads: set[threading.Thread] = set()

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
        with self._active_lock:
            if (command.cancelled or command._decision.is_set()
                    or command._hold_generation != self._hold_generation):
                return False
            command._accepted = True
            command._decision_at = time.monotonic()
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
        with self._active_lock:
            return (command._delivery_succeeded and not command.cancelled
                    and command._hold_generation == self._hold_generation)

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
                self._cancel_command(active)
            connections = tuple(self._connections)
            client_threads = tuple(self._client_threads)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for thread in client_threads:
            thread.join(timeout=1.0)

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
            thread = threading.Thread(
                target=self._serve_connection, args=(connection, peer),
                name="groot-voice-client", daemon=True,
            )
            with self._active_lock:
                self._connections.add(connection)
                self._client_threads.add(thread)
            thread.start()

    def _serve_connection(self, connection: socket.socket, peer: str) -> None:
        try:
            self._handle_connection(connection, peer)
        except VoiceConnectionClosed:
            LOGGER.info("Voice connection %s closed; pending proposal discarded", peer)
        except Exception:
            LOGGER.exception("Voice connection %s failed", peer)
        finally:
            connection.close()
            with self._active_lock:
                self._connections.discard(connection)
                self._client_threads.discard(threading.current_thread())

    @staticmethod
    def _cancel_command(command: VoiceCommand) -> None:
        command._cancelled.set()
        command._decision.set()
        command._delivery.set()

    def _request_hold(self, connection: socket.socket, message: dict[str, object]) -> None:
        request_id = self._validate_common(message, "request_hold")
        # Cancel before invoking the callback: even a failed controller request
        # must not leave an older confirmed goal eligible for execution.
        with self._active_lock:
            self._hold_generation += 1
            if self._active is not None:
                self._cancel_command(self._active)
                self._active = None
            while True:
                try:
                    self._cancel_command(self._commands.get_nowait())
                except queue.Empty:
                    break
        self._request_powered_hold(request_id, require_callback=True)
        self._send(connection, {
            "protocol": PROTOCOL_VERSION,
            "type": "hold_requested",
            "request_id": request_id,
        })

    def _handle_connection(self, connection: socket.socket, peer: str) -> None:
        connection.settimeout(0.02)
        decoder = JsonLineDecoder()
        buffered: list[dict[str, object]] = []
        command: VoiceCommand | None = None
        confirmed = False
        retired: set[str] = set()
        deadline = time.monotonic() + self._confirmation_timeout_s
        try:
            while not self._stop.is_set():
                # A different authenticated connection may have requested HOLD.
                with self._active_lock:
                    if command is not None and command._hold_generation != self._hold_generation:
                        retired.add(command.request_id)
                        self._cancel_command(command)
                        command = None
                        confirmed = False
                        deadline = time.monotonic() + self._confirmation_timeout_s
                try:
                    data = connection.recv(4096)
                except socket.timeout:
                    data = None
                except OSError as exc:
                    raise VoiceConnectionClosed from exc
                if data == b"":
                    raise VoiceConnectionClosed
                if data:
                    buffered.extend(decoder.feed(data))
                while buffered:
                    # Prioritize a HOLD already received in the same TCP read.
                    # In particular, confirm_goal + request_hold must not put
                    # the now-cancelled goal in the runner's queue first.
                    hold_index = next((i for i, item in enumerate(buffered)
                                       if item.get("type") == "request_hold"), None)
                    if hold_index is not None:
                        message = buffered[hold_index]
                        self._validate_common(message, "request_hold")
                        for discarded in buffered[:hold_index]:
                            if isinstance(discarded.get("request_id"), str):
                                retired.add(discarded["request_id"])
                        del buffered[:hold_index + 1]
                        if command is not None:
                            retired.add(command.request_id)
                        try:
                            self._request_hold(connection, message)
                        except VoiceProtocolError as exc:
                            self._send_error(connection, exc)
                        command = None
                        confirmed = False
                        deadline = time.monotonic() + self._confirmation_timeout_s
                        continue
                    message = buffered.pop(0)
                    if message.get("type") == "confirm_goal":
                        request_id = self._validate_common(message, "confirm_goal")
                        if request_id in retired:
                            self._send_error(connection, VoiceProtocolError(
                                "goal_cancelled", "HOLD cancelled this goal; propose it again to resume", request_id,
                            ))
                            continue
                    if command is None:
                        request_id, goal = self._validate_proposal(message)
                        retired.discard(request_id)
                        with self._active_lock:
                            generation = self._hold_generation
                        self._request_powered_hold(request_id)
                        command = VoiceCommand(request_id, secrets.token_urlsafe(24), goal, peer,
                                               _hold_generation=generation)
                        self._send(connection, {
                            "protocol": PROTOCOL_VERSION,
                            "type": "confirmation_required",
                            "request_id": request_id,
                            "confirmation_id": command.confirmation_id,
                            "goal": goal,
                            "robot_state": "HOLD_REQUESTED",
                            "warning": self._warning_factory(goal),
                        })
                        deadline = time.monotonic() + self._confirmation_timeout_s
                    else:
                        if confirmed:
                            raise VoiceProtocolError("controller_busy", "A confirmed goal is pending", command.request_id)
                        self._validate_confirmation(message, command.request_id, command.confirmation_id)
                        with self._active_lock:
                            if command._hold_generation != self._hold_generation:
                                retired.add(command.request_id)
                                self._cancel_command(command)
                                self._send_error(connection, VoiceProtocolError(
                                    "goal_cancelled", "HOLD cancelled this goal; propose it again to resume",
                                    command.request_id,
                                ))
                                command = None
                                continue
                            if self._active is not None:
                                raise VoiceProtocolError("controller_busy", "Another voice command is pending", command.request_id)
                            self._request_powered_hold(command.request_id)
                            self._active = command
                            try:
                                self._commands.put_nowait(command)
                            except queue.Full as exc:
                                raise VoiceProtocolError("controller_busy", "Another voice command is pending", command.request_id) from exc
                            confirmed = True
                if command is not None and confirmed and command._decision.is_set():
                    with self._active_lock:
                        cancelled = command.cancelled or command._hold_generation != self._hold_generation
                    if not cancelled:
                        if command._accepted:
                            # Preserve the previous short FIN-detection grace
                            # before authorizing execution. Unlike a blocking
                            # wait, this loop continues servicing HOLD traffic.
                            if (command._decision_at is not None
                                    and time.monotonic() - command._decision_at < 0.05):
                                continue
                            self._send(connection, {
                                "protocol": PROTOCOL_VERSION,
                                "type": "goal_accepted",
                                "request_id": command.request_id,
                            })
                            with self._active_lock:
                                command._delivery_succeeded = (
                                    not command.cancelled and command._hold_generation == self._hold_generation
                                )
                        else:
                            self._send_error(connection, VoiceProtocolError(
                                command._error_code, command._error_message, command.request_id,
                            ))
                    command._delivery.set()
                    with self._active_lock:
                        if self._active is command:
                            self._active = None
                    command = None
                    confirmed = False
                    deadline = time.monotonic() + self._confirmation_timeout_s
                if time.monotonic() >= deadline:
                    raise VoiceProtocolError(
                        "controller_timeout" if confirmed else "confirmation_timeout",
                        "Controller did not accept the goal in time" if confirmed else "Confirmation timed out",
                        "" if command is None else command.request_id,
                    )
        except VoiceProtocolError as exc:
            self._send_error(connection, exc)
        finally:
            if command is not None:
                with self._active_lock:
                    self._cancel_command(command)
                    if self._active is command:
                        self._active = None

    def _request_powered_hold(self, request_id: str, *, require_callback: bool = False) -> None:
        callback = self._on_proposal
        if callback is None:
            if require_callback:
                raise VoiceProtocolError("controller_unavailable", "No active robot controller can receive HOLD", request_id)
            return
        try:
            callback()
        except Exception as exc:
            raise VoiceProtocolError(
                "controller_unavailable",
                "Controller could not enter powered HOLD",
                request_id,
            ) from exc

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

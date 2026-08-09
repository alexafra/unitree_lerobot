"""Small Python 3.10 client for an Isaac-GR00T policy server.

The model stays in Isaac-GR00T's Python environment.  This module implements only
the msgpack/ZeroMQ wire protocol needed by the Unitree deployment process.
"""

from __future__ import annotations

import functools
import io
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class DeploymentError(RuntimeError):
    """Configuration, protocol, or safety error that should stop deployment."""


class MsgSerializer:
    """GR00T-compatible numeric serializer without importing the GR00T package."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(MsgSerializer._safe_encode, chain=lambda obj: obj)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(MsgSerializer._safe_decode, chain=lambda obj: obj)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj: Any, chain: Any = None) -> Any:
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError("Refusing to serialize an object-dtype ndarray")
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj: Any, chain: Any = None) -> Any:
        if isinstance(obj, dict):
            marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
            if marker:
                payload = obj.get("as_npy", obj.get(b"as_npy"))
                if payload is None:
                    raise ValueError("Malformed ndarray payload: missing 'as_npy'")
                return np.load(io.BytesIO(payload), allow_pickle=False)

            nd_value = obj.get(b"nd", obj.get("nd"))
            kind_value = obj.get(b"kind", obj.get("kind"))
            if nd_value and kind_value in (b"O", "O"):
                raise ValueError("Refusing to deserialize an object-dtype ndarray")

            modality_marker = obj.get("__ModalityConfig__", obj.get(b"__ModalityConfig__"))
            if modality_marker:
                payload = obj.get("as_json", obj.get(b"as_json"))
                if not isinstance(payload, dict):
                    raise ValueError("Malformed ModalityConfig payload")
                return payload

        return mnp.decode(obj, chain=chain)


class Gr00tClient:
    """Synchronous, timeout-bounded client for ``PolicyServer``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 5000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._open_socket()

    def _open_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        if self._socket is None:
            raise DeploymentError("GR00T client is closed")
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self._socket.recv())
        except zmq.error.Again as exc:
            self._open_socket()
            raise TimeoutError(
                f"GR00T server {self.host}:{self.port} did not respond within {self.timeout_ms} ms"
            ) from exc
        except zmq.ZMQError as exc:
            self._open_socket()
            raise DeploymentError(f"GR00T transport failed: {exc}") from exc
        if isinstance(response, dict) and "error" in response:
            raise DeploymentError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> bool:
        response = self.call("ping")
        return isinstance(response, dict) and response.get("status") == "ok"

    def reset(self) -> None:
        self.call("reset", {"options": None})

    def get_modality_config(self) -> dict[str, dict[str, Any]]:
        response = self.call("get_modality_config")
        if not isinstance(response, dict):
            raise DeploymentError("GR00T server returned an invalid modality configuration")
        return response

    def get_policy_metadata(self) -> dict[str, Any]:
        response = self.call("get_policy_metadata")
        if not isinstance(response, dict):
            raise DeploymentError("GR00T server returned invalid policy metadata")
        return response

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self.call("get_action", {"observation": observation, "options": None})
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise DeploymentError("GR00T server returned an invalid action response")
        action, info = response
        if not isinstance(action, dict) or not isinstance(info, dict):
            raise DeploymentError("GR00T server returned invalid action/info objects")
        return action

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._context.term()

    def __enter__(self) -> "Gr00tClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

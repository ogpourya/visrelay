"""Thread-safe holder for the latest encoded JPEG frame."""

from __future__ import annotations

import threading
import time


class FrameStore:
    """Holds the latest JPEG bytes; MJPEG handlers wait on `condition`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._timestamp = 0.0
        self.width = 0
        self.height = 0
        self.encode_ms = 0.0

    @property
    def condition(self) -> threading.Condition:
        return self._condition

    def publish(self, jpeg: bytes, width: int = 0, height: int = 0, encode_ms: float = 0.0) -> int:
        with self._condition:
            self._jpeg = jpeg
            self._frame_id += 1
            self._timestamp = time.monotonic()
            if width:
                self.width = width
            if height:
                self.height = height
            self.encode_ms = encode_ms
            self._condition.notify_all()
            return self._frame_id

    def snapshot(self) -> tuple[bytes | None, int, float]:
        with self._lock:
            return self._jpeg, self._frame_id, self._timestamp

    def wait_for(self, last_id: int, timeout: float = 5.0) -> tuple[bytes | None, int, float]:
        """Block until a frame newer than last_id arrives (or timeout)."""
        with self._condition:
            if self._frame_id != last_id and self._jpeg is not None:
                return self._jpeg, self._frame_id, self._timestamp
            self._condition.wait_for(lambda: self._frame_id != last_id, timeout=timeout)
            return self._jpeg, self._frame_id, self._timestamp

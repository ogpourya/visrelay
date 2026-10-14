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
        self._closed = False
        self.close_reason = ""
        self._viewers = 0
        self._last_viewer_ts = 0.0

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

    def close(self, reason: str = "") -> None:
        """Mark the source as gone; wakes MJPEG waiters so they exit."""
        with self._condition:
            if not self._closed:
                self._closed = True
                self.close_reason = reason
                self._condition.notify_all()

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def viewers(self) -> int:
        with self._lock:
            return self._viewers

    def viewer_open(self) -> int:
        with self._condition:
            self._viewers += 1
            self._condition.notify_all()
            return self._viewers

    def viewer_close(self) -> int:
        with self._condition:
            if self._viewers > 0:
                self._viewers -= 1
            if self._viewers == 0:
                self._last_viewer_ts = time.monotonic()
                self._condition.notify_all()
            return self._viewers

    def wants_capture(self, idle_timeout: float) -> bool:
        """True while a stream viewer is connected (or within linger)."""
        with self._lock:
            if self._closed:
                return False
            if self._viewers > 0:
                return True
            if self._last_viewer_ts == 0.0:
                return False
            return (time.monotonic() - self._last_viewer_ts) < idle_timeout

    def stalled(self, publish_timeout: float, idle_timeout: float) -> bool:
        """True if capture is demanded but no frame was published lately.

        Never fires while idle-paused (no demand) or before the first
        frame: those are normal states, not stalls.
        """
        with self._lock:
            if self._closed or self._frame_id == 0:
                return False
            if self._viewers > 0:
                demand = True
            elif self._last_viewer_ts == 0.0:
                return False
            else:
                demand = (time.monotonic() - self._last_viewer_ts) < idle_timeout
            if not demand:
                return False
            return (time.monotonic() - self._timestamp) > publish_timeout

    def wait_for(self, last_id: int, timeout: float = 5.0) -> tuple[bytes | None, int, float]:
        """Block until a frame newer than last_id arrives, closes, or timeout."""
        with self._condition:
            if self._frame_id != last_id and self._jpeg is not None:
                return self._jpeg, self._frame_id, self._timestamp
            self._condition.wait_for(
                lambda: (self._frame_id != last_id and self._jpeg is not None) or self._closed,
                timeout=timeout,
            )
            return self._jpeg, self._frame_id, self._timestamp

"""Thread-safe holder for the latest encoded JPEG frame."""

from __future__ import annotations

import threading
import time


class FrameStore:
    """Holds the latest JPEG bytes; MJPEG handlers wait on `condition`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._demand_cond = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._mjpeg_part: bytes | None = None
        self._frame_id = 0
        self._timestamp = 0.0
        self.width = 0
        self.height = 0
        self.encode_ms = 0.0
        self._closed = False
        self.close_reason = ""
        self._viewers = 0
        # Start warm: capture the initial frame immediately so width/height
        # populate on startup instead of starting cold-paused.
        self._last_viewer_ts = time.monotonic()

    @property
    def condition(self) -> threading.Condition:
        return self._condition

    @property
    def demand_condition(self) -> threading.Condition:
        return self._demand_cond

    def _wants_locked(self, idle_timeout: float) -> bool:
        if self._closed:
            return False
        if self._viewers > 0:
            return True
        # Warmup: capture until the first frame exists.
        if self._frame_id == 0:
            return True
        if self._last_viewer_ts == 0.0:
            return False
        return (time.monotonic() - self._last_viewer_ts) < idle_timeout

    def publish(self, jpeg: bytes, width: int = 0, height: int = 0, encode_ms: float = 0.0) -> int:
        with self._condition:
            self._jpeg = jpeg
            self._mjpeg_part = (
                b"--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
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

    def snapshot_fresh(self, max_age: float = 1.0, timeout: float = 5.0) -> bytes | None:
        """Return a fresh JPEG, waiting for capture to wake from idle if stale."""
        self.viewer_open()
        try:
            with self._lock:
                jpeg = self._jpeg
                fid = self._frame_id
                ts = self._timestamp
                closed = self._closed
            if closed:
                return jpeg
            if jpeg is not None and (time.monotonic() - ts) <= max_age:
                return jpeg
            jpeg2, fid2, _ = self.wait_for(fid, timeout=timeout)
            if fid2 != fid and jpeg2 is not None:
                return jpeg2
            # Timeout without a new frame: never serve the stale pre-idle frame.
            return None
        finally:
            self.viewer_close()

    def status(self) -> tuple[int, float, int, int, int, float, bool]:
        """Snapshot of store metadata under one lock (avoids repeated locking)."""
        with self._lock:
            return (
                self._frame_id,
                self._timestamp,
                self._viewers,
                self.width,
                self.height,
                self.encode_ms,
                self._closed,
            )

    def close(self, reason: str = "") -> None:
        """Mark the source as gone; wakes MJPEG waiters so they exit."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self.close_reason = reason
                self._condition.notify_all()
                self._demand_cond.notify_all()

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def viewers(self) -> int:
        with self._lock:
            return self._viewers

    def viewer_open(self) -> int:
        with self._demand_cond:
            self._viewers += 1
            self._demand_cond.notify_all()
            return self._viewers

    def viewer_close(self) -> int:
        with self._demand_cond:
            if self._viewers > 0:
                self._viewers -= 1
                if self._viewers == 0:
                    self._last_viewer_ts = time.monotonic()
                    self._demand_cond.notify_all()
            return self._viewers

    def wants_capture(self, idle_timeout: float) -> bool:
        """True while a stream viewer is connected (or within linger)."""
        with self._lock:
            return self._wants_locked(idle_timeout)

    def wait_for_demand(self, idle_timeout: float, timeout: float = 0.2) -> bool:
        """Wait on demand changes; True if capture wanted or store closed."""
        with self._demand_cond:
            if self._closed or self._wants_locked(idle_timeout):
                return True
            self._demand_cond.wait(timeout=timeout)
            return self._closed or self._wants_locked(idle_timeout)

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
        """Block until a frame newer than last_id arrives, closes, or timeout.

        On timeout with no new frame (frame_id == last_id) returns
        (None, frame_id, timestamp) so callers never resend stale frames.
        """
        with self._condition:
            if self._frame_id != last_id and self._jpeg is not None:
                return self._jpeg, self._frame_id, self._timestamp
            self._condition.wait_for(
                lambda: (self._frame_id != last_id and self._jpeg is not None) or self._closed,
                timeout=timeout,
            )
            if self._frame_id != last_id and self._jpeg is not None:
                return self._jpeg, self._frame_id, self._timestamp
            if self._frame_id == last_id:
                return None, self._frame_id, self._timestamp
            return self._jpeg, self._frame_id, self._timestamp

    def wait_for_part(
        self, last_id: int, timeout: float = 5.0
    ) -> tuple[bytes | None, bytes | None, int, float]:
        """Block until newer frame arrives; returns (part, jpeg, id, ts).

        On timeout with no new frame returns (None, None, frame_id,
        timestamp) so MJPEG handlers flush/probe instead of resending.
        """
        with self._condition:
            if self._frame_id != last_id and self._jpeg is not None:
                return self._mjpeg_part, self._jpeg, self._frame_id, self._timestamp
            self._condition.wait_for(
                lambda: (self._frame_id != last_id and self._jpeg is not None) or self._closed,
                timeout=timeout,
            )
            if self._frame_id != last_id and self._jpeg is not None:
                return self._mjpeg_part, self._jpeg, self._frame_id, self._timestamp
            if self._frame_id == last_id:
                return None, None, self._frame_id, self._timestamp
            return self._mjpeg_part, self._jpeg, self._frame_id, self._timestamp

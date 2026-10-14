"""Background capture loop: PipeWire -> JPEG -> FrameStore.

Paced to a target FPS (default 24) to leave headroom for HTTP serving.
JPEG encoding uses simplejpeg (libjpeg-turbo SIMD). The loop idles
(no pulls, no encodes) while no stream viewer is connected.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import simplejpeg

from .frames import FrameStore


# Feed-stall detection: a healthy PipeWire feed delivers a first frame
# quickly and never goes silent (None) for long while demanded. Static
# screens still deliver repeat frames, so these can't false-positive.
NONE_STALL_S = 10.0
FIRST_FRAME_S = 15.0


def encode_frame(frame: np.ndarray, quality: int) -> bytes:
    channels = frame.shape[2] if frame.ndim == 3 else 4
    colorspace = "RGB" if channels == 3 else "BGRA"
    # Avoid a copy when already C-contiguous (the common case).
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return bytes(
        simplejpeg.encode_jpeg(
            frame,
            quality=quality,
            colorspace=colorspace,
            fastdct=True,
        )
    )


class CaptureWorker(threading.Thread):
    """Pulls frames from CaptureStream, encodes, publishes. Daemon thread."""

    def __init__(
        self,
        stream,
        store: FrameStore,
        *,
        fps: float = 24.0,
        quality: int = 80,
        width: int = 0,
        height: int = 0,
        idle_timeout: float = 5.0,
    ) -> None:
        super().__init__(name="visrelay-capture", daemon=True)
        self._stream = stream
        self._store = store
        self._interval = 1.0 / fps if fps > 0 else 0.0
        self._quality = quality
        self._width = width
        self._height = height
        self._stop = threading.Event()
        self.frames = 0
        self.errors = 0
        self.ended = threading.Event()
        self.abrupt = False
        self.stalled = False
        self._idle_timeout = idle_timeout
        self._last_jpeg: bytes | None = None
        self._last_shape = None
        self._last_sample: bytes | None = None

    def stop(self) -> None:
        self._stop.set()
        try:
            with self._store.condition:
                self._store.condition.notify_all()
        except Exception:
            pass

    def _encode_if_changed(self, frame: np.ndarray) -> tuple[bytes, float]:
        """Encode, unless the buffer matches the last encoded one.

        Compares a 16KB stride sample (zero-alloc view, memcmp-fast) and
        re-serves the cached JPEG on match. Static desktops skip the
        ~25ms re-encode; a sub-stride change may lag one frame behind.
        """
        shape = frame.shape
        sample = None
        if frame.flags["C_CONTIGUOUS"]:
            sample = frame.reshape(-1)[::512].tobytes()
        if (
            self._last_jpeg is not None
            and sample is not None
            and self._last_sample is not None
            and shape == self._last_shape
            and sample == self._last_sample
        ):
            return self._last_jpeg, 0.0
        t0 = time.perf_counter()
        jpeg = encode_frame(frame, self._quality)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        self._last_jpeg = jpeg
        self._last_shape = shape
        self._last_sample = sample
        return jpeg, encode_ms

    def run(self) -> None:
        stream = self._stream
        abrupt = False
        stall = False
        try:
            # Wait briefly for the first buffer (mirrors main.py behaviour).
            frame = None
            no_frame_since: float | None = None
            while frame is None and not self._stop.is_set():
                if not self._store.wants_capture(self._idle_timeout):
                    no_frame_since = None
                    if getattr(stream, "window_invalid", False):
                        abrupt = True
                        break
                    with self._store.condition:
                        self._store.condition.wait(timeout=0.2)
                    continue
                if getattr(stream, "window_invalid", False):
                    abrupt = True
                    break
                try:
                    frame = stream.get_frame()
                except Exception:
                    self.errors += 1
                    abrupt = True
                    break
                if frame is None:
                    if no_frame_since is None:
                        no_frame_since = time.monotonic()
                    elif time.monotonic() - no_frame_since > FIRST_FRAME_S:
                        stall = True
                        break
                    time.sleep(0.005)
                else:
                    no_frame_since = None

            if abrupt or stall:
                return

            last_frame = frame
            none_since: float | None = None
            while not self._stop.is_set():
                if not self._store.wants_capture(self._idle_timeout):
                    none_since = None
                    if getattr(stream, "window_invalid", False) or getattr(stream, "error", None):
                        abrupt = True
                        break
                    with self._store.condition:
                        self._store.condition.wait(timeout=0.2)
                    continue
                if getattr(stream, "window_invalid", False):
                    abrupt = True
                    break
                if getattr(stream, "error", None):
                    abrupt = True
                    break
                loop_start = time.perf_counter()
                try:
                    latest = stream.get_frame()
                except Exception:
                    self.errors += 1
                    abrupt = True
                    break
                if latest is None:
                    if none_since is None:
                        none_since = time.monotonic()
                    elif time.monotonic() - none_since > NONE_STALL_S:
                        stall = True
                        break
                else:
                    none_since = None
                    last_frame = latest
                if last_frame is None:
                    time.sleep(0.005)
                    continue
                h, w = last_frame.shape[0], last_frame.shape[1]
                try:
                    jpeg, encode_ms = self._encode_if_changed(last_frame)
                except Exception:
                    self.errors += 1
                    time.sleep(0.01)
                    continue
                self._store.publish(jpeg, width=w or self._width, height=h or self._height, encode_ms=encode_ms)
                self.frames += 1
                elapsed = time.perf_counter() - loop_start
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    # Wake early on stop().
                    self._stop.wait(sleep_time)
        finally:
            if not self._stop.is_set():
                if abrupt:
                    self.abrupt = True
                    self._store.close("sharing stopped")
                elif stall:
                    self.stalled = True
                    self._store.close("capture stalled")
            self.ended.set()


class DemoWorker(threading.Thread):
    """Synthetic test pattern generator (no Wayland needed). --demo mode."""

    def __init__(self, store: FrameStore, *, fps: float = 24.0, quality: int = 80,
                 width: int = 640, height: int = 360, idle_timeout: float = 5.0) -> None:
        super().__init__(name="visrelay-demo", daemon=True)
        self._store = store
        self._idle_timeout = idle_timeout
        self._interval = 1.0 / fps if fps > 0 else 0.0
        self._quality = quality
        self._width = width
        self._height = height
        self._stop = threading.Event()
        self.frames = 0

    def stop(self) -> None:
        self._stop.set()
        try:
            with self._store.condition:
                self._store.condition.notify_all()
        except Exception:
            pass

    def run(self) -> None:
        n = 0
        w, h = self._width, self._height
        # Precompute gradients once; per frame is just modular adds (no arange).
        xs = np.arange(w, dtype=np.uint16)
        ys = (np.arange(h, dtype=np.uint16)[:, None] * 255 // max(h - 1, 1))
        while not self._stop.is_set():
            if not self._store.wants_capture(self._idle_timeout):
                with self._store.condition:
                    self._store.condition.wait(timeout=0.2)
                continue
            loop_start = time.perf_counter()
            frame = np.empty((h, w, 3), dtype=np.uint8)
            frame[:, :, 0] = ((xs + n * 8) % 256).astype(np.uint8)
            frame[:, :, 1] = ((ys + n * 2) % 256).astype(np.uint8)
            frame[:, :, 2] = (n * 5) % 256
            try:
                jpeg = encode_frame(frame, self._quality)
            except Exception:
                time.sleep(0.05)
                continue
            self._store.publish(jpeg, width=w, height=h)
            self.frames += 1
            n += 1
            elapsed = time.perf_counter() - loop_start
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                self._stop.wait(sleep_time)

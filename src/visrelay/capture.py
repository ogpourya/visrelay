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


NONE_STALL_S = 10.0
FIRST_FRAME_S = 15.0


def encode_frame(frame: np.ndarray, quality: int) -> bytes:
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3/Wx4 array, got shape {frame.shape}")
    channels = frame.shape[2]
    colorspace = "RGB" if channels == 3 else "BGRA"
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return bytes(
        simplejpeg.encode_jpeg(
            frame,
            quality=quality,
            colorspace=colorspace,
            colorsubsampling="444" if quality >= 90 else "420",
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
        self._last_frame: np.ndarray | None = None
        self._last_frame_buf: np.ndarray | None = None

    def stop(self) -> None:
        self._stop.set()
        try:
            cond = getattr(self._store, "demand_condition", None) or self._store.condition
            with cond:
                cond.notify_all()
        except Exception:
            pass

    def _encode_if_changed(self, frame: np.ndarray) -> tuple[bytes, float]:
        if frame is self._last_frame and self._last_jpeg is not None:
            return self._last_jpeg, 0.0
        shape = frame.shape
        if (
            self._last_jpeg is not None
            and self._last_frame is not None
            and shape == self._last_shape
        ):
            try:
                if np.array_equal(frame, self._last_frame):
                    return self._last_jpeg, 0.0
            except Exception:
                pass
        t0 = time.perf_counter()
        jpeg = encode_frame(frame, self._quality)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        self._last_jpeg = jpeg
        self._last_shape = shape
        try:
            if self._last_frame_buf is None or self._last_frame_buf.shape != shape:
                self._last_frame_buf = np.empty_like(frame)
            np.copyto(self._last_frame_buf, frame)
            self._last_frame = self._last_frame_buf
        except Exception:
            try:
                self._last_frame = frame.copy()
            except Exception:
                self._last_frame = None
        return jpeg, encode_ms

    def run(self) -> None:
        stream = self._stream
        abrupt = False
        stall = False
        try:
            frame = None
            no_frame_since: float | None = None
            while frame is None and not self._stop.is_set():
                if not self._store.wants_capture(self._idle_timeout):
                    no_frame_since = None
                    if getattr(stream, "window_invalid", False):
                        abrupt = True
                        break
                    self._store.wait_for_demand(self._idle_timeout, timeout=0.2)
                    continue
                if getattr(stream, "window_invalid", False) or getattr(stream, "error", None):
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

            if abrupt or stall or frame is None:
                if abrupt:
                    self.abrupt = True
                if stall:
                    self.stalled = True
                return

            last_frame = frame
            store_last_publish_ts = 0.0
            try:
                h, w = last_frame.shape[0], last_frame.shape[1]
            except Exception:
                h, w = self._height, self._width
            try:
                jpeg, encode_ms = self._encode_if_changed(last_frame)
            except Exception:
                self.errors += 1
                jpeg, encode_ms = None, 0.0  # type: ignore[assignment]
            if jpeg is not None:
                self._store.publish(jpeg, width=w or self._width, height=h or self._height, encode_ms=encode_ms, hardware=True)
                self.frames += 1
                store_last_publish_ts = time.monotonic()

            none_since: float | None = None
            while not self._stop.is_set():
                if not self._store.wants_capture(self._idle_timeout):
                    none_since = None
                    if getattr(stream, "window_invalid", False) or getattr(stream, "error", None):
                        abrupt = True
                        break
                    self._store.wait_for_demand(self._idle_timeout, timeout=0.2)
                    continue
                if getattr(stream, "window_invalid", False) or getattr(stream, "error", None):
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
                    # Static screen / no new damage frame from compositor.
                    if getattr(stream, "window_invalid", False) or getattr(stream, "error", None):
                        stall = True
                        break
                    if last_frame is None:
                        if none_since is None:
                            none_since = time.monotonic()
                        elif time.monotonic() - none_since > NONE_STALL_S:
                            stall = True
                            break
                        time.sleep(0.005)
                        continue

                    # No call to _encode_if_changed when latest is None!
                    now = time.monotonic()
                    if (now - store_last_publish_ts) >= 1.0 and self._last_jpeg is not None:
                        self._store.publish(self._last_jpeg, width=w or self._width, height=h or self._height, encode_ms=0.0, hardware=False)
                        store_last_publish_ts = now

                    elapsed = time.perf_counter() - loop_start
                    sleep_time = self._interval - elapsed
                    if sleep_time > 0:
                        self._stop.wait(sleep_time)
                    continue

                none_since = None
                last_frame = latest
                try:
                    h, w = last_frame.shape[0], last_frame.shape[1]
                except Exception:
                    h, w = self._height, self._width
                try:
                    jpeg, encode_ms = self._encode_if_changed(last_frame)
                except Exception:
                    self.errors += 1
                    time.sleep(0.01)
                    continue

                if encode_ms == 0.0:
                    now = time.monotonic()
                    if (now - store_last_publish_ts) < 1.0:
                        elapsed = time.perf_counter() - loop_start
                        sleep_time = self._interval - elapsed
                        if sleep_time > 0:
                            self._stop.wait(sleep_time)
                        continue

                self._store.publish(jpeg, width=w or self._width, height=h or self._height, encode_ms=encode_ms, hardware=True)
                self.frames += 1
                store_last_publish_ts = time.monotonic()
                elapsed = time.perf_counter() - loop_start
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    self._stop.wait(sleep_time)
        finally:
            if not self._stop.is_set():
                if abrupt:
                    self.abrupt = True
                    self._store.close("sharing stopped")
                elif stall:
                    self.stalled = True
                    self._store.close("capture stalled")
                else:
                    self._store.close("sharing stopped")
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
        self.ended = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        try:
            cond = getattr(self._store, "demand_condition", None) or self._store.condition
            with cond:
                cond.notify_all()
        except Exception:
            pass

    def run(self) -> None:
        n = 0
        w, h = self._width, self._height
        frame = np.empty((h, w, 3), dtype=np.uint8)
        xs = np.arange(w, dtype=np.uint8)[None, :]
        ys = (np.arange(h, dtype=np.uint16)[:, None] * 255 // max(h - 1, 1)).astype(np.uint8)
        try:
            while not self._stop.is_set():
                if not self._store.wants_capture(self._idle_timeout):
                    self._store.wait_for_demand(self._idle_timeout, timeout=0.2)
                    continue
                loop_start = time.perf_counter()
                np.add(xs, np.uint8((n * 8) & 0xFF), out=frame[:, :, 0])
                np.add(ys, np.uint8((n * 2) & 0xFF), out=frame[:, :, 1])
                frame[:, :, 2].fill((n * 5) % 256)
                try:
                    t0 = time.perf_counter()
                    jpeg = encode_frame(frame, self._quality)
                    encode_ms = (time.perf_counter() - t0) * 1000.0
                except Exception:
                    time.sleep(0.05)
                    continue
                self._store.publish(jpeg, width=w, height=h, encode_ms=encode_ms, hardware=True)
                self.frames += 1
                n += 1
                elapsed = time.perf_counter() - loop_start
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    self._stop.wait(sleep_time)
        finally:
            self._store.close("sharing stopped")
            self.ended.set()

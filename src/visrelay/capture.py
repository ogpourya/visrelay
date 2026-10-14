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
        self._idle_timeout = idle_timeout

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        stream = self._stream
        abrupt = False
        try:
            # Wait briefly for the first buffer (mirrors main.py behaviour).
            frame = None
            while frame is None and not self._stop.is_set():
                if not self._store.wants_capture(self._idle_timeout):
                    if getattr(stream, "window_invalid", False):
                        abrupt = True
                        break
                    self._stop.wait(0.2)
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
                    time.sleep(0.005)

            if abrupt:
                self._store.close("sharing stopped")
                return

            last_frame = frame
            while not self._stop.is_set():
                if not self._store.wants_capture(self._idle_timeout):
                    if getattr(stream, "window_invalid", False) or getattr(stream, "error", None):
                        abrupt = True
                        break
                    self._stop.wait(0.2)
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
                if latest is not None:
                    last_frame = latest
                if last_frame is None:
                    time.sleep(0.005)
                    continue
                try:
                    t0 = time.perf_counter()
                    jpeg = encode_frame(last_frame, self._quality)
                    encode_ms = (time.perf_counter() - t0) * 1000.0
                except Exception:
                    self.errors += 1
                    time.sleep(0.01)
                    continue
                h, w = last_frame.shape[0], last_frame.shape[1]
                self._store.publish(jpeg, width=w or self._width, height=h or self._height, encode_ms=encode_ms)
                self.frames += 1
                elapsed = time.perf_counter() - loop_start
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    # Wake early on stop().
                    self._stop.wait(sleep_time)
        finally:
            if abrupt and not self._stop.is_set():
                self.abrupt = True
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

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        n = 0
        w, h = self._width, self._height
        # Precompute gradients once; per frame is just modular adds (no arange).
        xs = np.arange(w, dtype=np.uint16)
        ys = (np.arange(h, dtype=np.uint16)[:, None] * 255 // max(h - 1, 1))
        while not self._stop.is_set():
            if not self._store.wants_capture(self._idle_timeout):
                self._stop.wait(0.2)
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

"""Background capture loop: PipeWire -> JPEG -> FrameStore.

Paced to a target FPS (default 15) to leave headroom for HTTP serving.
JPEG encoding uses simplejpeg (libjpeg-turbo SIMD), same as main.py.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .frames import FrameStore


def _pick_encoder():
    """Prefer simplejpeg (libjpeg-turbo SIMD); fall back to Pillow on 3.15+.

    simplejpeg has no 3.15 wheels yet (cmake build fails on 3.15rc), while
    Pillow ships 3.15 wheels. Both are libjpeg-turbo under the hood; the
    hot loop below works with either without re-encoding or copies.
    """
    try:
        import simplejpeg as _sj  # noqa: F401

        return "simplejpeg"
    except Exception:
        return "pillow"


_ENCODER = _pick_encoder()

if _ENCODER == "simplejpeg":
    def encode_frame(frame: np.ndarray, quality: int) -> bytes:
        import simplejpeg

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
else:
    import io as _io

    def encode_frame(frame: np.ndarray, quality: int) -> bytes:
        from PIL import Image

        if frame.ndim == 3 and frame.shape[2] == 4:
            # BGRA -> RGB: drop alpha + swap R/B in one fused pass.
            h, w = frame.shape[0], frame.shape[1]
            rgb = np.empty((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = frame[:, :, 2]
            rgb[:, :, 1] = frame[:, :, 1]
            rgb[:, :, 2] = frame[:, :, 0]
            im = Image.fromarray(rgb)
        else:
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            im = Image.fromarray(frame)
        buf = _io.BytesIO()
        # optimize=False + 4:2:0 subsampling = fastest encode, smallest stream.
        im.save(buf, format="JPEG", quality=quality, subsampling=2, optimize=False, progressive=False)
        return buf.getvalue()


class CaptureWorker(threading.Thread):
    """Pulls frames from CaptureStream, encodes, publishes. Daemon thread."""

    def __init__(
        self,
        stream,
        store: FrameStore,
        *,
        fps: float = 15.0,
        quality: int = 80,
        width: int = 0,
        height: int = 0,
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

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        stream = self._stream
        # Wait briefly for the first buffer (mirrors main.py behaviour).
        frame = None
        while frame is None and not self._stop.is_set():
            if getattr(stream, "window_invalid", False):
                return
            try:
                frame = stream.get_frame()
            except Exception:
                self.errors += 1
                break
            if frame is None:
                time.sleep(0.005)

        last_frame = frame
        while not self._stop.is_set():
            if getattr(stream, "window_invalid", False):
                break
            if getattr(stream, "error", None):
                break
            loop_start = time.perf_counter()
            try:
                latest = stream.get_frame()
            except Exception:
                self.errors += 1
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


class DemoWorker(threading.Thread):
    """Synthetic test pattern generator (no Wayland needed). --demo mode."""

    def __init__(self, store: FrameStore, *, fps: float = 15.0, quality: int = 80,
                 width: int = 640, height: int = 360) -> None:
        super().__init__(name="visrelay-demo", daemon=True)
        self._store = store
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

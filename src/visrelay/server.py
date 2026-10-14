"""Zero-copy, low-overhead HTTP server for visrelay.

Endpoints:
  GET /               -> live HTML page (<img src="/stream.mjpg">)
  GET /stream.mjpg    -> multipart/x-mixed-replace MJPEG (also /stream, /mjpeg)
  GET /snapshot.jpg   -> single latest JPEG (also /shot.jpg, /frame.jpg, /snapshot)
  GET /healthz        -> JSON status (also /health)
  GET /favicon.ico    -> 204 (avoid log spam)

Performance choices (optimize to death):
  - stdlib ThreadingHTTPServer only: no framework overhead, no per-request logging.
  - Single encoder thread (capture.py); HTTP threads do I/O only, never encode.
  - FrameStore.publish hands out immutable `bytes`; handlers write them directly,
    no copies, no re-encode, no resize.
  - MJPEG part headers are prebuilt prefix/suffix; per-frame work is one
    `b"--FRAME\\r\\nContent-Type: image/jpeg\\r\\nContent-Length: N\\r\\n\\r\\n"`
    + body + CRLF. No dict lookups in the hot loop.
  - Index/health payloads are pre-serialized once; snapshots send prebuilt
    header prefix + length.
  - TCP_NODELAY set per connection; daemon threads; allow_reuse_address.
  - `log_message` overridden to no-op unless verbose (stderr logging per
    request is a notorious MJPEG killer).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .frames import FrameStore

BOUNDARY = "FRAME"
BOUNDARY_B = b"--FRAME"
CRLF = b"\r\n"

SNAPSHOT_PATHS = {"/snapshot.jpg", "/snapshot", "/shot.jpg", "/frame.jpg", "/frame", "/jpg"}
STREAM_PATHS = {"/stream.mjpg", "/stream", "/mjpeg", "/video.mjpg", "/live.mjpg"}
HEALTH_PATHS = {"/healthz", "/health", "/status"}


def _strip_query(path: str) -> str:
    q = path.find("?")
    if q != -1:
        path = path[:q]
    h = path.find("#")
    if h != -1:
        path = path[:h]
    return path.rstrip("/") or "/"


class _Handler(BaseHTTPRequestHandler):
    server_version = ""
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # Class-level shared state, wired by `serve()`.
    store: FrameStore = None  # type: ignore[assignment]
    index_payload: bytes = b""
    verbose: bool = False
    start_time: float = 0.0
    target_fps: float = 15.0

    def setup(self) -> None:
        super().setup()
        try:
            import socket as _socket

            self.request.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN001, ANN002
        if self.verbose:
            super().log_message(fmt, *args)

    def _send(self, code: int, ctype: str, body: bytes, extra: tuple[tuple[str, str], ...] = ()) -> None:
        # Fast direct header write for tiny responses (health/index/errors).
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in extra:
            self.send_header(k, v)
        # Keep-alive for snapshots/health so browsers reuse the connection.
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        if self.command != "HEAD" and body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self) -> None:  # noqa: N802
        path = _strip_query(self.path)
        if path == "/":
            self._send(200, "text/html; charset=utf-8", self.index_payload)
        elif path in STREAM_PATHS:
            self._handle_mjpeg()
        elif path in SNAPSHOT_PATHS:
            self._handle_snapshot()
        elif path in HEALTH_PATHS:
            self._handle_health()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    do_HEAD = do_GET  # type: ignore[assignment]

    # -- hot paths ------------------------------------------------------
    def _handle_snapshot(self) -> None:
        store = self.server.state_store  # type: ignore[attr-defined]
        jpeg, _, _ = store.snapshot()
        # Wait briefly (not busy-loop) if encoder hasn't produced a frame yet.
        if jpeg is None:
            jpeg, _, _ = store.wait_for(-1, timeout=5.0)
        if jpeg is None:
            self._send(503, "text/plain; charset=utf-8", b"no frame yet, retry\n")
            return
        self._send(200, "image/jpeg", jpeg)

    def _handle_health(self) -> None:
        store = self.server.state_store  # type: ignore[attr-defined]
        _, frame_id, ts = store.snapshot()
        now = time.monotonic()
        body = json.dumps(
            {
                "ok": True,
                "frames": frame_id,
                "width": store.width,
                "height": store.height,
                "target_fps": self.server.state_fps,  # type: ignore[attr-defined]
                "encode_ms": round(store.encode_ms, 2),
                "age_s": round(now - ts, 3) if ts else -1,
                "uptime_s": round(now - self.server.state_start, 3),  # type: ignore[attr-defined]
            }
        ).encode("utf-8")
        self._send(200, "application/json", body)

    def _handle_mjpeg(self) -> None:
        store: FrameStore = self.server.state_store  # type: ignore[attr-defined]
        # One-shot header; afterwards raw socket writes only (no send_header).
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            return

        wfile = self.wfile
        write = wfile.write
        flush = wfile.flush
        wait_for = store.wait_for
        boundary = BOUNDARY_B
        last_id = -1
        # Local bindings: avoids attribute lookups in the 15Hz hot loop.
        try:
            while True:
                jpeg, fid, _ = wait_for(last_id, timeout=10.0)
                if jpeg is None:
                    # Encoder stalled/ended; keep connection open a bit, then exit
                    # so the client reconnects instead of spinning.
                    try:
                        flush()
                    except Exception:
                        break
                    if getattr(store, "_frame_id", 0) == 0:
                        continue
                    break
                last_id = fid
                # Prebuilt part header; single length format is the only alloc.
                header = (
                    boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                )
                try:
                    write(header)
                    write(jpeg)
                    write(CRLF)
                    flush()
                except (BrokenPipeError, ConnectionResetError, ValueError):
                    break
        finally:
            try:
                self.close_connection = True
            except Exception:
                pass


def build_index(*, port: int, fps: float, width: int = 0, height: int = 0) -> bytes:
    res = f"{width}x{height}" if width and height else "live"
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>visrelay :{port}</title>"
        "<style>html,body{margin:0;background:#0b0b0d;color:#e8e8ea;font:14px/1.4 system-ui,sans-serif}"
        "header{padding:10px 14px;background:#141419}code{background:#22222a;padding:1px 6px;border-radius:6px}"
        "main{padding:10px 14px}img{width:100%;max-height:86vh;object-fit:contain;background:#000;border-radius:8px}"
        ".row{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;color:#b9b9c2}</style></head>"
        f"<body><header><b>visrelay</b> &nbsp;<code>:{port}</code> &nbsp;<code>{fps:g} fps</code> &nbsp;<code>{res}</code></header>"
        "<main><img src='/stream.mjpg' alt='live screen'>"
        "<div class=row><span>MJPEG: <code>/stream.mjpg</code></span>"
        "<span>Snapshot: <code>/snapshot.jpg</code></span>"
        "<span>Health: <code>/healthz</code></span></div></main></body></html>"
    ).encode("utf-8")


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, addr, handler, *, store: FrameStore, fps: float, verbose: bool = False):  # noqa: ANN001, ANN002
        self.state_store = store
        self.state_fps = fps
        self.state_start = time.monotonic()
        handler.store = store
        handler.verbose = verbose
        handler.start_time = self.state_start
        handler.target_fps = fps
        super().__init__(addr, handler)


def serve_forever(
    store: FrameStore,
    *,
    host: str,
    port: int,
    fps: float,
    verbose: bool = False,
    ready: threading.Event | None = None,
) -> None:
    index = build_index(port=port, fps=fps, width=store.width, height=store.height)
    _Handler.index_payload = index
    srv = RelayServer((host, port), _Handler, store=store, fps=fps, verbose=verbose)
    # Refresh index once dimensions are known (portal may resolve after bind).
    def _refresh_index() -> None:
        time.sleep(0.5)
        saw = (store.width, store.height)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (store.width, store.height) != saw and store.width:
                _Handler.index_payload = build_index(port=port, fps=fps, width=store.width, height=store.height)
                break
            time.sleep(0.5)
    threading.Thread(target=_refresh_index, name="visrelay-index-refresh", daemon=True).start()
    if ready is not None:
        ready.set()
    try:
        srv.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()

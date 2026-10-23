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
  - MJPEG parts are prebuilt once per frame in FrameStore.publish; per-frame
    work is a single write + flush. No dict lookups in the hot loop.
  - Index/health payloads are pre-serialized once; snapshots send prebuilt
    header prefix + length.
  - TCP_NODELAY set per connection; daemon threads; allow_reuse_address.
  - `log_message` overridden to no-op unless verbose (stderr logging per
    request is a notorious MJPEG killer).
"""

from __future__ import annotations

import json
import socket
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

    def setup(self) -> None:
        super().setup()
        try:
            import socket as _socket

            self.request.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            self.request.settimeout(20.0)
        except Exception:
            pass

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except TimeoutError:
            self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN001, ANN002
        try:
            verbose = self.server.verbose  # type: ignore[attr-defined]
        except Exception:
            verbose = False
        if verbose:
            super().log_message(fmt, *args)

    def _send(self, code: int, ctype: str, body: bytes, extra: tuple[tuple[str, str], ...] = ()) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in extra:
            self.send_header(k, v)
        if self.close_connection:
            self.send_header("Connection", "close")
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        if self.command != "HEAD" and body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    def do_GET(self) -> None:  # noqa: N802
        path = _strip_query(self.path)
        if path == "/":
            try:
                payload = self.server.index_payload  # type: ignore[attr-defined]
            except Exception:
                payload = b""
            self._send(200, "text/html; charset=utf-8", payload)
        elif path in STREAM_PATHS:
            self._handle_mjpeg()
        elif path in SNAPSHOT_PATHS:
            self._handle_snapshot()
        elif path in HEALTH_PATHS:
            self._handle_health()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            if self.close_connection:
                self.send_header("Connection", "close")
            else:
                self.send_header("Connection", "keep-alive")
            self.end_headers()
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    do_HEAD = do_GET  # type: ignore[assignment]

    def _handle_snapshot(self) -> None:
        store = self.server.state_store  # type: ignore[attr-defined]
        try:
            fps = self.server.state_fps  # type: ignore[attr-defined]
        except Exception:
            fps = 24.0
        try:
            max_age = max(1.0, 2.0 / max(float(fps), 1.0))
        except Exception:
            max_age = 1.0
        jpeg = store.snapshot_fresh(max_age=max_age, timeout=5.0)
        if jpeg is None:
            self._send(503, "text/plain; charset=utf-8", b"no frame yet, retry\n")
            return
        self._send(200, "image/jpeg", jpeg)

    def _handle_health(self) -> None:
        store = self.server.state_store  # type: ignore[attr-defined]
        frame_id, ts, viewers, width, height, encode_ms, _closed = store.status()
        now = time.monotonic()
        body = json.dumps(
            {
                "ok": not _closed,
                "frames": frame_id,
                "viewers": viewers,
                "width": width,
                "height": height,
                "target_fps": self.server.state_fps,  # type: ignore[attr-defined]
                "encode_ms": round(encode_ms, 2),
                "age_s": round(now - ts, 3) if ts else -1,
                "uptime_s": round(now - self.server.state_start, 3),  # type: ignore[attr-defined]
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._send(503 if _closed else 200, "application/json", body)

    def _handle_mjpeg(self) -> None:
        if self.command == "HEAD":
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        store: FrameStore = self.server.state_store  # type: ignore[attr-defined]
        store.viewer_open()
        try:
            self._serve_mjpeg(store)
        finally:
            store.viewer_close()

    def _serve_mjpeg(self, store: FrameStore) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

        wfile = self.wfile
        writelines = wfile.writelines
        flush = wfile.flush
        wait_for_part = store.wait_for_part if hasattr(store, "wait_for_part") else None
        last_id = -1
        try:
            while True:
                if store.is_closed():
                    break
                if wait_for_part is not None:
                    header, jpeg, fid, _ = wait_for_part(last_id, timeout=10.0)
                else:
                    jpeg, fid, _ = store.wait_for(last_id, timeout=10.0)
                    header = None
                    if fid != last_id and jpeg is not None:
                        header = (
                            BOUNDARY_B + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                        )
                if store.is_closed():
                    break
                if header is None or jpeg is None or fid == last_id:
                    try:
                        flush()
                    except Exception:
                        break
                    continue
                last_id = fid
                try:
                    writelines([header, jpeg, CRLF])
                    flush()
                except (BrokenPipeError, ConnectionResetError, ValueError, TimeoutError):
                    break
        finally:
            try:
                self.close_connection = True
            except Exception:
                pass


def build_index(*, port: int, fps: float, width: int = 0, height: int = 0) -> bytes:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>visrelay :{port}</title>"
        "<style>"
        "html,body{margin:0;padding:0;height:100%;width:100%;overflow:hidden;background:#000}"
        "img{position:fixed;inset:0;width:100%;height:100%;object-fit:contain;background:#000}"
        "</style></head><body>"
        "<img src='/stream.mjpg' alt='live screen'>"
        "<script>const i=document.querySelector('img');"
        "i.onerror=()=>setTimeout(()=>i.src='/stream.mjpg?r='+Date.now(),1000);"
        "document.addEventListener('visibilitychange',()=>{if(!document.hidden)i.src='/stream.mjpg?r='+Date.now()});"
        "</script></body></html>"
    ).encode("utf-8")


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, addr, handler, *, store: FrameStore, fps: float, verbose: bool = False):  # noqa: ANN001, ANN002
        self.state_store = store
        self.state_fps = fps
        self.state_start = time.monotonic()
        self.verbose = verbose
        self.index_payload = b""
        if ":" in str(addr[0]):
            self.address_family = socket.AF_INET6
        super().__init__(addr, handler)


def serve_forever(
    store: FrameStore,
    *,
    host: str,
    port: int,
    fps: float,
    verbose: bool = False,
    ready: threading.Event | None = None,
    stop_when=None,
    on_ready=None,
) -> str:
    """Run the HTTP server until Ctrl+C or stop_when() is true.

    Returns "portal-closed" when the capture source went away (user
    stopped sharing from the desktop panel), else "ok".
    """
    index = build_index(port=port, fps=fps, width=store.width, height=store.height)
    srv = RelayServer((host, port), _Handler, store=store, fps=fps, verbose=verbose)
    srv.index_payload = index
    stopped_by_watch = threading.Event()
    if stop_when is not None:
        def _watch() -> None:
            while True:
                time.sleep(0.2)
                try:
                    if stop_when():
                        stopped_by_watch.set()
                        srv.shutdown()
                        break
                except Exception:
                    break
        threading.Thread(target=_watch, name="visrelay-stop-watch", daemon=True).start()
    if ready is not None:
        ready.set()
    if on_ready is not None:
        try:
            on_ready()
        except Exception:
            pass
    try:
        srv.serve_forever(poll_interval=0.2)
    finally:
        srv.server_close()
    return "portal-closed" if stopped_by_watch.is_set() else "ok"

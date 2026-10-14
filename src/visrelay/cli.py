"""CLI: parse args, acquire screen, start encoder + HTTP server.

Defaults (per spec): bind 0.0.0.0:29173 @ 15 fps. Lower-than-60fps default
keeps CPU headroom for JPEG encode + HTTP serving.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from . import _compat as _compat_shim  # noqa: F401  (typing shim for dbus-next on 3.15+)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 29173
DEFAULT_FPS = 15.0
DEFAULT_QUALITY = 80


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visrelay",
        description="Stream the selected screen/window over HTTP (MJPEG + snapshot).",
    )
    p.add_argument("--host", default=os.environ.get("VISRELAY_HOST", DEFAULT_HOST),
                   help=f"Bind address (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=int(os.environ.get("VISRELAY_PORT", DEFAULT_PORT)),
                   help=f"Bind port (default {DEFAULT_PORT})")
    p.add_argument("--fps", type=float, default=float(os.environ.get("VISRELAY_FPS", DEFAULT_FPS)),
                   help=f"Target capture fps (default {DEFAULT_FPS:g}; paced to save CPU)")
    p.add_argument("--quality", type=int, default=int(os.environ.get("VISRELAY_QUALITY", DEFAULT_QUALITY)),
                   help=f"JPEG quality 1-100 (default {DEFAULT_QUALITY}; lower = faster/smaller)")
    p.add_argument("--demo", action="store_true",
                   help="Serve a synthetic test pattern (no Wayland/portal needed)")
    p.add_argument("--demo-width", type=int, default=640)
    p.add_argument("--demo-height", type=int, default=360)
    p.add_argument("-v", "--verbose", action="store_true", help="Log HTTP requests")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    from .frames import FrameStore

    args = build_parser().parse_args(argv)

    if args.version:
        try:
            from importlib.metadata import version as _v

            print(_v("visrelay"))
        except Exception:
            print("0.1.0")
        return 0

    if not (1 <= args.quality <= 100):
        print("error: --quality must be 1-100", file=sys.stderr)
        return 2
    if not (0 < args.fps <= 60):
        print("error: --fps must be 0 < fps <= 60", file=sys.stderr)
        return 2

    store = FrameStore()
    worker = None
    cleanup = None

    if args.demo:
        from .capture import DemoWorker

        print(f"[visrelay] demo pattern {args.demo_width}x{args.demo_height} "
              f"-> http://{args.host}:{args.port}/stream.mjpg @ {args.fps:g} fps")
        worker = DemoWorker(store, fps=args.fps, quality=args.quality,
                            width=args.demo_width, height=args.demo_height)
        worker.start()
    else:
        # Heavy Wayland-only imports stay lazy so --help/--demo stay instant.
        try:
            from .portal import request_wayland_screen
            from pipewire_capture import CaptureStream
        except ImportError as e:
            print(f"[visrelay] missing dependency: {e}\n"
                  "  install with: uv sync  (needs pipewire-capture, simplejpeg, dbus-next, numpy)",
                  file=sys.stderr)
            print("  tip: `visrelay --demo` runs without Wayland.", file=sys.stderr)
            return 1

        print("[visrelay] opening Wayland screen picker (Monitor + Window)...", flush=True)
        loop = asyncio.new_event_loop()
        try:
            session = loop.run_until_complete(request_wayland_screen())
        except Exception as e:
            print(f"[visrelay] portal request failed: {e}", file=sys.stderr)
            loop.close()
            return 1
        if session is None:
            print("[visrelay] selection cancelled.", file=sys.stderr)
            loop.close()
            return 1
        pw_fd, node_id, width, height, bus = session
        print(f"[visrelay] capturing {width}x{height} -> "
              f"http://{args.host}:{args.port}/stream.mjpg @ {args.fps:g} fps", flush=True)

        from .capture import CaptureWorker

        stream = CaptureStream(pw_fd, node_id, width, height, capture_interval=0.0)
        try:
            stream.start()
        except Exception as e:
            print(f"[visrelay] failed to start PipeWire stream: {e}", file=sys.stderr)
            try:
                os.close(pw_fd)
            except OSError:
                pass
            bus.disconnect()
            loop.close()
            return 1

        worker = CaptureWorker(stream, store, fps=args.fps, quality=args.quality,
                               width=width, height=height)
        worker.start()

        def _cleanup() -> None:
            try:
                worker.stop()
            except Exception:
                pass
            try:
                stream.stop()
            except Exception:
                pass
            try:
                os.close(pw_fd)
            except OSError:
                pass
            try:
                bus.disconnect()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

        cleanup = _cleanup

    from .server import serve_forever

    print(f"[visrelay] serving  http://{args.host}:{args.port}/  "
          f"(mjpeg /stream.mjpg, snapshot /snapshot.jpg, health /healthz)", flush=True)
    try:
        serve_forever(store, host=args.host, port=args.port, fps=args.fps, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n[visrelay] stopping (Ctrl+C)...")
    except OSError as e:
        print(f"[visrelay] bind {args.host}:{args.port} failed: {e}", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
            # Give the encoder a beat to exit its sleep; don't block shutdown.
            if hasattr(worker, "join"):
                try:
                    worker.join(timeout=2.0)
                except Exception:
                    pass
        if cleanup is not None:
            cleanup()
        # Brief summary (mirrors main.py prototype stats).
        try:
            n = getattr(worker, "frames", 0)
            if n:
                print(f"[visrelay] served {n} frames.")
        except Exception:
            pass
        # Allow server threads to drain.
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

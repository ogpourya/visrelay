"""CLI: parse args, acquire screen, start encoder + HTTP server.

Defaults (per spec): bind 0.0.0.0:29173 @ 24 fps. Lower-than-60fps default
keeps CPU headroom for JPEG encode + HTTP serving.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import threading
import time

from . import _compat as _compat_shim  # noqa: F401  (typing shim for dbus-next on 3.15+)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 29173
DEFAULT_FPS = 24.0
DEFAULT_QUALITY = 80


def _lan_ip() -> str | None:
    """First IPv4 of a physical LAN interface (skips VPN/tunnel/bridge).

    The old UDP-to-8.8.8.8 trick follows the default route, which points
    into VPN tunnels (e.g. happ-xray 172.19.0.1) when one is active.
    Instead enumerate interfaces and require UP + BROADCAST while
    excluding POINTOPOINT / NOARP / LOOPBACK and virtual names, so we
    land on e.g. enp5s0 192.168.70.168.
    """
    try:
        import array
        import fcntl
        import struct
    except ImportError:
        return None
    try:
        names = [n for _, n in socket.if_nameindex()]
    except OSError:
        return None
    SIOCGIFFLAGS = 0x8913
    SIOCGIFADDR = 0x8915
    IFF_UP, IFF_BROADCAST, IFF_LOOPBACK, IFF_POINTOPOINT, IFF_NOARP = 0x1, 0x2, 0x8, 0x10, 0x80
    VIRTUAL = ("lo", "docker", "veth", "br-", "lxcbr", "virbr", "vboxnet",
               "vmnet", "tun", "tap", "wg", "tailscale", "happ", "xray", "ppp", "dummy")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        cands: list[tuple[int, str]] = []
        for name in names:
            ln = name.lower()
            if ln.startswith(VIRTUAL) or "happ" in ln or "xray" in ln:
                continue
            try:
                ifreq = struct.pack("16sH14s", name.encode()[:15], 0, b"\x00" * 14)
                flags = struct.unpack("16sH14s", fcntl.ioctl(s.fileno(), SIOCGIFFLAGS, ifreq))[1]
            except OSError:
                continue
            if not (flags & IFF_UP):
                continue
            if flags & (IFF_LOOPBACK | IFF_POINTOPOINT | IFF_NOARP):
                continue
            if not (flags & IFF_BROADCAST):
                continue
            try:
                res = fcntl.ioctl(s.fileno(), SIOCGIFADDR, struct.pack("256s", name.encode()[:15]))
                ip = socket.inet_ntoa(res[20:24])
            except OSError:
                continue
            if ip.startswith("127.") or ip.startswith("169.254."):
                continue
            # Prefer real NIC names (en/eth/wl/wlan) over the rest.
            prio = 0 if ln.startswith(("en", "eth", "wl", "wlan", "wlp")) else 1
            cands.append((prio, ip))
        cands.sort()
        return cands[0][1] if cands else None
    finally:
        s.close()


def _display_host(bind_host: str) -> str:
    """Host to show in copy-pasteable URLs (0.0.0.0 is not connectable)."""
    if bind_host not in ("0.0.0.0", "::", "", "0:0:0:0:0:0:0:0"):
        return bind_host
    try:
        ip = _lan_ip()
        if ip:
            return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No traffic sent; only selects the default-route source IP.
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return "127.0.0.1"


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
    p.add_argument("--idle-timeout", type=float, default=float(os.environ.get("VISRELAY_IDLE_TIMEOUT", 5.0)),
                   help="Keep capturing N seconds after the last viewer leaves (default 5; 0 = pause immediately)")
    p.add_argument("--quality", type=int, default=int(os.environ.get("VISRELAY_QUALITY", DEFAULT_QUALITY)),
                   help=f"JPEG quality 1-100 (default {DEFAULT_QUALITY}; lower = faster/smaller)")
    p.add_argument("--demo", action="store_true",
                   help="Serve a synthetic test pattern (no Wayland/portal needed)")
    p.add_argument("--demo-width", type=int, default=640)
    p.add_argument("--demo-height", type=int, default=360)
    p.add_argument("-v", "--verbose", action="store_true", help="Log HTTP requests")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    p.add_argument("--no-color", action="store_true", help="Disable colored output")
    return p


def main(argv: list[str] | None = None) -> int:
    from . import ui as _ui
    from .frames import FrameStore

    args = build_parser().parse_args(argv)
    if args.no_color:
        _ui.disable()

    if args.version:
        print(_ui.get_version())
        return 0

    if not (1 <= args.quality <= 100):
        _ui.error("visrelay: error: --quality must be 1-100")
        return 2
    if not (0 < args.fps <= 60):
        _ui.error("visrelay: error: --fps must be 0 < fps <= 60")
        return 2

    store = FrameStore()
    worker = None
    cleanup = None
    width = height = 0
    is_demo = bool(args.demo)

    _ui.header()

    if is_demo:
        from .capture import DemoWorker

        _ui.success(f"Demo {args.demo_width}x{args.demo_height} @ {args.fps:g} fps")
        worker = DemoWorker(store, fps=args.fps, quality=args.quality,
                            width=args.demo_width, height=args.demo_height,
                            idle_timeout=args.idle_timeout)
        worker.start()
    else:
        # Heavy Wayland-only imports stay lazy so --help/--demo stay instant.
        try:
            from .portal import request_wayland_screen
            from pipewire_capture import CaptureStream
        except ImportError as e:
            _ui.error(f"missing dependency: {e}\n  install with: uv sync\n"
                      "  try `visrelay --demo` without Wayland.")
            return 1

        spin = _ui.Spinner("Select a screen or window in the system dialog…")
        spin.start()
        loop = asyncio.new_event_loop()
        try:
            session = loop.run_until_complete(request_wayland_screen())
        except Exception as e:
            spin.stop(clear=True)
            _ui.error(f"screen picker failed: {e}")
            loop.close()
            return 1
        spin.stop(clear=True)
        if session is None:
            _ui.cancelled()
            loop.close()
            return 1
        pw_fd, node_id, width, height, bus = session
        _ui.success(f"Capturing {width}x{height} @ {args.fps:g} fps")

        from .capture import CaptureWorker

        stream = CaptureStream(pw_fd, node_id, width, height, capture_interval=0.0)
        try:
            stream.start()
        except Exception as e:
            _ui.error(f"could not start capture: {e}")
            try:
                os.close(pw_fd)
            except OSError:
                pass
            bus.disconnect()
            loop.close()
            return 1

        worker = CaptureWorker(stream, store, fps=args.fps, quality=args.quality,
                               width=width, height=height, idle_timeout=args.idle_timeout)
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

    _ui.ready(_display_host(args.host), args.port)

    def _stall_watch() -> None:
        while not store.is_closed():
            time.sleep(1.0)
            try:
                if store.stalled(12.0, args.idle_timeout):
                    store.close("capture stalled")
                    break
            except Exception:
                break

    threading.Thread(target=_stall_watch, name="visrelay-stall-watch", daemon=True).start()
    stop_when = None if is_demo else store.is_closed
    try:
        serve_forever(store, host=args.host, port=args.port, fps=args.fps,
                      verbose=args.verbose, stop_when=stop_when)
    except KeyboardInterrupt:
        _ui.stopped()
    except OSError as e:
        _ui.error(f"cannot listen on {args.host}:{args.port}: {e}")
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
        # Allow server threads to drain.
        time.sleep(0.1)
    if store.is_closed():
        if store.close_reason == "capture stalled":
            _ui.stalled()
        else:
            _ui.sharing_stopped()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

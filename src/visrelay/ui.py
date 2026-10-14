"""Bubble Tea-style inline status UI (not a fullscreen TUI).

Stdlib only: ANSI colors, spinner, aligned URL rows. Colors auto-disable
when piped, dumb terminal, or NO_COLOR / --no-color.
"""

from __future__ import annotations

import os
import sys
import threading
import time

_ENABLED = True


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def supports_color(stream=None) -> bool:
    if not _ENABLED:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    s = stream if stream is not None else sys.stdout
    try:
        return s.isatty()
    except Exception:
        return False


_COLOR = supports_color  # late-bound; call per print so pipes behave


def _st(text: str, *codes: str, stream=None) -> str:
    if not _COLOR(stream):
        return text
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"


def bold(t: str, stream=None) -> str:
    return _st(t, "1", stream=stream)


def dim(t: str, stream=None) -> str:
    return _st(t, "2", stream=stream)


def cyan(t: str, stream=None) -> str:
    return _st(t, "36", stream=stream)


def green(t: str, stream=None) -> str:
    return _st(t, "32", stream=stream)


def yellow(t: str, stream=None) -> str:
    return _st(t, "33", stream=stream)


def red(t: str, stream=None) -> str:
    return _st(t, "31", stream=stream)


def link(url: str, stream=None) -> str:
    """Clickable (OSC 8) + copy-pasteable URL; plain text when piped."""
    styled = _st(url, "36", "4", stream=stream)
    if not _COLOR(stream):
        return styled
    try:
        if not sys.stdout.isatty():
            return styled
    except Exception:
        return styled
    return f"\x1b]8;;{url}\x1b\\{styled}\x1b]8;;\x1b\\"


def get_version() -> str:
    try:
        from importlib.metadata import version as _v

        return _v("visrelay")
    except Exception:
        return "0.2.0"


def header() -> None:
    v = get_version()
    print(f"{cyan('◆')} {bold('visrelay')} {dim(v)}", flush=True)


def waiting(msg: str) -> None:
    print(f"{yellow('◇')} {msg}", flush=True)


def success(msg: str) -> None:
    print(f"{green('●')} {msg}", flush=True)


def row(label: str, value: str, width: int = 8) -> None:
    print(f"  {dim(label.ljust(width))} {value}", flush=True)


def ready(host: str, port: int) -> None:
    viewer = f"http://{host}:{port}/"
    stream = f"http://{host}:{port}/stream.mjpg"
    print("", flush=True)
    row("Viewer", link(viewer))
    row("Stream", link(stream))
    print("", flush=True)
    print(f"  {dim('Press Ctrl+C to stop')}", flush=True)


def error(msg: str) -> None:
    stream = sys.stderr
    print(f"{red('✕', stream=stream)} {msg}", file=stream, flush=True)


def cancelled(msg: str = "Selection cancelled.") -> None:
    stream = sys.stderr
    print(dim(f"○ {msg}", stream=stream), file=stream, flush=True)


def stopped() -> None:
    print(dim("\n○ Stopped."), flush=True)


def sharing_stopped() -> None:
    print(f"{yellow('○')} Sharing stopped.", flush=True)


class Spinner:
    """Inline Braille spinner for the portal wait. No-op when piped."""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, msg: str):
        self._msg = msg
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tty = supports_color()

    def start(self) -> None:
        if not self._tty:
            waiting(self._msg)
            return

        def _run() -> None:
            i = 0
            while not self._stop.is_set():
                f = self.FRAMES[i % len(self.FRAMES)]
                sys.stdout.write(f"\r{yellow(f)} {self._msg}")
                sys.stdout.flush()
                self._stop.wait(0.08)
                i += 1

        self._thread = threading.Thread(target=_run, name="visrelay-spinner", daemon=True)
        self._thread.start()

    def stop(self, clear: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._tty and clear:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()
            # Static line remains in scrollback so the run is auditable.
            waiting(self._msg)
            # Small beat so ●/● ordering reads correctly.
            time.sleep(0.02)

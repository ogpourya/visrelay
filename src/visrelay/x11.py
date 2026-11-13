"""Event-driven X11 screen capture using MIT-SHM (XShm) and XDamage.

Exposes the same interface as pipewire_capture.CaptureStream:
  .width, .height, .error, .start(), .get_frame(), .is_alive(), .stop()
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import os
import select
import threading

import numpy as np

ZPIXMAP = 2
ALL_PLANES = 0xFFFFFFFF
XDAMAGE_REPORT_NON_EMPTY = 1
IPC_PRIVATE = 0
IPC_CREAT = 0o1000
IPC_RMID = 0

_c_ulong = ctypes.c_ulong
_c_int = ctypes.c_int
_c_uint = ctypes.c_uint
_c_ushort = ctypes.c_ushort
_c_void_p = ctypes.c_void_p


@dataclass(frozen=True)
class X11Monitor:
    index: int  # 1-based monitor index; 0 = combined virtual screen
    name: str
    x: int
    y: int
    width: int
    height: int
    primary: bool = False


class _XShmSegmentInfo(ctypes.Structure):
    _fields_ = [
        ("shmseg", _c_ulong),
        ("shmid", _c_int),
        ("shmaddr", _c_void_p),
        ("readOnly", _c_int),
    ]


class _XImageFuncs(ctypes.Structure):
    _fields_ = [
        ("create_image", _c_void_p),
        ("destroy_image", _c_void_p),
        ("get_pixel", _c_void_p),
        ("put_pixel", _c_void_p),
        ("sub_image", _c_void_p),
        ("add_pixel", _c_void_p),
    ]


class _XImage(ctypes.Structure):
    _fields_ = [
        ("width", _c_int),
        ("height", _c_int),
        ("xoffset", _c_int),
        ("format", _c_int),
        ("data", _c_void_p),
        ("byte_order", _c_int),
        ("bitmap_unit", _c_int),
        ("bitmap_bit_order", _c_int),
        ("bitmap_pad", _c_int),
        ("depth", _c_int),
        ("bytes_per_line", _c_int),
        ("bits_per_pixel", _c_int),
        ("red_mask", _c_ulong),
        ("green_mask", _c_ulong),
        ("blue_mask", _c_ulong),
        ("obdata", _c_void_p),
        ("f", _XImageFuncs),
    ]


class _XEvent(ctypes.Union):
    _fields_ = [
        ("type", _c_int),
        ("pad", ctypes.c_long * 24),
    ]


class _XRRScreenResources(ctypes.Structure):
    _fields_ = [
        ("timestamp", _c_ulong),
        ("configTimestamp", _c_ulong),
        ("ncrtc", _c_int),
        ("crtcs", ctypes.POINTER(_c_ulong)),
        ("noutput", _c_int),
        ("outputs", ctypes.POINTER(_c_ulong)),
        ("nmode", _c_int),
        ("modes", _c_void_p),
    ]


class _XRRCrtcInfo(ctypes.Structure):
    _fields_ = [
        ("timestamp", _c_ulong),
        ("x", _c_int),
        ("y", _c_int),
        ("width", _c_uint),
        ("height", _c_uint),
        ("mode", _c_ulong),
        ("rotation", _c_ushort),
        ("noutput", _c_int),
        ("outputs", ctypes.POINTER(_c_ulong)),
        ("rotations", _c_ushort),
        ("npossible", _c_int),
        ("possible", ctypes.POINTER(_c_ulong)),
    ]


class _XRROutputInfo(ctypes.Structure):
    _fields_ = [
        ("timestamp", _c_ulong),
        ("crtc", _c_ulong),
        ("name", ctypes.c_char_p),
        ("nameLen", _c_int),
        ("mm_width", _c_ulong),
        ("mm_height", _c_ulong),
        ("connection", _c_ushort),
        ("subpixel_order", _c_ushort),
        ("ncrtc", _c_int),
        ("crtcs", ctypes.POINTER(_c_ulong)),
        ("nclone", _c_int),
        ("clones", ctypes.POINTER(_c_ulong)),
        ("nmode", _c_int),
        ("npreferred", _c_int),
        ("modes", ctypes.POINTER(_c_ulong)),
    ]


_ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(_c_int, _c_void_p, _c_void_p)
_DESTROY_IMAGE_TYPE = ctypes.CFUNCTYPE(_c_int, ctypes.POINTER(_XImage))

_last_x11_error = 0


@_ERROR_HANDLER_TYPE
def _x11_error_handler(_dpy: int, event_ptr: int) -> int:
    global _last_x11_error
    try:
        err_code = ctypes.c_ubyte.from_address(event_ptr + 32).value
        _last_x11_error = int(err_code)
    except Exception:
        _last_x11_error = 1
    return 0


def _load_lib(name: str, fallback: str) -> ctypes.CDLL:
    path = ctypes.util.find_library(name) or fallback
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError(f"missing X11 library {fallback}: {e}") from e


def _bind_libs():
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    x11 = _load_lib("X11", "libX11.so.6")
    xext = _load_lib("Xext", "libXext.so.6")
    xdamage = _load_lib("Xdamage", "libXdamage.so.1")

    libc.shmget.argtypes = [_c_int, ctypes.c_size_t, _c_int]
    libc.shmget.restype = _c_int
    libc.shmat.argtypes = [_c_int, _c_void_p, _c_int]
    libc.shmat.restype = _c_void_p
    libc.shmdt.argtypes = [_c_void_p]
    libc.shmdt.restype = _c_int
    libc.shmctl.argtypes = [_c_int, _c_int, _c_void_p]
    libc.shmctl.restype = _c_int

    x11.XInitThreads.argtypes = []
    x11.XInitThreads.restype = _c_int
    x11.XSetErrorHandler.argtypes = [_ERROR_HANDLER_TYPE]
    x11.XSetErrorHandler.restype = _c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = _c_void_p
    x11.XCloseDisplay.argtypes = [_c_void_p]
    x11.XCloseDisplay.restype = _c_int
    x11.XDefaultScreen.argtypes = [_c_void_p]
    x11.XDefaultScreen.restype = _c_int
    x11.XRootWindow.argtypes = [_c_void_p, _c_int]
    x11.XRootWindow.restype = _c_ulong
    x11.XDefaultDepth.argtypes = [_c_void_p, _c_int]
    x11.XDefaultDepth.restype = _c_int
    x11.XDefaultVisual.argtypes = [_c_void_p, _c_int]
    x11.XDefaultVisual.restype = _c_void_p
    x11.XDisplayWidth.argtypes = [_c_void_p, _c_int]
    x11.XDisplayWidth.restype = _c_int
    x11.XDisplayHeight.argtypes = [_c_void_p, _c_int]
    x11.XDisplayHeight.restype = _c_int
    x11.XConnectionNumber.argtypes = [_c_void_p]
    x11.XConnectionNumber.restype = _c_int
    x11.XPending.argtypes = [_c_void_p]
    x11.XPending.restype = _c_int
    x11.XNextEvent.argtypes = [_c_void_p, ctypes.POINTER(_XEvent)]
    x11.XNextEvent.restype = _c_int
    x11.XFlush.argtypes = [_c_void_p]
    x11.XFlush.restype = _c_int
    x11.XSync.argtypes = [_c_void_p, _c_int]
    x11.XSync.restype = _c_int

    xext.XShmQueryExtension.argtypes = [_c_void_p]
    xext.XShmQueryExtension.restype = _c_int
    xext.XShmCreateImage.argtypes = [
        _c_void_p, _c_void_p, _c_uint, _c_int, _c_void_p,
        ctypes.POINTER(_XShmSegmentInfo), _c_uint, _c_uint,
    ]
    xext.XShmCreateImage.restype = ctypes.POINTER(_XImage)
    xext.XShmAttach.argtypes = [_c_void_p, ctypes.POINTER(_XShmSegmentInfo)]
    xext.XShmAttach.restype = _c_int
    xext.XShmDetach.argtypes = [_c_void_p, ctypes.POINTER(_XShmSegmentInfo)]
    xext.XShmDetach.restype = _c_int
    xext.XShmGetImage.argtypes = [
        _c_void_p, _c_ulong, ctypes.POINTER(_XImage), _c_int, _c_int, _c_ulong,
    ]
    xext.XShmGetImage.restype = _c_int

    xdamage.XDamageQueryExtension.argtypes = [
        _c_void_p, ctypes.POINTER(_c_int), ctypes.POINTER(_c_int),
    ]
    xdamage.XDamageQueryExtension.restype = _c_int
    xdamage.XDamageCreate.argtypes = [_c_void_p, _c_ulong, _c_int]
    xdamage.XDamageCreate.restype = _c_ulong
    xdamage.XDamageDestroy.argtypes = [_c_void_p, _c_ulong]
    xdamage.XDamageDestroy.restype = None
    xdamage.XDamageSubtract.argtypes = [_c_void_p, _c_ulong, _c_ulong, _c_ulong]
    xdamage.XDamageSubtract.restype = None

    return libc, x11, xext, xdamage


def _bind_xrandr() -> ctypes.CDLL | None:
    try:
        xrandr = _load_lib("Xrandr", "libXrandr.so.2")
    except RuntimeError:
        return None
    xrandr.XRRGetScreenResourcesCurrent.argtypes = [_c_void_p, _c_ulong]
    xrandr.XRRGetScreenResourcesCurrent.restype = ctypes.POINTER(_XRRScreenResources)
    xrandr.XRRFreeScreenResources.argtypes = [ctypes.POINTER(_XRRScreenResources)]
    xrandr.XRRFreeScreenResources.restype = None
    xrandr.XRRGetCrtcInfo.argtypes = [_c_void_p, ctypes.POINTER(_XRRScreenResources), _c_ulong]
    xrandr.XRRGetCrtcInfo.restype = ctypes.POINTER(_XRRCrtcInfo)
    xrandr.XRRFreeCrtcInfo.argtypes = [ctypes.POINTER(_XRRCrtcInfo)]
    xrandr.XRRFreeCrtcInfo.restype = None
    xrandr.XRRGetOutputInfo.argtypes = [_c_void_p, ctypes.POINTER(_XRRScreenResources), _c_ulong]
    xrandr.XRRGetOutputInfo.restype = ctypes.POINTER(_XRROutputInfo)
    xrandr.XRRFreeOutputInfo.argtypes = [ctypes.POINTER(_XRROutputInfo)]
    xrandr.XRRFreeOutputInfo.restype = None
    xrandr.XRRGetOutputPrimary.argtypes = [_c_void_p, _c_ulong]
    xrandr.XRRGetOutputPrimary.restype = _c_ulong
    return xrandr


def _query_monitors(dpy: int, root: int, screen: int, x11: ctypes.CDLL) -> tuple[list[X11Monitor], X11Monitor]:
    full_w = int(x11.XDisplayWidth(dpy, screen))
    full_h = int(x11.XDisplayHeight(dpy, screen))
    all_mon = X11Monitor(index=0, name="All monitors", x=0, y=0, width=full_w, height=full_h, primary=False)

    xrandr = _bind_xrandr()
    if xrandr is None:
        return [X11Monitor(index=1, name="Screen 0", x=0, y=0, width=full_w, height=full_h, primary=True)], all_mon

    res = xrandr.XRRGetScreenResourcesCurrent(dpy, root)
    if not res:
        return [X11Monitor(index=1, name="Screen 0", x=0, y=0, width=full_w, height=full_h, primary=True)], all_mon

    try:
        primary_out = int(xrandr.XRRGetOutputPrimary(dpy, root))
        crtc_meta: dict[int, tuple[str, bool]] = {}
        for i in range(res.contents.noutput):
            out_id = int(res.contents.outputs[i])
            oinfo = xrandr.XRRGetOutputInfo(dpy, res, out_id)
            if not oinfo:
                continue
            try:
                crtc_id = int(oinfo.contents.crtc)
                if crtc_id != 0:
                    raw_name = oinfo.contents.name or b""
                    name = raw_name.decode("utf-8", errors="replace") or f"Output-{i + 1}"
                    is_prim = out_id == primary_out
                    prev = crtc_meta.get(crtc_id)
                    if prev is None or is_prim:
                        crtc_meta[crtc_id] = (name, is_prim or (prev[1] if prev else False))
            finally:
                xrandr.XRRFreeOutputInfo(oinfo)

        raw_mons: list[tuple[int, int, int, int, str, bool]] = []
        for i in range(res.contents.ncrtc):
            crtc_id = int(res.contents.crtcs[i])
            cinfo = xrandr.XRRGetCrtcInfo(dpy, res, crtc_id)
            if not cinfo:
                continue
            try:
                if cinfo.contents.mode != 0 and cinfo.contents.width > 0 and cinfo.contents.height > 0:
                    name, is_prim = crtc_meta.get(crtc_id, (f"Display-{len(raw_mons) + 1}", False))
                    raw_mons.append((
                        int(cinfo.contents.x),
                        int(cinfo.contents.y),
                        int(cinfo.contents.width),
                        int(cinfo.contents.height),
                        name,
                        is_prim,
                    ))
            finally:
                xrandr.XRRFreeCrtcInfo(cinfo)
    finally:
        xrandr.XRRFreeScreenResources(res)

    if not raw_mons:
        return [X11Monitor(index=1, name="Screen 0", x=0, y=0, width=full_w, height=full_h, primary=True)], all_mon

    raw_mons.sort(key=lambda r: (r[0], r[1]))
    if not any(m[5] for m in raw_mons):
        x0, y0, w0, h0, n0, _ = raw_mons[0]
        raw_mons[0] = (x0, y0, w0, h0, n0, True)

    monitors = [
        X11Monitor(index=idx + 1, name=name, x=x, y=y, width=w, height=h, primary=is_prim)
        for idx, (x, y, w, h, name, is_prim) in enumerate(raw_mons)
    ]
    return monitors, all_mon


def list_monitors(display: str | None = None) -> tuple[list[X11Monitor], X11Monitor]:
    """Query active X11 monitors and the combined full-screen geometry."""
    x11 = _load_lib("X11", "libX11.so.6")
    x11.XInitThreads.argtypes = []
    x11.XInitThreads.restype = _c_int
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = _c_void_p
    x11.XCloseDisplay.argtypes = [_c_void_p]
    x11.XCloseDisplay.restype = _c_int
    x11.XDefaultScreen.argtypes = [_c_void_p]
    x11.XDefaultScreen.restype = _c_int
    x11.XRootWindow.argtypes = [_c_void_p, _c_int]
    x11.XRootWindow.restype = _c_ulong
    x11.XDisplayWidth.argtypes = [_c_void_p, _c_int]
    x11.XDisplayWidth.restype = _c_int
    x11.XDisplayHeight.argtypes = [_c_void_p, _c_int]
    x11.XDisplayHeight.restype = _c_int

    x11.XInitThreads()
    disp_b = display.encode("utf-8") if display else None
    dpy = x11.XOpenDisplay(disp_b)
    if not dpy:
        disp = display or os.environ.get("DISPLAY", "<unset>")
        raise RuntimeError(f"cannot open X11 display ({disp})")
    try:
        screen = x11.XDefaultScreen(dpy)
        root = x11.XRootWindow(dpy, screen)
        return _query_monitors(dpy, root, screen, x11)
    finally:
        x11.XCloseDisplay(dpy)


class X11Stream:
    """Zero-copy XShm screen stream gated by XDamage events."""

    def __init__(self, display: str | None = None, monitor: int = 1) -> None:
        self._display_name = display.encode("utf-8") if display else None
        self._monitor = monitor
        self._lock = threading.Lock()
        self._libc = None
        self._x11 = None
        self._xext = None
        self._xdamage = None
        self._dpy: int | None = None
        self._root: int = 0
        self._ximg = None
        self._shminfo: _XShmSegmentInfo | None = None
        self._shm_attached = False
        self._shm_addr: int | None = None
        self._damage: int = 0
        self._damage_event_type: int = 0
        self._poll: select.poll | None = None
        self._arr: np.ndarray | None = None
        self._first_frame = True
        self._closed = False
        self.monitor_name = ""
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.error: str | None = None

    def start(self) -> None:
        global _last_x11_error
        with self._lock:
            if self._dpy is not None:
                return
            libc, x11, xext, xdamage = _bind_libs()
            x11.XInitThreads()
            x11.XSetErrorHandler(_x11_error_handler)
            _last_x11_error = 0

            dpy = x11.XOpenDisplay(self._display_name)
            if not dpy:
                disp = os.environ.get("DISPLAY", "<unset>")
                raise RuntimeError(f"cannot open X11 display ({disp})")

            self._libc = libc
            self._x11 = x11
            self._xext = xext
            self._xdamage = xdamage
            self._dpy = dpy

            try:
                screen = x11.XDefaultScreen(dpy)
                root = x11.XRootWindow(dpy, screen)
                depth = x11.XDefaultDepth(dpy, screen)
                if depth not in (24, 32):
                    raise RuntimeError(f"unsupported X11 color depth {depth} (expected 24 or 32)")

                if not xext.XShmQueryExtension(dpy):
                    raise RuntimeError("X11 MIT-SHM (XShm) extension unavailable")

                ev_base = _c_int(0)
                err_base = _c_int(0)
                if not xdamage.XDamageQueryExtension(dpy, ctypes.byref(ev_base), ctypes.byref(err_base)):
                    raise RuntimeError("X11 XDamage extension unavailable")

                monitors, all_mon = _query_monitors(dpy, root, screen, x11)
                if self._monitor == 0:
                    target = all_mon
                elif 1 <= self._monitor <= len(monitors):
                    target = monitors[self._monitor - 1]
                else:
                    raise RuntimeError(
                        f"monitor {self._monitor} out of range (found {len(monitors)} active monitor(s); use 0 for all)"
                    )

                x, y, w, h = target.x, target.y, target.width, target.height
                if w <= 0 or h <= 0:
                    raise RuntimeError(f"invalid X11 capture dimensions {w}x{h}")

                visual = x11.XDefaultVisual(dpy, screen)
                shminfo = _XShmSegmentInfo()
                ximg = xext.XShmCreateImage(
                    dpy, visual, depth, ZPIXMAP, None, ctypes.byref(shminfo), w, h
                )
                if not ximg:
                    raise RuntimeError("XShmCreateImage failed")
                self._ximg = ximg
                self._shminfo = shminfo

                stride = int(ximg.contents.bytes_per_line)
                size = stride * h
                shmid = libc.shmget(IPC_PRIVATE, size, IPC_CREAT | 0o600)
                if shmid < 0:
                    raise RuntimeError("shmget failed")
                shminfo.shmid = shmid

                addr = libc.shmat(shmid, None, 0)
                if not addr or addr == ctypes.c_void_p(-1).value:
                    libc.shmctl(shmid, IPC_RMID, None)
                    raise RuntimeError("shmat failed")
                self._shm_addr = addr
                shminfo.shmaddr = addr
                shminfo.readOnly = 0
                ximg.contents.data = addr

                if not xext.XShmAttach(dpy, ctypes.byref(shminfo)):
                    raise RuntimeError("XShmAttach failed")
                x11.XSync(dpy, 0)
                libc.shmctl(shmid, IPC_RMID, None)
                if _last_x11_error:
                    raise RuntimeError(f"X11 error {_last_x11_error} during XShmAttach")
                self._shm_attached = True

                damage = xdamage.XDamageCreate(dpy, root, XDAMAGE_REPORT_NON_EMPTY)
                x11.XSync(dpy, 0)
                if not damage or _last_x11_error:
                    raise RuntimeError("XDamageCreate failed")

                self._root = root
                self._damage = damage
                self._damage_event_type = int(ev_base.value)
                self.monitor_name = target.name
                self.x = x
                self.y = y
                self.width = w
                self.height = h

                raw = (ctypes.c_ubyte * size).from_address(addr)
                self._arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, stride // 4, 4))[:, :w, :]

                xfd = int(x11.XConnectionNumber(dpy))
                p = select.poll()
                p.register(xfd, select.POLLIN | select.POLLHUP | select.POLLERR)
                self._poll = p
            except Exception:
                self._cleanup_locked()
                raise

    def is_alive(self) -> bool:
        return not self._closed and self._dpy is not None and self.error is None

    def get_frame(self) -> np.ndarray | None:
        global _last_x11_error
        with self._lock:
            if self._closed or self._dpy is None or self._arr is None:
                return None

            dpy = self._dpy
            x11 = self._x11
            ev = _XEvent()

            if self._first_frame:
                self._first_frame = False
                while x11.XPending(dpy) > 0:
                    x11.XNextEvent(dpy, ctypes.byref(ev))
                self._xdamage.XDamageSubtract(dpy, self._damage, 0, 0)
                ok = self._xext.XShmGetImage(
                    dpy, self._root, self._ximg, self.x, self.y, ALL_PLANES
                )
                if not ok or _last_x11_error:
                    self.error = f"XShmGetImage failed (err={_last_x11_error})"
                    return None
                return self._arr

            has_input = x11.XPending(dpy) > 0
            if not has_input and self._poll is not None:
                events = self._poll.poll(0)
                for _fd, mask in events:
                    if mask & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                        self.error = "X11 display disconnected"
                        return None
                    if mask & select.POLLIN:
                        has_input = True

            if not has_input:
                return None

            damaged = False
            while x11.XPending(dpy) > 0:
                x11.XNextEvent(dpy, ctypes.byref(ev))
                if ev.type == self._damage_event_type:
                    damaged = True

            if not damaged:
                return None

            self._xdamage.XDamageSubtract(dpy, self._damage, 0, 0)
            ok = self._xext.XShmGetImage(
                dpy, self._root, self._ximg, self.x, self.y, ALL_PLANES
            )
            if not ok or _last_x11_error:
                self.error = f"XShmGetImage failed (err={_last_x11_error})"
                return None
            return self._arr

    def stop(self) -> None:
        with self._lock:
            self._cleanup_locked()

    def _cleanup_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._arr = None
        dpy = self._dpy
        if dpy is not None:
            if self._damage and self._xdamage is not None:
                try:
                    self._xdamage.XDamageDestroy(dpy, self._damage)
                except Exception:
                    pass
                self._damage = 0
            if self._shm_attached and self._xext is not None and self._shminfo is not None:
                try:
                    self._xext.XShmDetach(dpy, ctypes.byref(self._shminfo))
                except Exception:
                    pass
                self._shm_attached = False
            if self._ximg is not None:
                try:
                    self._ximg.contents.data = None
                    destroy_fn = self._ximg.contents.f.destroy_image
                    if destroy_fn:
                        _DESTROY_IMAGE_TYPE(destroy_fn)(self._ximg)
                except Exception:
                    pass
                self._ximg = None
            if self._shm_addr is not None and self._libc is not None:
                try:
                    self._libc.shmdt(self._shm_addr)
                except Exception:
                    pass
                self._shm_addr = None
            if self._x11 is not None:
                try:
                    self._x11.XCloseDisplay(dpy)
                except Exception:
                    pass
            self._dpy = None

"""Wayland ScreenCast portal session via raw D-Bus messages.

Bypasses broken XML introspection by issuing raw portal calls, registers
session Closed signals for instant termination detection, and returns the
PipeWire FD + node id + dimensions + session controller.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Callable

from . import _compat as _compat_shim  # noqa: F401  (patches typing for dbus-next on 3.15+)
from dbus_next.aio import MessageBus
from dbus_next.constants import MessageType
from dbus_next.message import Message
from dbus_next.signature import Variant

PORTAL_DEST = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
SESSION_IFACE = "org.freedesktop.portal.Session"


class PortalSession:
    """Manages Wayland ScreenCast portal session lifetime and signals."""

    def __init__(
        self,
        pw_fd: int,
        node_id: int,
        width: int,
        height: int,
        bus: MessageBus,
        session_handle: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.pw_fd = pw_fd
        self.node_id = node_id
        self.width = width
        self.height = height
        self.bus = bus
        self.session_handle = session_handle
        self.loop = loop
        self._on_closed: Callable[[], None] | None = None
        self._loop_thread: threading.Thread | None = None
        self._cleaned_up = False

    def __iter__(self):
        """Preserve backwards compatibility for tuple unpacking."""
        return iter((self.pw_fd, self.node_id, self.width, self.height, self.bus))

    def start_listener(self, on_closed: Callable[[], None]) -> None:
        """Run the D-Bus asyncio loop in background to catch Session.Closed instantly."""
        self._on_closed = on_closed

        def _message_handler(msg: Message) -> bool | None:
            if (
                msg.message_type == MessageType.SIGNAL
                and msg.interface == SESSION_IFACE
                and msg.member == "Closed"
                and msg.path == self.session_handle
            ):
                if self._on_closed:
                    self._on_closed()
                return True
            return None

        self.bus.add_message_handler(_message_handler)

        def _run_loop() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run_loop, name="visrelay-dbus-loop", daemon=True
        )
        self._loop_thread.start()

    def close(self) -> None:
        """Cleanly close portal session and release D-Bus resources."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        # Ask the portal to close the session so GNOME/KDE hides the recording indicator
        if self.loop.is_running():
            try:
                close_msg = Message(
                    destination=PORTAL_DEST,
                    path=self.session_handle,
                    interface=SESSION_IFACE,
                    member="Close",
                    signature="",
                    body=[],
                )
                fut = asyncio.run_coroutine_threadsafe(self.bus.call(close_msg), self.loop)
                fut.result(timeout=0.5)
            except Exception:
                pass

            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=0.5)

        try:
            self.bus.disconnect()
        except Exception:
            pass

        try:
            if not self.loop.is_closed():
                self.loop.close()
        except Exception:
            pass


async def request_wayland_screen(loop: asyncio.AbstractEventLoop | None = None) -> PortalSession | None:
    """Request a ScreenCast session, return PortalSession or None."""
    bus = await MessageBus(negotiate_unix_fd=True).connect()
    current_loop = loop or asyncio.get_running_loop()
    try:
        sender_name = bus.unique_name.lstrip(":").replace(".", "_")
        token_counter = 0

        def next_token() -> str:
            nonlocal token_counter
            token_counter += 1
            return f"u{token_counter}"

        async def call_portal(member: str, signature: str, body: list):
            token = next_token()
            body[-1]["handle_token"] = Variant("s", token)
            req_path = f"/org/freedesktop/portal/desktop/request/{sender_name}/{token}"

            fut = current_loop.create_future()

            def on_notify(msg: Message):
                if (
                    msg.message_type == MessageType.SIGNAL
                    and msg.path == req_path
                    and msg.member == "Response"
                ):
                    if not fut.done():
                        fut.set_result((msg.body[0], msg.body[1]))
                    return True

            bus.add_message_handler(on_notify)
            match_rule = (
                "type='signal',interface='org.freedesktop.portal.Request',"
                f"path='{req_path}'"
            )
            await bus.call(
                Message(
                    destination="org.freedesktop.DBus",
                    path="/org/freedesktop/DBus",
                    interface="org.freedesktop.DBus",
                    member="AddMatch",
                    signature="s",
                    body=[match_rule],
                )
            )

            try:
                reply = await bus.call(
                    Message(
                        destination=PORTAL_DEST,
                        path=PORTAL_PATH,
                        interface=SCREENCAST_IFACE,
                        member=member,
                        signature=signature,
                        body=body,
                    )
                )
                if reply.message_type == MessageType.ERROR:
                    if not fut.done():
                        fut.cancel()
                    raise RuntimeError(f"Portal {member} failed: {reply.body}")
                timeout = 120.0 if member == "Start" else 30.0
                return await asyncio.wait_for(fut, timeout=timeout)
            finally:
                bus.remove_message_handler(on_notify)
                try:
                    await bus.call(
                        Message(
                            destination="org.freedesktop.DBus",
                            path="/org/freedesktop/DBus",
                            interface="org.freedesktop.DBus",
                            member="RemoveMatch",
                            signature="s",
                            body=[match_rule],
                        )
                    )
                except Exception:
                    pass

        # 1. Create Portal Session
        code, res = await call_portal(
            "CreateSession",
            "a{sv}",
            [{"session_handle_token": Variant("s", next_token())}],
        )
        if code != 0:
            bus.disconnect()
            return None
        session_handle = res["session_handle"].value

        # 2. Select Sources: types=3 enables Monitor (1) and Window (2)
        code, _ = await call_portal(
            "SelectSources",
            "oa{sv}",
            [
                session_handle,
                {
                    "types": Variant("u", 3),
                    "multiple": Variant("b", False),
                },
            ],
        )
        if code != 0:
            bus.disconnect()
            return None

        # 3. Start Portal Stream (shows system dialog)
        code, res = await call_portal("Start", "osa{sv}", [session_handle, "", {}])
        if code != 0:
            bus.disconnect()
            return None

        if "streams" not in res or len(res["streams"].value) == 0:
            bus.disconnect()
            return None

        streams = res["streams"].value
        node_id = int(streams[0][0])
        props = streams[0][1]
        size = props.get("size").value if "size" in props else [1920, 1080]
        width, height = int(size[0]), int(size[1])

        # Match rule for Session Closed signal
        session_match_rule = (
            f"type='signal',interface='{SESSION_IFACE}',member='Closed',path='{session_handle}'"
        )
        await bus.call(
            Message(
                destination="org.freedesktop.DBus",
                path="/org/freedesktop/DBus",
                interface="org.freedesktop.DBus",
                member="AddMatch",
                signature="s",
                body=[session_match_rule],
            )
        )

        # 4. Get PipeWire file descriptor
        reply = await bus.call(
            Message(
                destination=PORTAL_DEST,
                path=PORTAL_PATH,
                interface=SCREENCAST_IFACE,
                member="OpenPipeWireRemote",
                signature="oa{sv}",
                body=[session_handle, {}],
            )
        )
        if reply.message_type == MessageType.ERROR:
            bus.disconnect()
            raise RuntimeError(f"Portal OpenPipeWireRemote failed: {reply.body}")

        if not reply.unix_fds:
            bus.disconnect()
            raise RuntimeError(f"OpenPipeWireRemote returned no UNIX FDs: {reply.body}")

        idx = reply.body[0] if reply.body else 0
        try:
            idx = int(idx)
        except Exception:
            idx = 0
        if not 0 <= idx < len(reply.unix_fds):
            bus.disconnect()
            raise RuntimeError(f"Portal OpenPipeWireRemote bad fd index: {reply.body}")

        raw_fd = reply.unix_fds[idx]
        pw_fd = os.dup(raw_fd)
        for fd in reply.unix_fds:
            try:
                os.close(fd)
            except OSError:
                pass

        return PortalSession(
            pw_fd=pw_fd,
            node_id=node_id,
            width=width,
            height=height,
            bus=bus,
            session_handle=session_handle,
            loop=current_loop,
        )
    except Exception:
        try:
            bus.disconnect()
        except Exception:
            pass
        raise

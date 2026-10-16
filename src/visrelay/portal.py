"""Wayland ScreenCast portal session via raw D-Bus messages.

Reused from main.py prototype: bypasses broken XML introspection by issuing
raw portal calls, then returns the PipeWire FD + node id + dimensions.
"""

from __future__ import annotations

import asyncio
import os

from . import _compat as _compat_shim  # noqa: F401  (patches typing for dbus-next on 3.15+)
from dbus_next.aio import MessageBus
from dbus_next.constants import MessageType
from dbus_next.message import Message
from dbus_next.signature import Variant

PORTAL_DEST = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"


async def request_wayland_screen():
    """Request a ScreenCast session, return (pw_fd, node_id, w, h, bus) or None."""
    # negotiate_unix_fd=True is required to receive the PipeWire file descriptor
    bus = await MessageBus(negotiate_unix_fd=True).connect()
    try:
        sender_name = bus.unique_name.lstrip(":").replace(".", "_")

        token_counter = 0

        def next_token():
            nonlocal token_counter
            token_counter += 1
            return f"u{token_counter}"

        async def call_portal(member, signature, body):
            token = next_token()
            body[-1]["handle_token"] = Variant("s", token)
            req_path = f"/org/freedesktop/portal/desktop/request/{sender_name}/{token}"

            loop = asyncio.get_running_loop()
            fut = loop.create_future()

            def on_notify(msg):
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

        # 2. Select Sources: types=3 enables BOTH Monitor (1) and Window (2)
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

        # 3. Start Portal Stream (shows the system picker popup)
        code, res = await call_portal("Start", "osa{sv}", [session_handle, "", {}])
        if code != 0:
            bus.disconnect()
            return None

        if "streams" not in res or len(res["streams"].value) == 0:
            bus.disconnect()
            return None

        streams = res["streams"].value
        node_id, props = streams[0]
        size = props["size"].value if "size" in props else [1920, 1080]
        width, height = int(size[0]), int(size[1])

        # 4. Get the raw PipeWire file descriptor
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

        if reply.unix_fds:
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
        else:
            raw_fd = reply.body[0]
            pw_fd = os.dup(raw_fd)

        return pw_fd, node_id, width, height, bus
    except Exception:
        try:
            bus.disconnect()
        except Exception:
            pass
        raise

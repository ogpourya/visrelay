import asyncio
import os
import shutil
import tempfile
import time
from dbus_next.aio import MessageBus
from dbus_next.constants import MessageType
from dbus_next.message import Message
from dbus_next.signature import Variant
import numpy as np
from pipewire_capture import CaptureStream
import simplejpeg

PORTAL_DEST = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"


async def request_wayland_screen():
  """Requests a ScreenCast session via raw D-Bus messages (bypasses broken XML introspection)."""
  # negotiate_unix_fd=True is required to receive the PipeWire file descriptor
  bus = await MessageBus(negotiate_unix_fd=True).connect()
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
    await bus.call(
        Message(
            destination="org.freedesktop.DBus",
            path="/org/freedesktop/DBus",
            interface="org.freedesktop.DBus",
            member="AddMatch",
            signature="s",
            body=[
                "type='signal',interface='org.freedesktop.portal.Request',"
                f"path='{req_path}'"
            ],
        )
    )

    try:
      await bus.call(
          Message(
              destination=PORTAL_DEST,
              path=PORTAL_PATH,
              interface=SCREENCAST_IFACE,
              member=member,
              signature=signature,
              body=body,
          )
      )
      return await fut
    finally:
      bus.remove_message_handler(on_notify)

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

  # 2. Select Sources: types=3 enables BOTH Whole Screen/Monitor (1) and App Window (2)
  code, _ = await call_portal(
      "SelectSources",
      "oa{sv}",
      [
          session_handle,
          {
              "types": Variant("u", 3),  # 1 = Monitor, 2 = Window, 3 = Both
              "multiple": Variant("b", False),
          },
      ],
  )
  if code != 0:
    bus.disconnect()
    return None

  # 3. Start Portal Stream (Shows the Ubuntu system popup with Whole Screen tab)
  code, res = await call_portal("Start", "osa{sv}", [session_handle, "", {}])
  if code != 0:
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

  # Extract the Unix FD safely whether dbus-next places it in unix_fds or body[0]
  raw_fd = reply.unix_fds[0] if reply.unix_fds else reply.body[0]
  pw_fd = os.dup(raw_fd)

  return pw_fd, node_id, width, height, bus


def monitor_wayland_fast(target_per_minute=60):
  interval = 60.0 / target_per_minute
  temp_dir = tempfile.mkdtemp(prefix="wayland_fps_")

  print("[INFO] Opening Ubuntu Wayland Screen Picker (Whole Screen enabled)...")
  loop = asyncio.new_event_loop()
  session_data = loop.run_until_complete(request_wayland_screen())

  if session_data is None:
    print("[ERROR] Screen selection cancelled by user.")
    shutil.rmtree(temp_dir)
    return

  pw_fd, node_id, width, height, bus = session_data

  # Pass the D-Bus PipeWire FD & Node ID directly into the Rust CaptureStream
  stream = CaptureStream(pw_fd, node_id, width, height, capture_interval=0.0)
  stream.start()

  print(
      f"[INFO] Capturing at {width}x{height} -> Saving temporary JPEGs to:"
      f" {temp_dir}"
  )
  print("[INFO] PipeWire stream active! Press Ctrl+C to stop and clean up.\n")

  frame_count = 0
  total_process_time = 0.0

  # Wait briefly for the first PipeWire frame buffer to arrive
  frame = None
  while frame is None and not getattr(stream, "window_invalid", False):
    frame = stream.get_frame()
    time.sleep(0.005)

  start_session_time = time.perf_counter()

  try:
    while not getattr(stream, "window_invalid", False):
      loop_start = time.perf_counter()

      # 1. Grab latest frame from the PipeWire stream buffer
      latest_frame = stream.get_frame()
      if latest_frame is not None:
        frame = latest_frame

      # Detect channel layout dynamically (RGB vs BGRA)
      channels = frame.shape[2] if frame.ndim == 3 else 4
      colorspace = "RGB" if channels == 3 else "BGRA"

      # 2. Encode array directly to JPEG using SIMD libjpeg-turbo
      jpeg_bytes = simplejpeg.encode_jpeg(
          np.ascontiguousarray(frame),
          quality=85,
          colorspace=colorspace,
          fastdct=True,
      )

      file_path = os.path.join(temp_dir, f"shot_{frame_count:05d}.jpg")
      with open(file_path, "wb") as f:
        f.write(jpeg_bytes)

      # 3. Calculate performance metrics
      process_time = time.perf_counter() - loop_start
      total_process_time += process_time
      frame_count += 1

      theoretical_max_fps = (
          1.0 / process_time if process_time > 0 else float("inf")
      )

      # 4. Pace the loop to hit 60/minute
      sleep_time = max(0.0, interval - process_time)
      if sleep_time > 0:
        time.sleep(sleep_time)

      total_loop_time = time.perf_counter() - loop_start
      actual_fps = 1.0 / total_loop_time

      print(
          f"Frame #{frame_count:04d} | "
          f"PipeWire+JPEG+Disk: {process_time*1000:5.2f} ms | "
          f"Max Capacity: {theoretical_max_fps:6.1f} FPS | "
          f"Paced: {actual_fps*60:.1f}/min"
      )

    if getattr(stream, "error", None):
      print(f"\n[ERROR] PipeWire stream closed: {stream.error}")

  except KeyboardInterrupt:
    print("\n\n[INFO] Stopping script via user interrupt (Ctrl+C)...")

  finally:
    # Stop PipeWire thread, close FD, and disconnect D-Bus portal session
    stream.stop()
    try:
      os.close(pw_fd)
    except OSError:
      pass
    bus.disconnect()
    loop.close()

    total_runtime = time.perf_counter() - start_session_time
    if frame_count > 0:
      avg_ms = (total_process_time / frame_count) * 1000
      avg_max_fps = 1.0 / (total_process_time / frame_count)
      print("\n--- Ubuntu Wayland Performance Summary ---")
      print(f"Total Screenshots Saved : {frame_count}")
      print(f"Total Runtime           : {total_runtime:.2f}s")
      print(f"Avg Grab + JPEG + Disk  : {avg_ms:.2f} ms")
      print(f"Avg Max Hardware FPS    : {avg_max_fps:.1f} FPS")
      print("------------------------------------------")

    if os.path.exists(temp_dir):
      shutil.rmtree(temp_dir)
      print(
          f"[CLEANUP] Deleted {frame_count} JPEG files and removed {temp_dir}"
      )


if __name__ == "__main__":
  monitor_wayland_fast(target_per_minute=60)

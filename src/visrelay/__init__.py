"""visrelay: Wayland screen -> HTTP MJPEG relay.

Defaults: bind 0.0.0.0:29173 @ 15 fps (paced to leave CPU for HTTP serving).
"""

from .cli import DEFAULT_FPS, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_QUALITY, main

__all__ = ["DEFAULT_FPS", "DEFAULT_HOST", "DEFAULT_PORT", "DEFAULT_QUALITY", "main"]

# visrelay

Streams the selected screen or application to an HTTP address.

## Defaults

- Listens on `0.0.0.0:29173` at `24` fps
- `/` viewer, `/stream.mjpg` MJPEG, `/snapshot.jpg` still, `/healthz` status
- Pauses capture when nobody watches; resumes on connect
- Skips re-encode on static screens (cached JPEG re-served)

## Supported platforms

Linux Desktop: Wayland (desktop portal + PipeWire; approve the system picker when prompted) and X11 (XFCE, etc. via XShm + XDamage; prompts to pick a screen or window, or pass `--monitor 1` / `0` / `--window <title-or-id>`). `visrelay --demo` runs anywhere without a display.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv tool install https://github.com/ogpourya/visrelay.git
```

## Run

```sh
visrelay
visrelay --host 0.0.0.0 --port 29173 --fps 24
visrelay --demo
```

Open `http://<host>:29173/`.

## Known limitations

Fullscreen games or maximized windows can freeze the stream: the compositor stops sending frames and there is nothing to relay (OBS has the same wall). Un-maximizing usually resumes it. A permanently dead feed exits with `○ Capture stalled.` instead of freezing forever.

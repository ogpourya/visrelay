# visrelay

Stream the selected Wayland screen or window to HTTP as MJPEG.

Defaults: bind `0.0.0.0:29173` at 15 fps. Endpoints: `/` (viewer), `/stream.mjpg`, `/snapshot.jpg`, `/healthz`.

## Support scope

Currently supports and is tested only on the author's system:

- Ubuntu 26.04 LTS (kernel 7.0.0-38-generic), GNOME Shell 50.1 on Wayland (`ubuntu:GNOME`)
- PipeWire 1.6.2 via xdg-desktop-portal ScreenCast
- Python 3.15, Intel i3-9100F, AMD Ellesmere RX 470/480/570/580/590

Other systems are not supported and not tested for now.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv tool install git+https://github.com/anomalyco/visrelay
visrelay --help
```

## Run

```sh
visrelay
# custom bind / rate
visrelay --host 0.0.0.0 --port 29173 --fps 15
# no screen picker (synthetic pattern, for testing HTTP)
visrelay --demo
```

Open `http://<host>:29173/`.

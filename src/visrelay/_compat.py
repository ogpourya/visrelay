"""Python-version compatibility shims. Import first, before third-party deps."""

from __future__ import annotations

import typing

# dbus-next 0.2.3 does `from typing import no_type_check_decorator`, which was
# removed in Python 3.15 (only `no_type_check` remains). The whole dbus_next
# package fails to import without it, even though we only need dbus_next.aio.
# Restore the 3.12 semantics: mark the decorated callable as not type-checked.
if not hasattr(typing, "no_type_check_decorator"):
    def _no_type_check_decorator(decorator):  # type: ignore[no-redef]
        def _wrapper(func):
            f = decorator(func)
            f.__no_type_check__ = True
            return f

        return _wrapper

    typing.no_type_check_decorator = _no_type_check_decorator  # type: ignore[attr-defined]

"""Lets the project's shared code run under Python 3.9, which mmrotate 1.0.0rc1 requires.

The shared code calls zip(..., strict=True), added in Python 3.10. Import this module before any
herdsight module. The replacement keeps the check: unequal lengths still raise ValueError.
"""

from __future__ import annotations

import builtins
import sys
from itertools import zip_longest

if sys.version_info < (3, 10):  # noqa: UP036  (the G7 environment is Python 3.9)
    _zip = builtins.zip
    _missing = object()

    def _strict_zip(*iterables, strict=False):  # noqa: ANN001, ANN202
        if not strict:
            return _zip(*iterables)  # noqa: B905

        def checked():  # noqa: ANN202
            for row in zip_longest(*iterables, fillvalue=_missing):
                if any(v is _missing for v in row):
                    raise ValueError("zip() arguments have different lengths")
                yield row

        return checked()

    builtins.zip = _strict_zip

"""Where individual animal identities would come from (D-12).

Stage A counts animals by vision only and needs no identities. This interface is left empty on
purpose, so ear tags or collars can be added later as a separate product without changing the
register or the alert rules. Nothing in Stage A implements or calls it.
"""

from __future__ import annotations

from typing import Protocol


class IdentitySource(Protocol):
    """A source of animal identities seen in a camp (for example tag readers)."""

    def animals_seen(self, camp: str, day: float) -> set[str]:
        """Identities seen in `camp` on `day`."""
        ...

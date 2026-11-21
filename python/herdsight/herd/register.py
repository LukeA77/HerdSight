"""The herd register: the expected number of animals in each camp (D-13).

The expected count changes only through recorded events, kept in an append-only log, so the
expected count at any past time can be rebuilt and every alert explained. A move between camps is
one event with two sides, so the farm total is unchanged by it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Kind(StrEnum):
    BIRTH = "birth"
    PURCHASE = "purchase"
    SALE = "sale"
    DEATH = "death"
    MOVE = "move"
    CORRECTION = "correction"  # a counted difference the farmer accepts (e.g. after a check)


ADDS = {Kind.BIRTH, Kind.PURCHASE}
REMOVES = {Kind.SALE, Kind.DEATH}


@dataclass(frozen=True)
class Event:
    day: float
    kind: Kind
    camp: str
    n: int
    to_camp: str | None = None  # MOVE only
    note: str = ""

    def __post_init__(self) -> None:
        if self.n < 0 and self.kind is not Kind.CORRECTION:
            raise ValueError("n must be positive except for corrections")
        if (self.kind is Kind.MOVE) != (self.to_camp is not None):
            raise ValueError("to_camp is required for moves and only for moves")


@dataclass
class HerdRegister:
    """Expected count per camp, from a start count and the recorded events."""

    start: dict[str, int]
    events: list[Event] = field(default_factory=list)

    def record(self, event: Event) -> None:
        if event.camp not in self.start or (event.to_camp and event.to_camp not in self.start):
            raise KeyError(f"unknown camp in {event}")
        self.events.append(event)

    def expected(self, camp: str, day: float) -> int:
        """Expected animals in `camp` after every event recorded up to and including `day`."""
        n = self.start[camp]
        for e in self.events:
            if e.day > day:
                continue
            if e.kind in ADDS and e.camp == camp:
                n += e.n
            elif e.kind in REMOVES and e.camp == camp:
                n -= e.n
            elif e.kind is Kind.CORRECTION and e.camp == camp:
                n += e.n
            elif e.kind is Kind.MOVE:
                n -= e.n if e.camp == camp else 0
                n += e.n if e.to_camp == camp else 0
        return n

    def farm_total(self, day: float) -> int:
        return sum(self.expected(c, day) for c in self.start)

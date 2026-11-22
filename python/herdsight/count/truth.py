"""The owner's link review applied to the candidate identities: which labels are one animal.

The candidate identities of a flight (`components` in the saved links,
$HERDSIGHT_STORAGE/processed/s3_links/) are groups of labelled boxes, each box written as
(image index, box index). The review (scripts/make_link_review.py, three rounds) asked two kinds
of question, and `apply_review` applies the answers by the rules in scripts/build_s3_truth.py:

* "Same animal?": a picture marked as a different animal is split off as its own group;
* "Missed link?": "boxed" joins the lone animal's group to the group of the nearest box at the
  matching place in the other image; other answers change no group.

A group may end with two boxes from one image (a conflict: a "boxed" answer joined two groups
that share an image). Such a group is at least as many animals as its most repeated image shows,
which is how the verified count treats it; for pair labels its members are ambiguous.

`verified_identities` reads every review round from the storage drive and returns the groups of
one flight; scripts/build_s3_truth.py writes the counts from the same function.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from herdsight.utils.paths import storage_path

Member = tuple[int, int]  # (image index, labelled box index)
REVIEW_FOLDERS = ("link_review", "link_review_2", "link_review_3")


@dataclass
class Reviewed:
    groups: list[list[Member]]
    splits: int = 0
    merges: int = 0
    missing_labels: int = 0
    unanswered: int = 0
    conflicts: list[int] = field(default_factory=list)  # indices of groups with a repeated image

    @property
    def count(self) -> int:
        return sum(max(Counter(i for i, _ in g).values()) for g in self.groups)


def apply_review(components: list, tasks: list[dict], answers: dict) -> Reviewed:
    """Apply one flight's answers to its candidate groups (`components` as saved)."""
    comps: list[list[Member]] = [[tuple(m) for m in c] for c in components]  # type: ignore[misc]
    owner: dict[Member, int] = {m: ci for ci, c in enumerate(comps) for m in c}
    r = Reviewed(groups=[])
    for t in tasks:
        a = answers.get(t["id"])
        if a is None:
            r.unanswered += 1
            continue
        if t["kind"] == "same":
            for n in a.get("different", []):
                m: Member = tuple(t["pics"][n]["member"])  # type: ignore[assignment]
                comps[owner[m]].remove(m)
                comps.append([m])
                owner[m] = len(comps) - 1
                r.splits += 1
        elif a.get("answer") == "boxed" and t["other"][1] is not None:
            ca, cb = owner[tuple(t["member"])], owner[tuple(t["other"])]  # type: ignore[index]
            if ca != cb:
                for m in comps[cb]:
                    owner[m] = ca
                comps[ca] += comps[cb]
                comps[cb] = []
                r.merges += 1
        elif a.get("answer") == "unboxed":
            r.missing_labels += 1
    r.groups = [c for c in comps if c]
    r.conflicts = [k for k, g in enumerate(r.groups) if len({i for i, _ in g}) < len(g)]
    return r


def load_review(paths: list[Path]) -> tuple[dict, dict[str, list[dict]], set[str]]:
    """Answers, tasks by flight and reviewed flights from answer files (each next to its
    review's tasks.json)."""
    answers: dict = {}
    tasks: dict[str, list[dict]] = {}
    flights: set[str] = set()
    for path in paths:
        answers |= json.loads(path.read_text(encoding="utf-8"))
        task_file = json.loads((path.parent / "tasks.json").read_text(encoding="utf-8"))
        for t in task_file["tasks"]:
            tasks.setdefault(t["flight"], []).append(t)
        flights.update(task_file["flights"].values())  # includes flights with no questions
    for key in flights:
        tasks.setdefault(key, [])
    return answers, tasks, flights


def review_answer_files() -> list[Path]:
    """The answer files of every review round on the storage drive, in round order."""
    out = []
    for n, folder in enumerate(REVIEW_FOLDERS, start=1):
        name = "link_answers.json" if n == 1 else f"link_answers_{n}.json"
        p = storage_path("exports", folder) / name
        if p.exists():
            out.append(p)
    return out


def verified_identities(key: str, links: dict | None = None) -> Reviewed:
    """The reviewed groups of flight `key` ("farm/flight"); `links` is its saved links file
    (read from the storage drive when not given)."""
    answers, tasks, flights = load_review(review_answer_files())
    if key not in flights:
        raise KeyError(f"{key} was not reviewed")
    if links is None:
        farm, flight = key.split("/")
        path = storage_path("processed", "s3_links") / f"{farm}__{flight}.json"
        links = json.loads(path.read_text(encoding="utf-8"))
    return apply_review(links["components"], tasks[key], answers)

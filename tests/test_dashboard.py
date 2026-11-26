"""Farmer dashboard backend: store, mock farm and web API (Phase 6)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import jsonschema
import pytest

from herdsight.dashboard import mock
from herdsight.dashboard.server import farm_view, make_server
from herdsight.dashboard.store import Store, validate
from herdsight.herd.messages import banned_words


@pytest.fixture
def store() -> Store:
    s = Store()
    mock.fill(s)
    yield s
    s.close()


def test_mock_records_are_valid_and_cover_every_status(store: Store) -> None:
    recs = mock.records()
    for r in recs:
        validate(r)
        assert r["simulated"] is True
    statuses = {r["status"] for r in store.latest(mock.FARM_ID).values()}
    assert statuses == {"ok", "check", "moved", "missing", "unknown"}


def test_store_rejects_a_record_that_breaks_the_schema(store: Store) -> None:
    bad = dict(mock.records()[0], count=-1)
    with pytest.raises(jsonschema.ValidationError):
        store.add(bad)


def test_latest_history_and_alerts(store: Store) -> None:
    latest = store.latest(mock.FARM_ID)
    assert len(latest) == len(mock.CAMPS)
    hist = store.history(mock.FARM_ID, "c4")
    assert len(hist) == mock.WEEKS
    assert hist[0]["surveyed_at"] > hist[-1]["surveyed_at"]  # newest first
    alerts = store.open_alerts(mock.FARM_ID)
    assert {a["camp_id"] for a in alerts} == {"c3", "c4"}


def test_actions_close_alerts_but_a_second_check_does_not(store: Store) -> None:
    alert = next(a for a in store.open_alerts(mock.FARM_ID) if a["camp_id"] == "c4")
    store.act(alert["alert_id"], "second_check")
    assert any(a["camp_id"] == "c4" for a in store.open_alerts(mock.FARM_ID))
    store.act(alert["alert_id"], "found_them")
    assert not any(a["camp_id"] == "c4" for a in store.open_alerts(mock.FARM_ID))
    with pytest.raises(ValueError):
        store.act(alert["alert_id"], "delete_everything")
    with pytest.raises(KeyError):
        store.act("no-such-alert", "dismissed")


def test_farm_view_lists_camps_needing_attention_first(store: Store) -> None:
    v = farm_view(store, mock.FARM_ID)
    assert v["needs_attention"][0] == "c4"  # missing comes before moved, checking and unknown
    assert set(v["needs_attention"]) == {"c1", "c3", "c4", "c5", "c6"}  # c5: more than records
    assert v["simulated"] is True
    for c in v["camps"]:
        assert not banned_words(c["label"])
        for note in (c["latest"] or {}).get("notes", []):
            assert not banned_words(note)


def test_matches_only_when_records_are_inside_the_range(store: Store) -> None:
    calving = store.latest(mock.FARM_ID)["c5"]
    assert "more than your records show" in calving["notes"][0]
    assert calving["status"] == "check"  # never "all here" with clearly more animals
    top = store.latest(mock.FARM_ID)["c2"]
    assert "matches your records" in top["notes"][0]


def _call(base: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            kind = r.headers.get("Content-Type", "")
            return r.status, json.loads(raw) if "json" in kind else raw.decode()
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_web_api_end_to_end(store: Store) -> None:
    srv = make_server(store, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert _call(base, "/api/farms") == (200, [mock.FARM_ID])
        status, farm = _call(base, f"/api/farms/{mock.FARM_ID}")
        assert status == 200 and len(farm["camps"]) == len(mock.CAMPS)
        status, camp = _call(base, f"/api/farms/{mock.FARM_ID}/camps/c4")
        assert status == 200 and camp["history"][0]["status"] == "missing"
        status, alerts = _call(base, f"/api/farms/{mock.FARM_ID}/alerts")
        aid = next(a["alert_id"] for a in alerts if a["camp_id"] == "c4")
        status, done = _call(base, f"/api/alerts/{aid}/action", {"action": "records_updated"})
        assert status == 200 and done["action"] == "records_updated"
        assert _call(base, f"/api/alerts/{aid}/action", {"action": "nope"})[0] == 400
        assert _call(base, "/api/alerts/none/action", {"action": "dismissed"})[0] == 404
        assert _call(base, "/api/farms/none")[0] == 404
        status, page = _call(base, "/")
        assert status == 200 and "HerdSight" in page
        assert _call(base, "/../pyproject.toml")[0] == 404  # nothing outside the frontend folder
    finally:
        srv.shutdown()
        srv.server_close()

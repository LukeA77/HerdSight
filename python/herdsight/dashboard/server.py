"""Dashboard web server: a small JSON API over the store, plus the phone pages (Phase 6, D-17).

Usage: python -m herdsight.dashboard.server --db <file.sqlite> [--mock] [--port 8080]

Python's standard library only (no web framework to install or keep patched). API:
  GET  /api/farms                         farm ids
  GET  /api/farms/<farm>                  layout, each camp's newest check, status labels
  GET  /api/farms/<farm>/camps/<camp>     the camp's checks, newest first
  GET  /api/farms/<farm>/alerts           open alerts, newest first
  POST /api/alerts/<alert>/action         {"action": "second_check" | "records_updated" |
                                           "found_them" | "dismissed", "note": "..."}
Everything else is served from dashboard/frontend/. --mock fills an empty database with the
made-up farm from herdsight.dashboard.mock.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from herdsight.dashboard.store import ACTIONS, Store
from herdsight.herd.messages import glossary
from herdsight.utils.config import REPO_ROOT

FRONTEND = REPO_ROOT / "dashboard" / "frontend"
MAX_BODY = 4096


def farm_view(store: Store, farm_id: str) -> dict[str, Any] | None:
    lay = store.layout(farm_id)
    if lay is None:
        return None
    latest = store.latest(farm_id)
    g = glossary()
    camps = []
    for c in lay["camps"]:
        rec = latest.get(c["camp_id"])
        status = rec["status"] if rec else "unknown"
        camps.append(
            {
                **c,
                "status": status,
                "label": g["status"][status]["label"],
                "colour": g["status"][status]["colour"],
                "latest": rec,
            }
        )
    order = {"missing": 0, "moved": 1, "check": 2, "unknown": 3, "ok": 4}
    attention = sorted(
        (c for c in camps if c["status"] != "ok"), key=lambda c: order.get(c["status"], 5)
    )
    return {
        "farm_id": farm_id,
        "name": lay.get("name", farm_id),
        "camps": camps,
        "needs_attention": [c["camp_id"] for c in attention],
        "open_alerts": len(store.open_alerts(farm_id)),
        "simulated": any(c["latest"] and c["latest"]["simulated"] for c in camps),
    }


class Handler(BaseHTTPRequestHandler):
    store: Store  # set on the class by make_server

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass

    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        if parts[:1] == ["api"]:
            self._api_get(parts[1:])
        else:
            self._static(parts)

    def _api_get(self, p: list[str]) -> None:
        s = self.store
        if p == ["farms"]:
            return self._json(s.farms())
        if len(p) == 2 and p[0] == "farms":
            view = farm_view(s, p[1])
            return self._json(view) if view else self._error(HTTPStatus.NOT_FOUND, "no such farm")
        if len(p) == 3 and p[0] == "farms" and p[2] == "alerts":
            return self._json(s.open_alerts(p[1]))
        if len(p) == 4 and p[0] == "farms" and p[2] == "camps":
            hist = s.history(p[1], p[3])
            if not hist:
                return self._error(HTTPStatus.NOT_FOUND, "no checks for this camp")
            return self._json({"camp_id": p[3], "history": hist})
        self._error(HTTPStatus.NOT_FOUND, "unknown address")

    def do_POST(self) -> None:
        p = [unquote(x) for x in urlparse(self.path).path.strip("/").split("/") if x]
        if not (len(p) == 4 and p[:2] == ["api", "alerts"] and p[3] == "action"):
            return self._error(HTTPStatus.NOT_FOUND, "unknown address")
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            action, note = body.get("action"), str(body.get("note", ""))[:500]
        except (ValueError, AttributeError):
            return self._error(HTTPStatus.BAD_REQUEST, "send JSON with an action")
        if action not in ACTIONS:
            return self._error(HTTPStatus.BAD_REQUEST, f"action must be one of {list(ACTIONS)}")
        try:
            self._json(self.store.act(p[2], action, note))
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "no such alert")

    def _static(self, parts: list[str]) -> None:
        rel = "/".join(parts) or "index.html"
        path = (FRONTEND / rel).resolve()
        if FRONTEND.resolve() not in path.parents or not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind.endswith("javascript"):
            kind += "; charset=utf-8"
        self._send(HTTPStatus.OK, path.read_bytes(), kind)


def make_server(store: Store, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"store": store})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--mock", action="store_true", help="fill an empty database with made-up data")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)
    store = Store(args.db)
    if args.mock and not store.farms():
        from herdsight.dashboard.mock import fill

        fill(store)
    srv = make_server(store, args.host, args.port)
    print(f"dashboard on http://{args.host}:{srv.server_address[1]}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

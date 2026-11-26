# Dashboard backend

The backend code is the Python package `python/herdsight/dashboard/` (store, server, mock farm), so it shares the
project's tests and settings. Run it with:

```bash
python -m herdsight.dashboard.server --db dashboard.sqlite --mock      # made-up farm
python scripts/flights_to_dashboard.py --db dashboard.sqlite --run A=<flight folder>   # simulated flights
```

Then open `http://127.0.0.1:8080/` on the same computer.

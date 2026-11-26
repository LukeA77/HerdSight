# Phase 6: farmer dashboard (plan and record)

## Goal
A farmer with no technical background can tell within 10 seconds whether any camp needs attention, and can act on an
alert without help (S6). The dashboard reads only survey result records (`dashboard/schemas/survey_result.v1.json`)
and a farm layout; it never touches the models or the drone software (D-17).

## Built (25 November 2026)
- **Store** (`python/herdsight/dashboard/store.py`): SQLite; every record is checked against the schema before it is
  stored; the farmer's actions on alerts are kept with their time. Asking for a second check keeps an alert open;
  "I found them", "my records were wrong" and "hide" close it.
- **Server** (`python/herdsight/dashboard/server.py`): Python standard library only (no web framework to install or
  keep patched); a small JSON API and the phone pages; nothing outside `dashboard/frontend/` is served.
- **Mock farm** (`python/herdsight/dashboard/mock.py`): six camps, eight weekly checks each, every status on the farm
  map; built with the same record code as the pipeline; marked as made-up data.
- **Pages** (`dashboard/frontend/`): farm summary ("all camps look fine" or "N camps need attention"), a map coloured
  by status, a list with the camps that need attention first, a camp page (count, likely range, your records, the
  plain message, recent checks with their ranges) and an alerts page with large buttons. Plain JavaScript, no build
  step; a banner marks simulated or made-up data.
- **Weak signal**: the last answers are kept on the phone and shown with a notice when there is no signal; actions
  taken without signal are queued and sent later; the pages themselves are cached so the dashboard opens offline.
- **Pipeline connection** (`scripts/flights_to_dashboard.py`): finished Gazebo survey flights are counted at home base
  as for the S5 checks and appear as camps of a simulated farm. Mission to count to dashboard runs end to end in
  simulation.
- **Wording fix found with the mock farm**: a camp with more animals than the records was told its count "matches your
  records"; "matches" is now used only when the records lie inside the likely range (`ok_more` and `ok_fewer` added to
  the glossary).
- Tests: `tests/test_dashboard.py` (store, mock farm, actions, the web API end to end, no path escape).

## Still to do
- Usability test (S6) with five non-technical testers: test plan and tasks to be written; the sessions are run by the
  owner.
- Evidence images (crops of counted herds) on the camp page once the pipeline writes them.
- Checked on a desktop browser at phone width only; to be checked on real phones during the usability test.

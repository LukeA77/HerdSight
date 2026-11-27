# Phase 7: final test and write-up (plan and record)

## Goal
Score the final system once on data that was never used to build, tune or calibrate it, and report the result as it
is. Every test set is run once, after the settings it depends on are frozen; a second run is refused by the scripts
(`python/herdsight/utils/run_once.py`).

## Test sets
| Test set | What it measures | Frozen as | Script | Result |
|---|---|---|---|---|
| Held-back farm (ICAERUS Other_farms: 62 photos, 1,355 cattle, five flights) | Detection and per-photo counting (S1, S2) and onboard herd detection (S7) | `phase7` | `scripts/phase7_test.py` | `results/phase7/test_farm.json` (D-54) |
| 4000 simulated farms on the test seed | Alerts (S4) with the final detector's count error | `phase7_s4` | `scripts/run_s4.py --set test` | `results/phase4/s4_test.json` (D-56) |
| 200 simulated flights on the test seed | Counting a whole flight and its ranges in simulation | `phase7_s3` | `scripts/phase3_sim.py --detector dfine_ref --set test` | `results/phase3/sim_test_dfine_ref.json` (D-57) |

## Steps taken (26-27 November 2026)
1. **Final models.** The home-base detector (D-FINE S, reference recipe) and the onboard herd detector (D-FINE N on
   stills reduced to 1,920 px) were trained once on all three cross-validation folds (`gpu/runpod/job_final.sh`, one
   rented GPU session, `results/phase2/run_ledger.csv`). Their file hashes are recorded in `configs/phase7.yaml`, and the
   test refuses a model whose hash differs.
2. **Calibration from cross-validation only.** The operating threshold and per-photo count ranges come from the same
   recipes' out-of-fold predictions on the three development farms, never from the test farm.
3. **Rehearsal.** The whole test pipeline was run on three development photos to check that it works end to end before
   anything was frozen.
4. **Freeze, then one run.** `scripts/freeze.py --phase phase7`, then `scripts/phase7_test.py` once.
5. **Refits before the simulated tests.** The simulation's detection error model, the flight-count error used for
   alerts, and the real-flight counting chain (pairs, grouping, ranges) were refitted to the final detector's
   out-of-fold detections. The alert dial and the counting method were checked on validation farms and flights by rules
   written beforehand (`docs/hypotheses.md`, H29 and the Phase 3 note), then frozen and run once on the test seeds.
6. **After the runs.** Each result is recorded in `docs/decisions.md` (D-54, D-56, D-57), `docs/limitations.md` and the
   README, whose numbers come from `scripts/make_readme_figures.py`. The misses on the test farm are broken down from
   the stored predictions of the single run (`scripts/phase7_failures.py`); nothing is re-run or tuned.

## Results in one line each
- Held-back farm: 1 of 9 targets met. 69% of cattle found and 74% of marks correct (targets 80%); most misses are
  animals standing close together.
- Alerts on simulated test farms: 1.6 false alarms per 100 surveys and 46% of losses of five or more caught, against at
  most 1.5 and at least 65%.
- Whole-flight counting: 25% error on the 20 hand-checked real flights (target 15%); 28% on simulated test flights.

All are public-data or simulation results. None is a field result.

## Not tested here
- Onboard speed on a Raspberry Pi 5 (none available; D-29).
- The dashboard with non-technical users (S6): the sessions are planned in `docs/plans/s6-usability-test.md`.

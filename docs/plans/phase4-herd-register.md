# Phase 4 plan: herd register and alerts

Status: agreed with the owner (2026-11-21). Runs in parallel with Phase 2, because it does not depend on the detector: it works on simulated survey results whose error model is fitted to the measured count errors and updated when Phase 2 and 3 results arrive.

## Goal
Decide, for every camp after every survey, whether the farmer should be told that animals may be missing, so that:
- real losses are caught (as many as possible, as soon as possible);
- false alarms are rare (the owner's first priority, D-13);
- the farmer can choose the balance with one setting (the "dial"), knowing what each setting means.

Exit (brief §12): the S4 trade-off curve on validation scenarios.

## What S4 measures
- **False alarms per 100 surveys:** alerts raised when no animal was actually lost.
- **Share of real losses caught:** losses of at least k animals that raise an alert within a stated number of surveys, and how many surveys it took.
- Reported as a curve over the dial settings, with 95% intervals over simulated farms. The target is set by the owner after the baseline (as in D-31) and frozen before the alert rule is tuned.

## Approach

### 1. Herd register (Python, `herdsight.register`)
- One expected count per camp, changed only by recorded events: birth, purchase, sale, death, move between camps, and correction.
- Every event is kept (append-only log with who and when), so the expected count at any past time can be rebuilt and every alert explained.
- Moves are one event with two sides, so the farm total stays consistent.
- Empty `IdentitySource` interface kept (D-12).

### 2. Survey results the rule works on
Each survey of a camp gives a count, a calibrated interval (68/90/95%) and a flag for dense herds (from Phase 2's density cross-check). This is the same record the dashboard will read (D-17), so a first version of the **results schema** (`dashboard/schemas/survey_result.v1.json`, kept beside the dashboard that reads it) is written here and reused in Phase 6.

### 3. Simulator of surveys (Python, `herdsight.sim.surveys`)
A simulated farm: camps, herds, events and surveys over months. Known truth at every moment.
- **Farm and herd sizes:** a sweep from small enclosed fields to large extensive camps (owner answer Q4), with ranges taken from published UK and South African farm statistics (cited in the plan's Verification section).
- **Count error model:** how far a survey count is from the truth, fitted to measured results: per-flight errors after duplicate removal (`results/baselines/dfine_s3.json`, later Phase 3), with wider errors for dense herds and for partly covered camps. Intervals are produced by the same conformal method as the real pipeline, so they can be too narrow in the same ways. The error model is refitted whenever Phase 2 or 3 results change, and every S4 number states which version it used.
- **Losses injected with known time and size:** single deaths, theft of several animals at once, gradual straying, predation.
- **Tricky cases the brief names:** animals moved between camps without being recorded; register errors (a birth or sale not entered, a wrong number); packed herds (larger undercount); partial surveys (part of the camp not flown, or animals under trees).
- **Two scenario sets with different random seeds:** validation (for choosing the rule and the dial) and test (run once in Phase 7). Nothing is tuned on the test set.

### 4. Alert rules compared
1. **Naive baseline:** alert when the count is below the expected count.
2. **Interval rule (D-13):** alert only when the whole calibrated interval is below the expected count.
3. **Interval rule with recount:** a first shortfall triggers a recount (second survey or herd mode); an alert needs two independent surveys to agree, unless the shortfall is larger than a threshold (alert at once for a large loss).
4. **Farm-level check:** a shortfall in one camp with a matching surplus in a neighbouring camp is reported as "animals may have moved", not as a loss.

The **dial** is the interval level (for example 80% to 99%) and the large-loss threshold. The S4 curve is drawn by sweeping it.

### 5. Messages
Every alert carries a plain sentence from the glossary (D-17), for example "Some cows may be missing: 4 fewer than expected, checked twice", and the evidence (counts, intervals, photos when real). Technical terms never reach the farmer's text.

### 6. Code
Python reference with tests. The rule is small and runs at home base, so it stays in Python unless the onboard mission needs it (then a C++ version with a parity test, D-14).

## Hypotheses (written before the baseline is run)
- H13: the interval rule has at least 5 times fewer false alarms per 100 surveys than the naive rule, at the cost of catching no more than 10 percentage points fewer real losses of 5 or more animals.
- H14: requiring two surveys to agree removes at least half of the interval rule's remaining false alarms, and delays detection of losses of 5 or more animals by no more than one survey on average.
- H15: unrecorded moves between camps cause most false alarms that remain after the recount, and the farm-level check removes at least half of those.
Each is judged by a rule fixed in code before the run.

## Alternatives considered
- *Alert on any shortfall:* simple, but the naive baseline is expected to raise many false alarms, which the owner most wants to avoid; kept only as the baseline.
- *A learned model that predicts losses from count history:* needs real farm histories, which do not exist yet; it would only learn the simulator. Rejected for Stage A.
- *Bayesian tracking of each camp's true count over time (for example a Kalman-type filter):* a strong option, and could beat single-survey rules by using history. Included as a stretch candidate only if rules 1–4 leave a clear gap, because it is harder for a farmer to understand why an alert fired.

## How success is measured
- The S4 curve (false alarms per 100 surveys against share of losses caught, and surveys until caught) on validation scenarios, for each rule, with 95% intervals over simulated farms.
- Results also broken down by farm size, herd density, loss type and each tricky case.
- Every number states the count error model version it used.

## Risks
- **The simulator decides the result.** If the error model is too kind, alerts look better than they will be. Mitigation: the error model comes from measured results, intervals are calibrated the same way as the real pipeline, the tricky cases are included on purpose, and results are labelled simulation results, never field results.
- **Count errors may be correlated** (the same camp surveyed the same way twice gives similar errors), which makes "two surveys agree" less independent than it looks. The simulator includes a correlation setting, and results are shown with it on.
- **Farm statistics** vary widely between countries and farm types; the sweep covers the range rather than one typical farm.

## What the owner will be asked to do
1. After the baseline: set the S4 target (as for the Phase 1 targets), with a reason.

The owner asked for the farm assumptions (loss types and rates, how often moves go unrecorded, farm and herd sizes) to be chosen from published sources (docs/questions.md Q11). Each value is recorded in `configs/sim_farms.yaml` with its source, and results are also shown across a range around each value, so no single assumption decides the outcome.

## Review
Self-review against brief §3.2 (Phase 4 is not one of the phases that needs an independent reviewer):
- *Easier metric?* No: S4 is the brief's curve, with surveys-until-caught added.
- *Testing on related data?* Validation and test scenarios use different seeds and are fixed before tuning; the test set is run once in Phase 7.
- *Mocked or hard-coded?* The simulator stands in for farms by design (Stage A has no farms); this is stated with every result. The count errors are fitted, not invented.
- *Narrowed scope?* All four tricky cases from the brief are included, plus correlated errors.
- *Single numbers?* Every point on the curve has an interval over simulated farms.
- *Untested assumptions?* Recount independence is tested with the correlation setting rather than assumed.
- *Technical words to farmers?* All messages come from the glossary.

Answers to the brief's questions:
1. *Best way regardless of effort:* fit the error model and loss patterns on real survey series from real farms, with real recorded losses. Not possible in Stage A; the plan fits what it can to measured public-data results and states the rest as a limitation.
2. *Most likely failure on a real farm:* unrecorded moves and register mistakes producing false alarms, and correlated survey errors making recounts agree when both are wrong. Both are simulated on purpose.
3. *Sceptical attack:* "the simulator is tuned to make the rule look good". Answered by fitting errors to measured results, fixing the scenarios before tuning, and keeping a held-out test set.

## Changes after review
- Added correlated survey errors and a correlation setting (risk found in the review).
- Added surveys-until-caught to S4, because a loss caught three weeks late is worth less.
- Added the farm-level check as its own rule, so its effect is measured separately.

## Verification
*(Filled in as each step completes.)*

- **S4 baseline** (`scripts/run_s4.py`, `results/phase4/s4_validation.json`; 400 simulated validation farms; simulation results, not field results). With the count error measured in Phase 1:
  - naive rule: 19.68 false alarms per 100 surveys (18.90–20.45), 86% of losses of 5 or more caught (70%–100%), 72% of all losses;
  - interval rule at 90%: 3.75 false alarms per 100 surveys (3.52–4.01), 50% of losses of 5 or more caught (28%–73%), 26% of all losses;
  - recount rule at 90%: 1.51 false alarms per 100 surveys (1.34–1.68), 45% of losses of 5 or more caught (23%–70%), 17% of all losses;
  - recount rule at 95%: 0.67 false alarms per 100 surveys (0.59–0.75), 32% of losses of 5 or more caught (12%–55%), 9% of all losses;
  - farm-level rule at 90%: 1.44 false alarms per 100 surveys (1.28–1.61), 45% of losses of 5 or more caught (25%–69%), 17% of all losses.
  With the S3 target's count error (15%), the recount rule at 90%: 1.22 false alarms per 100 surveys (1.10–1.35), 64% of losses of 5 or more caught (43%–83%), 18% of all losses.
  Only 22 losses of 5 or more animals occurred across the validation farms, so the share caught for them has wide intervals. Single deaths are rarely caught at any setting: a one-animal change is far inside the count error of a whole camp.
- **Larger run** (D-37; `scripts/run_s4.py --tag 4000farms`, `results/phase4/s4_validation_4000farms.json`; 4000 simulated validation farms, of which the first 400 are the baseline farms; 212 losses of 5 or more animals). With the count error measured in Phase 1:
  - naive rule: 18.92 false alarms per 100 surveys (18.65-19.19), 92% of losses of 5 or more caught (87%-96%);
  - interval rule at 90%: 3.62 (3.55-3.69), 66% (60%-72%);
  - recount rule at 90%: 1.45 (1.40-1.49), 58% (51%-64%);
  - recount rule at 95%: 0.61 (0.59-0.64), 41% (34%-47%);
  - farm-level rule at 90%: 1.38 (1.34-1.43), 58% (51%-64%).
  With the S3 target's count error, the recount rule at 90%: 1.17 (1.14-1.20), 67% (60%-73%). With correlated surveys of a camp (correlation 0.5) and the Phase 1 count error: 1.95 (1.89-2.00), 54% (47%-61%). The 400-farm figures for losses of 5 or more were low by chance; H13-H15 come out the same on 4000 farms.
- **S4 target set** (D-37, `configs/specs.yaml`): recount rule at 90%, at most 1.5 false alarms per 100 surveys and at least 65% of losses of 5 or more caught within 3 surveys.
- **Hypotheses** (`scripts/judge_s4.py`, `results/phase4/hypotheses.json`): H13 not held, H14 held, H15 not held.
- **Results schema and messages** (steps 2 and 5): `dashboard/schemas/survey_result.v1.json` (one record per drone check of a camp: count, likely ranges, number in the records, status, alerts, notes, evidence, and a `simulated` flag so simulator output can never pass as field results); `dashboard/glossary.yaml` (every farmer-facing word and sentence, ready for translation, with a list of technical words that must never appear); `python/herdsight/herd/messages.py` builds the records. Alerts now record how many checks agreed and, for "may have moved", the other camp. Empty `IdentitySource` interface for later tags (D-12): `python/herdsight/herd/identity.py`.
- **Tests:** `tests/test_messages.py` (schema valid; a record for each status validates; every glossary sentence is free of technical words; every survey of a simulated farm gives a valid, plain record with whole animal numbers).
- **Tests:** `tests/test_herd.py` (register arithmetic and moves, fast expected-count table against the register, interval coverage at its nominal level, rule and scoring cases with known answers). A first version of the simulator's intervals scaled with the square root of the count and covered 83% at a nominal 68%, because flight-total errors grow with the herd; intervals now scale with the count and cover at their level.

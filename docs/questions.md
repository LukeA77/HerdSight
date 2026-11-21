# Open questions

Questions that need the project owner's decision. Answers are recorded here with the date.

| # | Question | Answer | Date |
|---|---|---|---|
| Q1 | Project name | **HerdSight** | 2026-11-18 |
| Q2 | Is a Raspberry Pi 5 available for S7 timing? What GPU is available for training? | Training on free cloud GPUs: Kaggle first, Google Colab as fallback (see D-18). No local NVIDIA GPU. Raspberry Pi 5 availability: not yet known | 2026-11-18 |
| Q3 | Target country and regulations | United Kingdom and South Africa. Mission planning must satisfy both countries' rules; the stricter limit applies unless a per-country profile is selected (see D-19) | 2026-11-18 |
| Q4 | Typical camp sizes and herd sizes to simulate | Must work across the full range, from small enclosed fields to large extensive camps. Simulation covers a sweep of camp and herd sizes rather than one typical value; the range will be set from published farm statistics for both countries | 2026-11-18 |
| Q5 | Breeds of interest | Cattle first, all common breeds; no single breed targeted. Sheep later | 2026-11-18 |
| Q6 | Who can run the S6 usability tests, and how many people? | At least 5 testers: one with farming knowledge and at least four without, including a graphic designer and a business person. S6 sample size set to n ≥ 5 | 2026-11-18 |
| Q7 | Licence for this repository | All rights reserved (see `LICENSE`) | 2026-11-18 |
| Q8 | Is a Raspberry Pi 5 available, or can one be borrowed, for timing the onboard herd detector (S7)? | No. S7 latency is reported as an estimate, never as a measurement (D-29) | 2026-11-19 |
| Q9 | Dataset licence policy (D-21): train only on commercially usable data, use non-commercial data for evaluation only, and skip unlicensed data until permission is given? | Yes (D-21 adopted) | 2026-11-19 |
| Q10 | Brazilian duplicate-removal dataset (Soares et al., 2021): its download links need checking in a browser, and it has no licence. Ask the authors for permission? | No. The dataset is not used; S3 is measured on ICAERUS flights with owner-verified links | 2026-11-19 |
| Q11 | Phase 4 farm assumptions: how losses happen (theft, death, straying, predation), how often, how often animals are moved between camps without being recorded, and typical farm and herd sizes | Owner has no farm data and asked for these to be set from published sources, each recorded with its source in `configs/sim_farms.yaml`, with results shown across a range around each value | 2026-11-21 |

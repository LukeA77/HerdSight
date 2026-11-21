# Limitations

What the system cannot do, and what has not been tested. Kept up to date as work progresses.

- The system has not been flown on a farm. All results come from public datasets and simulation.
- Detection and counting are trained and measured on one public dataset (ICAERUS, France: three farms for cross-validation, one held out for the final test). How well they transfer to UK and South African farms, breeds and pastures is not measured.
- Duplicate removal across images (S3) is measured on ICAERUS flights whose animal links were checked by hand. Ten flights are verified, at least one from each farm-date group, all checked by one person, and there is no published external benchmark to compare against.
- No Raspberry Pi 5 was available, so onboard herd-detection speed (S7) is an estimate, not a measurement.
- ICAERUS has no calf labels, so the pixels needed for calves are inferred from adult cattle and published calf sizes, not measured on calves.
- Some ICAERUS images labelled as empty contain cattle (at least two found). Scores use the published labels unchanged (D-32), so precision and false-detection rates are slightly understated.
- Derval has only one survey day, so in cross-validation it is scored by a model that never saw that farm at all. It is where the Phase 1 detector does worst (`results/baselines/dfine_failures.json`). This is the closest evidence available of how the detector may do on new farms, breeds and light.

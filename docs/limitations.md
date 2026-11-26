# Limitations

What the system cannot do, and what has not been tested. Kept up to date as work progresses.

- The system has not been flown on a farm. All results come from public datasets and simulation.
- Detection and counting are trained and measured on one public dataset (ICAERUS, France: three farms for cross-validation, one held out for the final test). How well they transfer to farms, breeds and pastures elsewhere is not measured.
- Duplicate removal across images (S3) is measured on ICAERUS flights whose animal links were checked by hand. Ten flights are verified, at least one from each farm-date group, all checked by one person, and there is no published external benchmark to compare against.
- No Raspberry Pi 5 was available, so onboard herd-detection speed (S7) is an estimate, not a measurement.
- ICAERUS has no calf labels, so the pixels needed for calves are inferred from adult cattle and published calf sizes, not measured on calves.
- Some ICAERUS images labelled as empty contain cattle (at least two found). Scores use the published labels unchanged (D-32), so precision and false-detection rates are slightly understated.
- Derval has only one survey day, so in cross-validation it is scored by a model that never saw that farm at all. It is where the Phase 1 detector does worst (`results/baselines/dfine_failures.json`). This is the closest evidence available of how the detector may do on new farms, breeds and light.
- The herd-register alerts (S4) are measured only on simulated farms. The farm assumptions come from published sources where they exist (`configs/sim_farms.yaml`), and results are reported across the range of each one, but how much larger the count error is for closely packed herds (1.5 times, from 20 animals in a camp) is assumed, not measured, and it drives most of the remaining false alarms.
- A single dead animal is rarely detected by a drone count of a whole camp: about 1 in 5 single deaths is caught within 3 checks in simulation, mostly by chance. Alerts are designed for losses of several animals.
- How herds spread across a whole camp (number of groups, distances between them, preference for water and shade) is not in the public data, which was photographed where cattle were. The simulator sweeps it (`configs/sim_worlds.yaml`).
- Simulated detections of calves and of animals partly hidden by trees are assumptions: no labels of calves or hidden animals exist, so their detection is extrapolated from adults and swept.
- The flying height uses a provisional pixel budget for calves (the smallest labelled adults), until the Phase 2 resolution study measures it.
- On a farm it has never seen, the detector finds 55% of cattle, against 73% on farms it was trained on (leave-one-farm-out, `results/phase2/g6_lofo.json`). Two training changes aimed at this did not help (D-49, D-50). The working approach is to train on photos of each farm before use; how many photos that takes is being measured (G13), and it will still need checking on real farms.
- The onboard herd detector does not yet meet its targets (S7): at the onboard image size it finds 94% of herds but groups only 75% correctly, against targets of 95% and 85% (D-51). A larger onboard image is being tested (G10b); onboard speed is still an estimate.
- Returning home with the camera after GPS loss works in simulation in calm air and in 5 m/s wind, not yet in 10 m/s: the camera is fixed to the body, and when the drone leans hard into strong wind it sees the ground at a slant and matching fails. It assumes a downward laser rangefinder (without GPS, the barometer's height drifted several metres), and fine texture on the ground near the landing point. A stabilised camera is the likely fix; none of this has been flown.

# Data

Every dataset is checked before use: is it available, what licence does it carry, what format are its labels, and how big is it. Datasets are not stored in this repository. They are downloaded to the storage folder (`$HERDSIGHT_STORAGE/datasets/raw/<name>/`) and verified against their published checksums.

**Licence policy (D-21).** HerdSight is all rights reserved and may become a product. So data is sorted by what its licence allows:

- **Commercial use allowed** (for example CC BY 4.0): may be used for training and evaluation.
- **Non-commercial only** (for example CC BY-NC): evaluation only, clearly marked, and never used to train a model that could ship.
- **No licence stated:** not used until the authors grant permission in writing. With no licence, all rights stay with the authors.

## Audit (2026-11-18)

### Aerial cattle, RGB

| Dataset | Status | Licence | Size | Labels | Notes |
|---|---|---|---|---|---|
| **ICAERUS grazing cows v2**: Institut de l'Elevage, *Drone images and their annotations of grazing cows*, Zenodo [10.5281/zenodo.11048412](https://doi.org/10.5281/zenodo.11048412), April 2024 | Available; downloaded, checksum verified, inventoried (below) | CC BY 4.0: training and evaluation | 16.6 GB, one archive | 4,941 upright boxes on 1,385 images; PASCAL VOC and YOLO | France; DJI Mavic 3E / 3T; nadir; 60 m or 100 m above take-off; 4000×3000 or 5280×3956 px; June 2023 to March 2024. Folders by farm (Mauron, Derval, Jalogny, Other_farms) and by flight; `conditions_summary.xlsx` gives altitude, weather, drone and camera per flight; EXIF metadata per image. Planned flights cover whole pastures, so overlapping, geotagged sequences may support duplicate-removal tests (S3); to be confirmed from the EXIF |
| ICAERUS grazing cows v1: Zenodo [10.5281/zenodo.10245396](https://doi.org/10.5281/zenodo.10245396), December 2023 | Available | CC BY 4.0 | 13.2 GB | Boxes, about 1,100 images | Superseded by v2; not used |
| ICAERUS raw cattle images: Zenodo [10.5281/zenodo.8234156](https://doi.org/10.5281/zenodo.8234156), 2023 | Available | CC BY 4.0 | 9.2 GB | None | About 900 unlabelled images; possible source of hard negatives or extra labels later. Overlap with v2 not yet checked |
| **Cattle counting in the wild** (Brazil): Soares, Ponti, Gonçalves and Campello, *Computers and Electronics in Agriculture* 189, 106354, 2021, [10.1016/j.compag.2021.106354](https://doi.org/10.1016/j.compag.2021.106354) | **Unconfirmed.** The authors' page (vhasoares.github.io/downloads.html) links two Google Drive folders (training set; BR_set test collection). Both returned "not found" to automated requests, which Google Drive often does, so they must be checked in a browser | **None stated:** cannot be used until the authors give permission | Not known | Boxes (LabelImg); 5,058 images reported by a 2024 survey | Brazil; six farms; geolocated, overlapping images at 80–120 m; benchmark for duplicate removal across images. The brief's key dataset for S3 |
| Shao et al. (Japan): *International Journal of Remote Sensing* 41(1), 31–52, 2020, [10.1080/01431161.2019.1624858](https://doi.org/10.1080/01431161.2019.1624858) | Mirror available (Dataset Ninja); original site (bird.nae-lab.org/cattle) did not respond | CC BY-NC-ND 4.0 according to the mirror (not confirmed at source): **evaluation only**. "No derivatives" also rules out publishing converted labels | 3.3 GB | 1,948 upright boxes on 670 images (340 unlabelled); cattle IDs; quality flags | Aso, Kumamoto; about 50 m; 4000×3000; Dataset 1 has 7 flights on one day; Dataset 2 has 14 images from another area and season. Candidate unseen-source test set |
| Aerial-livestock-dataset: Han et al., 2019 | **Not available.** The GitHub repository has no releases and no licence | None | — | — | Not usable |
| AerialCattle2017: University of Bristol | Available | Non-Commercial Government Licence | 0.7 GB | Crops of tracked individuals | Crops for identity work, not full survey frames; reserved for the later identification track |
| WAID: Mou et al., *Applied Sciences* 13(18), 2023 | Repository available | **None stated** in the authors' repository (a third-party mirror claims CC BY 4.0) | — | Boxes; 6 species including cattle and sheep | Not usable until the licence is confirmed; image provenance unclear |
| arXiv 2110.01329 (resolution study) | Combined dataset not released | — | — | — | Its method (downsampling to set ground sample distances) is reused in the Phase 2 resolution study; its data is not available |
| Porto et al. (feedlots) | Not located | — | — | 90 images reported | Not pursued: feedlot scenes differ from pasture surveys |

### Thermal

| Dataset | Status | Licence | Size | Labels | Notes |
|---|---|---|---|---|---|
| COW_MOTS: Bárbulo Barrios, Valente and van Langevelde, Zenodo [10.5281/zenodo.6370315](https://doi.org/10.5281/zenodo.6370315), 2022 | Available | CC BY 4.0 | 1.7 GB | Instance masks and tracks: 959 frames, 20,647 masks, 239 tracks | Netherlands; Parrot ANAFI Thermal; 8–28 m, oblique and nadir; daytime at 10–26.5 °C. Only thermal cattle data found; far lower than survey height, so useful for thermal appearance, not survey-height accuracy |
| BIRDSAI (Conservation Drones, LILA) | Available; already on the storage drive (from drone-ranger) | Community Data License Agreement (permissive) | 43 GB (3.7 GB real) | Boxes and tracks, MOT CSV | Southern Africa, night, thermal; wildlife and people, **no cattle**. Useful for thermal hard negatives and night conditions |

### For later (sheep)

| Dataset | Notes |
|---|---|
| AUTH-Sheep (Zenodo [10.5281/zenodo.13843376](https://doi.org/10.5281/zenodo.13843376)) | Sheep videos with **oriented** boxes and track IDs; licence to check |
| ICAERUS sheep and goat records (Zenodo 10400302, 18889354, 14591324) | Licences to check |
| UAV footage of Merino sheep in South Africa (Mendeley Data, nppwbkdf85) | Page did not load; to check |

### Searches for newer data (2024–2026)

Searches of Zenodo, survey papers ([Public Computer Vision Datasets for Precision Livestock Farming](https://arxiv.org/abs/2406.10628), 2024) and the general web found no newer public aerial cattle dataset at survey height with a clear licence. **No public aerial cattle dataset with oriented boxes was found**, which confirms the plan to generate oriented labels (brief §8.4).

## What this means

1. **ICAERUS v2 is the main training and evaluation source.** It is the only large aerial cattle set at survey height whose licence allows a product. Its four farm groups allow a whole farm to be held out as the unseen test set (brief §8.2).
2. **Duplicate removal (S3) depends on the EXIF check.** If ICAERUS flights overlap and carry usable GPS, they can be the S3 benchmark (hand-linked identities, brief §8.3), with the Brazilian set as the published comparison if its authors agree.
3. **Shao et al. is a useful second source for testing only**, because of its licence.
4. **Thermal cattle data is thin.** Only COW_MOTS exists, at low height. Stage A's thermal work is therefore limited, and this is stated as a limitation.

## Downloads

| Dataset | Path under `$HERDSIGHT_STORAGE` | Checksum | Verified |
|---|---|---|---|
| ICAERUS grazing cows v2 | `datasets/raw/icaerus_grazing_cows_v2/Cattle_drone_images_042024.zip`, unpacked to `…/unpacked/` | MD5 `c18911fb11cb58741b9cfa16516ca8cd` (Zenodo) | yes, 2026-11-18 |

## ICAERUS v2: what the archive actually contains (inventory 2026-11-18)

Numbers in this section come from `results/data/icaerus_inventory.json` (`scripts/inspect_icaerus.py`). Some differ from the dataset's own readme.

- **Images:** 1,385 RGB nadir stills in 25 flights. Mavic 3E (5280×3956) and Mavic 3T (4000×3000). Every image has GPS, relative altitude and gimbal angles in its metadata; the gimbal points straight down in every image. Heights are 60 m or about 100 m above the take-off point. There are no thermal images.
- **Labels:** the YOLO labels cover every image. The PASCAL VOC labels cover only 1,079 images, and none from Other_farms; where both exist they agree to within 1 pixel. The loader therefore reads the YOLO labels.
- **Boxes:** 4,747 cow boxes match an image. The readme says 4,941. Two label files in Other_farms (157 boxes) have no matching image.
- **No calves.** Some flights define a "calf" class, but no box uses it. Calf sizes for the resolution study (D-6) therefore cannot come from these labels; this is a gap for Phase 2.
- **Mismatched folder names:** three label folders carry a different flight name from their image folder. Labels are matched by farm and file name instead (image names hold a capture timestamp).
- **Animal size:** box long side median 80 px, 99.9th percentile 163 px, maximum 181 px; short side 1st percentile 17 px. This sets the 192 px tile overlap.
- **Overlap between consecutive images:** median 32% to 87% per flight, estimated from GPS and the camera footprint. Overlapping geotagged sequences exist, so ICAERUS can serve as the duplicate-removal benchmark (S3) once animal identities are linked by hand on a few flights.
- **Empty images:** many flights cover whole pastures, so most of their images are empty (for example 379 of 416 in one Mauron flight). Other_farms has no empty images.
- **Density:** up to 90 animals in one image (Other_farms); up to 53 in the other farms.

## Splits (D-25, D-26)

`results/splits.json`, frozen before any model was run (`results/frozen/phase1_splits.json`):

- **Test:** Other_farms (5 flights, 62 images, 1,355 boxes). It is the farm closest to a 20% share of boxes (28.5%). Untouched until Phase 7. Because it has no empty images and the densest herds, it tests counting in dense herds harder than empty-pasture false alarms.
- **Cross-validation folds:** the other 20 flights form 10 farm-date groups, dealt into 3 folds balanced by boxes (1,497 / 925 / 970 boxes).

## How well do geotags alone place an animal? (2026-11-18)

`results/data/geolocation_check.json` (`scripts/check_geolocation.py`, cross-validation folds only). Label centres in consecutive images were projected to the ground from GPS, relative altitude, focal length and gimbal yaw, then paired one-to-one within 15 m:

- **Camera orientation, measured:** the top of the image faces the gimbal yaw, not mirrored. Median paired distance 1.52 m, against 1.95 m with the flight yaw, 2.02 m mirrored, and 10.6 m reversed.
- **Accuracy:** median 1.5 m, 90th percentile 6.4 m, 61% of pairs within 2 m (2,176 pairs). This combines GPS, height (relative to take-off, not to the ground below), yaw and animal movement over a few seconds.
- **Consequence:** cattle in a packed herd stand about 2 to 3 m apart, so geotags alone cannot tell neighbours apart. Removing duplicates across images (D-9) must refine the relative position of overlapping images from the images themselves (registration), with the geotag as the starting point and as a gate (D-27).

## Spacing of cattle (2026-11-18)

`results/data/herd_stats.json` (`scripts/herd_stats.py`, cross-validation folds, 304 images with animals):

- **Nearest neighbour on the ground:** median 3.3 m; a quarter of animals have a neighbour within 1.8 m; 95% within 11.5 m. Many neighbours are closer than the geotag error (median 1.5 m), which confirms D-27.
- **Groups by single linkage** (animals closer than d belong together): at 10 m, 87% of animals are in groups of 3 or more, with a median group size of 2 and a largest group of 52. Groups stop at image edges, so large herds are undercounted in size.

## Linking animals across overlapping images (2026-11-18)

`results/data/s3_candidate_links.json` (`scripts/link_animals.py`): 20 cross-validation flights; overlapping pairs found from geotag footprints:

- **Registration (D-27) succeeded on most pairs.** After registration, linked animals sit a median 0.2 to 0.6 m apart per flight (90th percentile 0.5 to 1.0 m), against 1.5 m from geotags alone.
- **Chaining (D-30):** plain connected components merged neighbours (for example 2 "animals" from 142 labels in one flight). Constrained grouping gives counts that are never below the largest single-image count.
- **Candidate unique animals are not yet ground truth.** Three flights, one per fold (rule in `scripts/make_link_review.py`), go to the owner for review: 145 "same animal?" checks and 89 "missed link?" checks.

# Training sessions on a rented GPU (RunPod)

Phase 2 trains on rented GPUs (decision D-35). Each session is self-contained: nothing is kept on RunPod between sessions.

## How a session runs
1. A pod (one GPU) is created on RunPod Community Cloud with:
   - image: an official RunPod PyTorch image;
   - environment: `HS_JOB`, `HS_TAG`, `HS_COMMIT`, `HS_PRICE_PER_HR`, plus two RunPod secrets passed as `GITHUB_TOKEN={{ RUNPOD_SECRET_github_token }}` and `KAGGLE_API_TOKEN={{ RUNPOD_SECRET_kaggle_api_token }}`;
   - start command:
     ```
     bash -c 'git clone -q https://x-access-token:$GITHUB_TOKEN@github.com/LukeA77/HerdSight.git /workspace/HerdSight && git -C /workspace/HerdSight checkout -q $HS_COMMIT && timeout <seconds> bash /workspace/HerdSight/gpu/runpod/session.sh; runpodctl pod stop $RUNPOD_POD_ID || runpodctl stop pod $RUNPOD_POD_ID; sleep infinity'
     ```
2. `session.sh` runs `job_<HS_JOB>.sh`, which writes its outputs to `/workspace/out`.
3. The outputs are uploaded to a new **private** Kaggle dataset `herdsight-rp-<HS_TAG>`.
4. When the upload is confirmed the pod deletes itself. If it cannot be confirmed, the pod only stops: GPU billing ends and the outputs stay on the pod's disk.

The start command ends with `runpodctl pod stop` and `sleep infinity` so that a failure anywhere never lets the container exit and restart, which would run the job again. `timeout` caps the job's length, so a stuck job cannot keep billing.

## Jobs
| Job | What it does | Expected time |
|---|---|---|
| `smoke` | Machine report; downloads the training tiles from Kaggle; times the Zenodo download; trains D-FINE S on fold 0 for 3 epochs with the Phase 1 recipe and exports it; writes `benchmark.json` with the speed relative to the Phase 1 Kaggle T4 run | about 20 minutes |
| `g0` | Phase 2 step G0 (H9): D-FINE S on fold 0's training data minus the hold-out group in `configs/phase2.yaml`, trained for 72 and for 220 epochs with the same recipe and seed; both scored on the hold-out tiles (`gpu/eval_tiles.py`); writes `g0_summary.json` | about 5 hours on an RTX 4090 or 5080 |

Training code: `gpu/train_dfine.py` (Phase 2; `gpu/train_dfine_baselines_kaggle.py` stays as the Phase 1 record), with extra augmentations in `gpu/dfine_ops.py`. Shared steps: `gpu/runpod/lib.sh`.

## Costs
Each session records its GPU, hourly price and duration in `session.json`. These rows are collected into `results/phase2/run_ledger.csv`.

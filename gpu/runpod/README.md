# Training sessions on a rented GPU (RunPod)

Phase 2 trains on rented GPUs (decision D-35). Each session is self-contained: nothing is kept on RunPod between sessions.

## How a session runs
1. A pod (one GPU) is created on RunPod Community Cloud with:
   - image: an official RunPod PyTorch image;
   - environment: `HS_JOB`, `HS_TAG`, `HS_COMMIT`, `HS_PRICE_PER_HR`, plus two RunPod secrets passed as `GITHUB_TOKEN={{ RUNPOD_SECRET_github_token }}` and `KAGGLE_API_TOKEN={{ RUNPOD_SECRET_kaggle_api_token }}`;
   - start command:
     ```
     bash -c 'git clone -q https://x-access-token:$GITHUB_TOKEN@github.com/LukeA77/HerdSight.git /workspace/HerdSight && git -C /workspace/HerdSight checkout -q $HS_COMMIT && bash /workspace/HerdSight/gpu/runpod/session.sh; runpodctl pod stop $RUNPOD_POD_ID; sleep infinity'
     ```
2. `session.sh` runs `job_<HS_JOB>.sh`, which writes its outputs to `/workspace/out`.
3. The outputs are uploaded to a new **private** Kaggle dataset `herdsight-rp-<HS_TAG>`.
4. When the upload is confirmed the pod deletes itself. If it cannot be confirmed, the pod only stops: GPU billing ends and the outputs stay on the pod's disk.

The start command ends with `runpodctl pod stop` and `sleep infinity` so that a failure anywhere never lets the container exit and restart, which would run the job again.

## Jobs
| Job | What it does | Expected time |
|---|---|---|
| `smoke` | Machine report; downloads the training tiles from Kaggle; times the Zenodo download; trains D-FINE S on fold 0 for 3 epochs with the Phase 1 recipe and exports it; writes `benchmark.json` with the speed relative to the Phase 1 Kaggle T4 run | about 20 minutes |

## Costs
Each session records its GPU, hourly price and duration in `session.json`. These rows are collected into `results/phase2/run_ledger.csv`.

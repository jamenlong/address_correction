# address_correction

Address malformation, T5 correction, and ranking pipelines for Databricks.

## Databricks Repos: Git pull fails (working tree > 1 GB)

Databricks limits the **Repos working directory** to about **1 GB** for Git operations. Hugging Face `Trainer` checkpoints (`optimizer.pt`, etc.) under `product_corrector/checkpoint-*` are often **400+ MB each** and are **not** meant to live in Git.

### Fix once (on the cluster / Repos copy)

1. In the Repos folder, **delete** the local training tree (it is not required for Git; models should be in MLflow / DBFS):
   - `product_corrector/` (entire folder)
   - `logs/` under the repo if present
2. **Git pull** again.

From a notebook on the same repo path you can run:

```python
import shutil
from pathlib import Path

repo = Path(".").resolve()  # or your repo root
for name in ("product_corrector", "logs"):
    p = repo / name
    if p.is_dir():
        shutil.rmtree(p)
        print("Removed", p)
```

Or remove the folders via the Databricks workspace file UI under your Repo.

### Prevent recurrence

The correction notebook writes Trainer output to **`/dbfs/FileStore/address_correction_training/`** instead of `./product_corrector`. After pulling the latest code, new training runs will not refill the Repo tree.

`.gitignore` already excludes `product_corrector/`, `*.pt`, and `checkpoint-*`; ignored files still count toward the Repos size limit if they exist on disk under the repo path—hence DBFS for checkpoints.

## Recovery vs T5 benchmark (Phase 1)

Compare **malform recovery + hybrid override** to a saved T5 eval table without retraining or re-running `model.generate`.

| Module | Role |
|--------|------|
| `address_malform_recovery.py` | Undo malform steps (start with `replace_char`), catalog JW match |
| `address_recovery_benchmark.py` | Enrich baseline `model_output.*_output` rows with hybrid metrics |
| `Run_Recovery_Benchmark.py` | Databricks notebook: widgets, Delta output, wins/regressions |

**Default baseline:** `model_output.t5_product_corrector_training_20may2026_16_07_06_output`  
**Default output:** `model_output.t5_product_corrector_training_20may2026_16_07_06_recovery_eval`

1. Open `Run_Recovery_Benchmark.py` on a cluster.
2. Set `run_phase` to `smoke_replace_char`, run all cells.
3. Then `full_replace_char` for the full single-step `replace_char` slice.
4. Use SQL summaries and the wins/regressions displays to decide whether to tune thresholds or add the next malform step.

Local tests: `python3 -m unittest tests.test_address_malform_recovery tests.test_address_recovery_benchmark`

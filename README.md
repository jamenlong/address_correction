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

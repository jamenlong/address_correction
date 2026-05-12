# Databricks notebook source
# MAGIC %md
# MAGIC # Address correction ranking (hard negatives)
# MAGIC
# MAGIC Runs [`address_correction_ranking_pipeline.py`](./address_correction_ranking_pipeline.py) on the cluster.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Cluster: **ML** runtime (GPU recommended for training; CPU works for small smoke tests).
# MAGIC - Data: Delta table from your malformation pipeline (e.g. `smarty.smarty_malformed_1000_training`) **or** a parquet export with the same columns.
# MAGIC - Optional: MLflow model URI for warm-start (`runs:/.../model`).
# MAGIC
# MAGIC **Repos:** keep this notebook in the **same folder** as `address_correction_ranking_pipeline.py` (repo root for this project).
# MAGIC
# MAGIC **Artifacts:** set the **`artifact_dir`** widget to **`/dbfs/FileStore/...`** so large outputs (HF `Trainer` checkpoints under `hf_trainer_checkpoints/`, tokenized data, `trained_model`) stay **outside** the Git-linked Repo tree (Databricks Repos ~1GB working-directory limit).

# COMMAND ----------

# Do NOT `pip install mlflow` here with a loose lower bound (e.g. mlflow>=2.10).
# That upgrades mlflow-skinny to 3.x and often pulls protobuf 6+, which breaks
# packages Databricks pins on ML runtimes (databricks-feature-engineering wants
# mlflow-skinny<3; TensorFlow/tensorboard want protobuf<5). Use the runtime's MLflow.
#
# If pip still warns about transitive deps, the next cell restarts Python so imports see the new wheels.
# MAGIC %pip install -q "datasets>=2.14" "accelerate>=0.26" "transformers>=4.36" "scikit-learn>=1.3" "tqdm"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC **After restart:** the kernel is fresh. Use **Run All** again from the top, or run from the **widgets** cell downward so `dbutils` and your imports run after pip + restart.

# COMMAND ----------

dbutils.widgets.dropdown("data_source", "delta_table", ["delta_table", "parquet"], "Data source")
dbutils.widgets.text("source_table", "smarty.smarty_malformed_1000_training", "Delta table (if delta_table)")
dbutils.widgets.text("source_parquet", "", "Parquet path /dbfs/... (if parquet)")
dbutils.widgets.text("artifact_dir", "/dbfs/FileStore/address_correction_ranking", "DBFS base path; HF Trainer uses subfolder hf_trainer_checkpoints/")
dbutils.widgets.text("hub_model_id", "t5-small", "Hub model id when not using MLflow checkpoint")
dbutils.widgets.text("model_uri", "", "MLflow model URI, e.g. runs:/<run_id>/model (leave empty for Hub only)")
dbutils.widgets.text("malform_filter", "legacy_one_comma", "malform_steps_filter: legacy_one_comma | none | min_steps | max_steps")
dbutils.widgets.text("max_rows", "", "Optional max rows after load (empty = all)")
dbutils.widgets.text("epochs", "2", "num_train_epochs")
dbutils.widgets.text("eval_shortlist_k", "64", "Eval encoder shortlist size")
dbutils.widgets.text("mlflow_experiment", "", "Optional MLflow experiment path (empty = skip MLflow logging)")
dbutils.widgets.text("predictions_table", "", "Optional Hive table for eval rows, e.g. model_output.my_eval_run")

# COMMAND ----------

# -----------------------------------------------------------------------------
# Why this cell exists (for new users)
# -----------------------------------------------------------------------------
# Databricks runs each notebook with a default working directory that is NOT
# necessarily your Git repo folder. Python only imports modules that live on
# sys.path (or the current working directory in some cases).
#
# `dbutils...notebookPath()` often returns a *workspace* path such as
#   `/Users/you@domain.com/repo_name/NotebookName`
# That path is not always a real directory on the driver, so `os.chdir` can
# fail with ENOENT. We map it to the driver path Databricks uses, e.g.
#   `/Workspace/Users/you@domain.com/repo_name`
# and the same idea for `/Repos/...` -> `/Workspace/Repos/...`.
#
# We add the first existing candidate to sys.path and chdir only if that
# directory exists. If nothing matches, fall back to os.getcwd() and you may
# need to set REPO_ROOT in a widget or sys.path.insert manually.
# -----------------------------------------------------------------------------

import os
import sys
from pathlib import Path


def _repo_root_candidates(nb_path: str):
    """Ordered list of Path parents that might contain address_correction_ranking_pipeline.py."""
    if not nb_path:
        return []
    nb_path = nb_path.strip()
    p = Path(nb_path)
    out = [p.parent]
    if nb_path.startswith("/Users/"):
        tail = nb_path[len("/Users/") :].lstrip("/")
        mapped = Path("/Workspace/Users") / tail
        out.append(mapped.parent)
    if nb_path.startswith("/Repos/"):
        mapped = Path("/Workspace") / nb_path.lstrip("/")
        out.append(mapped.parent)
    # De-dupe while preserving order
    seen = set()
    deduped = []
    for q in out:
        s = str(q)
        if s not in seen:
            seen.add(s)
            deduped.append(q)
    return deduped


try:
    nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    repo_root = None
    fallback_root = None
    if nb_path:
        marker = Path("address_correction_ranking_pipeline.py")
        for cand in _repo_root_candidates(nb_path):
            if not cand.is_dir():
                continue
            if fallback_root is None:
                fallback_root = str(cand)
            if (cand / marker).is_file():
                repo_root = str(cand)
                break
        if repo_root is None:
            repo_root = fallback_root

    if repo_root:
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            os.chdir(repo_root)
        except OSError as e:
            print(f"Note: could not chdir to {repo_root!r} ({e}); sys.path was updated.")
    else:
        raise RuntimeError(f"No existing repo root derived from notebookPath={nb_path!r}")
except Exception as e:
    print(f"Could not resolve notebook path ({e}); using cwd only.")
    sys.path.insert(0, os.getcwd())

from address_correction_ranking_pipeline import PipelineConfig, run_pipeline

# COMMAND ----------


def _opt_int(name: str):
    raw = dbutils.widgets.get(name).strip()
    return int(raw) if raw else None


model_uri = dbutils.widgets.get("model_uri").strip() or None
source = dbutils.widgets.get("data_source")
if source == "parquet":
    pq = dbutils.widgets.get("source_parquet").strip()
    if not pq:
        raise ValueError("Set source_parquet when data_source is parquet.")
    cfg = PipelineConfig(
        source_table=None,
        source_parquet=pq,
        artifact_dir=dbutils.widgets.get("artifact_dir").strip(),
        hub_model_id=dbutils.widgets.get("hub_model_id").strip(),
        model_checkpoint_uri=model_uri,
        malform_steps_filter=dbutils.widgets.get("malform_filter").strip(),
        max_rows=_opt_int("max_rows"),
        num_train_epochs=int(dbutils.widgets.get("epochs").strip() or "2"),
        eval_shortlist_k=int(dbutils.widgets.get("eval_shortlist_k").strip() or "64"),
        mlflow_experiment_name=(dbutils.widgets.get("mlflow_experiment").strip() or None),
        predictions_table=(dbutils.widgets.get("predictions_table").strip() or None),
    )
else:
    cfg = PipelineConfig(
        source_table=dbutils.widgets.get("source_table").strip(),
        source_parquet=None,
        artifact_dir=dbutils.widgets.get("artifact_dir").strip(),
        hub_model_id=dbutils.widgets.get("hub_model_id").strip(),
        model_checkpoint_uri=model_uri,
        malform_steps_filter=dbutils.widgets.get("malform_filter").strip(),
        max_rows=_opt_int("max_rows"),
        num_train_epochs=int(dbutils.widgets.get("epochs").strip() or "2"),
        eval_shortlist_k=int(dbutils.widgets.get("eval_shortlist_k").strip() or "64"),
        mlflow_experiment_name=(dbutils.widgets.get("mlflow_experiment").strip() or None),
        predictions_table=(dbutils.widgets.get("predictions_table").strip() or None),
    )

print("Config:", cfg)

# COMMAND ----------

result = run_pipeline(cfg)

print("Done.")
print("Artifacts:", result.artifact_dir)
print("Trained model dir:", result.model_dir)
print("Eval accuracy (shortlist):", result.eval_accuracy)
print(
    "Malformation breakdown JSON:",
    result.artifact_dir / "eval_accuracy_by_malform_steps.json",
)
try:
    import pandas as pd

    display(pd.DataFrame(result.eval_accuracy_by_malform_steps))
except Exception:
    for row in result.eval_accuracy_by_malform_steps:
        print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: inspect a few eval rows
# MAGIC Run the cell below after a successful run.

# COMMAND ----------

display(result.eval_examples[:20])

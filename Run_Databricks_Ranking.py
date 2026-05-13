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
dbutils.widgets.text("max_rows", "500", "Optional max rows after load (empty = all)")
dbutils.widgets.text("epochs", "2", "num_train_epochs")
dbutils.widgets.text("eval_shortlist_k", "64", "Eval encoder shortlist size")
dbutils.widgets.text(
    "eval_ranker_max_encoder_rank",
    "24",
    "T5 ranker only sees first N encoder-shortlist lines (0 = all; stops deep hijacks)",
)
dbutils.widgets.dropdown(
    "eval_pick_policy",
    "encoder_margin_blend",
    ["encoder_margin_blend", "max_yes_no_margin", "min_nll_yes"],
    "How to combine NLL margin vs encoder in eval",
)
dbutils.widgets.text("eval_encoder_blend_weight", "2.0", "For encoder_margin_blend: weight on [0,1] norm encoder cos")
dbutils.widgets.text(
    "eval_lexical_penalty_per_missing_unit",
    "2.5",
    "Subtract this x (# suite/unit tokens in noisy missing from candidate) from pick_score; 0=off",
)
dbutils.widgets.text(
    "eval_lexical_max_unit_penalties",
    "6",
    "Max suite/unit misses counted per candidate toward that penalty (0=no cap)",
)
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

from address_correction_ranking_pipeline import (
    PipelineConfig,
    eval_examples_as_flat_records,
    run_pipeline,
)

# COMMAND ----------


def _opt_int(name: str):
    raw = dbutils.widgets.get(name).strip()
    return int(raw) if raw else None


def _opt_int_nonneg(name: str, default: int) -> int:
    raw = dbutils.widgets.get(name).strip()
    if not raw:
        return default
    return int(raw)


def _opt_float(name: str, default: float) -> float:
    raw = dbutils.widgets.get(name).strip()
    if not raw:
        return default
    return float(raw)


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
        eval_ranker_max_encoder_rank=_opt_int_nonneg(
            "eval_ranker_max_encoder_rank", 24
        ),
        eval_pick_policy=dbutils.widgets.get("eval_pick_policy").strip(),
        eval_encoder_blend_weight=_opt_float(
            "eval_encoder_blend_weight", 2.0
        ),
        eval_lexical_penalty_per_missing_unit=_opt_float(
            "eval_lexical_penalty_per_missing_unit", 2.5
        ),
        eval_lexical_max_unit_penalties=_opt_int_nonneg(
            "eval_lexical_max_unit_penalties", 6
        ),
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
        eval_ranker_max_encoder_rank=_opt_int_nonneg(
            "eval_ranker_max_encoder_rank", 24
        ),
        eval_pick_policy=dbutils.widgets.get("eval_pick_policy").strip(),
        eval_encoder_blend_weight=_opt_float(
            "eval_encoder_blend_weight", 2.0
        ),
        eval_lexical_penalty_per_missing_unit=_opt_float(
            "eval_lexical_penalty_per_missing_unit", 2.5
        ),
        eval_lexical_max_unit_penalties=_opt_int_nonneg(
            "eval_lexical_max_unit_penalties", 6
        ),
        mlflow_experiment_name=(dbutils.widgets.get("mlflow_experiment").strip() or None),
        predictions_table=(dbutils.widgets.get("predictions_table").strip() or None),
    )

print("Config:", cfg)

# COMMAND ----------

# Pre-tokenization scale hints (exact row counts appear in driver logs as
# "=== Tokenization preflight ===" from address_correction_ranking_pipeline.run_pipeline).
print("\n--- Tokenization / driver memory (before run_pipeline) ---")
print(f"  max_rows: {cfg.max_rows!r}")
if cfg.max_rows is not None:
    m = int(cfg.max_rows)
    approx_train_src = max(0, int(m * (1 - float(cfg.test_size))))
    print(
        f"  Upper bound on loaded rows: {m}; rough train split rows ~{approx_train_src} "
        f"(1 - test_size={cfg.test_size}) before hard-negative mining."
    )
else:
    print(
        "  max_rows is None: full table/parquet is loaded; tokenization cost scales with "
        "source size — use max_rows for smoke tests."
    )
print(f"  max_hard_negatives_per_row: {cfg.max_hard_negatives_per_row}")
print(f"  use_spark_tokenization: {cfg.use_spark_tokenization}")
print(f"  eval_ranker_max_encoder_rank: {cfg.eval_ranker_max_encoder_rank}")
print(f"  eval_pick_policy: {cfg.eval_pick_policy!r}")
print(f"  eval_encoder_blend_weight: {cfg.eval_encoder_blend_weight}")
print(
    f"  eval_lexical_penalty_per_missing_unit: {cfg.eval_lexical_penalty_per_missing_unit}"
)
print(f"  eval_lexical_max_unit_penalties: {cfg.eval_lexical_max_unit_penalties}")
print(
    "  Ranking examples per source row ≈ 1 + len(hard_negatives); worst case grows with "
    "neighbors/noisy candidates. tqdm 'Tokenize (chunk concat)' shows source rows/s; a long "
    "pause after the bar completes is often concatenate_datasets() on the driver."
)

# COMMAND ----------

result = run_pipeline(cfg)

print("Done.")
print("Artifacts:", result.artifact_dir)
print("Trained model dir:", result.model_dir)
print("Eval top-1 accuracy (see PipelineConfig.eval_pick_policy):", result.eval_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: inspect a few eval rows
# MAGIC Run the cell below after a successful run. Rows are flattened for ``display()`` (nested ``scores`` / ``shortlist`` would otherwise break Spark schema inference).

# COMMAND ----------

# Raw eval rows have nested "scores" (map) and "shortlist" (array); Databricks
# display() -> createDataFrame cannot infer types. eval_examples_as_flat_records
# also coerces null/NaN to empty strings and strict bools so schema inference is stable.
display(eval_examples_as_flat_records(result.eval_examples, limit=20))

# COMMAND ----------



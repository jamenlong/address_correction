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

# COMMAND ----------

# MAGIC %pip install -q "datasets>=2.14" "accelerate>=0.26" "transformers>=4.36" "scikit-learn>=1.3" "mlflow>=2.10" "tqdm"
<<<<<<< HEAD
=======
# MAGIC
# MAGIC dbutils.library.restartPython()
>>>>>>> 9ec63d26807dc488effc36d2a989bb516ce0c682

# COMMAND ----------

dbutils.widgets.dropdown("data_source", "delta_table", ["delta_table", "parquet"], "Data source")
dbutils.widgets.text("source_table", "smarty.smarty_malformed_1000_training", "Delta table (if delta_table)")
dbutils.widgets.text("source_parquet", "", "Parquet path /dbfs/... (if parquet)")
dbutils.widgets.text("artifact_dir", "/dbfs/FileStore/address_correction_ranking", "Artifact / output base (DBFS)")
dbutils.widgets.text("hub_model_id", "t5-small", "Hub model id when not using MLflow checkpoint")
dbutils.widgets.text("model_uri", "", "MLflow model URI, e.g. runs:/<run_id>/model (leave empty for Hub only)")
dbutils.widgets.text("malform_filter", "legacy_one_comma", "malform_steps_filter: legacy_one_comma | none | min_steps | max_steps")
<<<<<<< HEAD
dbutils.widgets.text("max_rows", "", "Optional max rows after load (empty = all)")
=======
dbutils.widgets.text("max_rows", "500", "Optional max rows after load (empty = all)")
>>>>>>> 9ec63d26807dc488effc36d2a989bb516ce0c682
dbutils.widgets.text("epochs", "2", "num_train_epochs")
dbutils.widgets.text("eval_shortlist_k", "64", "Eval encoder shortlist size")
dbutils.widgets.text("mlflow_experiment", "", "Optional MLflow experiment path (empty = skip MLflow logging)")
dbutils.widgets.text("predictions_table", "", "Optional Hive table for eval rows, e.g. model_output.my_eval_run")

# COMMAND ----------

<<<<<<< HEAD
# -----------------------------------------------------------------------------
# Why this cell exists (for new users)
# -----------------------------------------------------------------------------
# Databricks runs each notebook with a default working directory that is NOT
# necessarily your Git repo folder. Python only imports modules that live on
# sys.path (or the current working directory in some cases).
#
# This cell does three things:
#   1. Ask Databricks for this notebook's workspace path (e.g. under Repos).
#   2. Treat the *folder containing this notebook file* as the "repo root"
#      and add it to sys.path so `import address_correction_ranking_pipeline`
#      finds address_correction_ranking_pipeline.py sitting next to this file.
#   3. os.chdir(repo_root) so any relative paths in the pipeline behave as if
#      you ran from that folder.
#
# If step 1 fails (rare API/context issues), we fall back to os.getcwd() and
# print a hint — you may then need to sys.path.insert manually with your Repo path.
# -----------------------------------------------------------------------------

=======
>>>>>>> 9ec63d26807dc488effc36d2a989bb516ce0c682
import os
import sys
from pathlib import Path

<<<<<<< HEAD
try:
    nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    if nb_path:
        # Parent of this notebook = directory that should contain the pipeline .py
=======
# Resolve repo folder (directory containing this notebook) and import the pipeline module
try:
    nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    if nb_path:
        # Workspace path like /Repos/.../address_correction/Run_Databricks_Ranking
>>>>>>> 9ec63d26807dc488effc36d2a989bb516ce0c682
        repo_root = str(Path(nb_path).parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        os.chdir(repo_root)
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: inspect a few eval rows
# MAGIC Run the cell below after a successful run.

# COMMAND ----------

display(result.eval_examples[:20])

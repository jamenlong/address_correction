# Databricks notebook source
# MAGIC %md
# MAGIC # Recovery benchmark vs saved T5 predictions
# MAGIC
# MAGIC Compares the incremental recovery pipeline to a **baseline** `model_output.*_output` table
# MAGIC (JW metrics already computed for T5). Writes `model_output.<run>_recovery_eval` and prints
# MAGIC per-malform-step help/hurt vs the baseline.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Baseline table exists (default: `t5_product_corrector_training_20may2026_16_07_06_output`).
# MAGIC - `%run ./CharacterReplacementDictionary` for `char_malform_dct`.
# MAGIC - Repo contains `address_malform_recovery.py` and `address_recovery_benchmark.py`.
# MAGIC
# MAGIC ## Phase 1: recover **vs** T5 (hybrid override, no T5 re-run)
# MAGIC
# MAGIC | Step | Widgets | Goal |
# MAGIC |------|---------|------|
# MAGIC | 1 Smoke | `limit_rows` = `500`, `extra_where` = `malformed_first_line_malform_steps = ', replace_char'` | Fast sanity check |
# MAGIC | 2 Slice | `limit_rows` empty, same `extra_where` | Full `replace_char` slice vs May T5 baseline |
# MAGIC | 3 All rows | `extra_where` empty, `enabled_steps` = `replace_char` | Recovery only runs where steps allow; other rows unchanged |
# MAGIC
# MAGIC Read **`pct_exact_gain`**, **`fixed_t5_miss`**, **`broke_t5_hit`** in the summary tables. Inspect **wins** / **regressions** cells at the bottom.
# MAGIC
# MAGIC T5 baseline table is **read-only**. Results go to `model_output.<run>_recovery_eval` (or `output_table` if set).
# MAGIC
# MAGIC **Run order:** widgets → repo path → `%pip` + `%restart_python` → repo path again →
# MAGIC `%run CharacterReplacementDictionary` → benchmark cells. After restart, always re-run
# MAGIC from the **repo path** cell through the end (not only cell 9).

# COMMAND ----------

dbutils.widgets.dropdown(
    "run_phase",
    "smoke_replace_char",
    [
        "smoke_replace_char",
        "full_replace_char",
        "full_all_steps_allowed",
        "custom",
    ],
    "Preset widget bundle (custom = use widgets as-is)",
)

# COMMAND ----------

dbutils.widgets.text(
    "baseline_table",
    "model_output.t5_product_corrector_training_20may2026_16_07_06_output",
    "T5 benchmark Delta table",
)
dbutils.widgets.text(
    "output_table",
    "",
    "Recovery eval output (empty = model_output.<baseline_stem>_recovery_eval)",
)
dbutils.widgets.text(
    "enabled_steps",
    "replace_char",
    "Comma-separated malform undo steps to enable",
)
dbutils.widgets.text(
    "extra_where",
    "",
    "Optional SQL AND filter, e.g. malformed_first_line_malform_steps = ', replace_char'",
)
dbutils.widgets.text("limit_rows", "", "Optional LIMIT for smoke tests (empty = all)")
dbutils.widgets.text("min_jw_to_accept", "0.92", "Recovery catalog accept threshold")
dbutils.widgets.text("min_jw_margin_over_t5", "0.05", "Hybrid override margin vs T5 JW")

# COMMAND ----------

import os
import sys

notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
candidates = []
if notebook_path.startswith("/Users/"):
    candidates.append("/Workspace" + notebook_path.rsplit("/", 1)[0])
elif notebook_path.startswith("/Repos/"):
    candidates.append("/Workspace" + notebook_path.rsplit("/", 1)[0])
else:
    candidates.append(os.path.dirname(notebook_path))
candidates.append(os.getcwd())
for p in candidates:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
        os.chdir(p)
        break

# COMMAND ----------

# Optional: speeds up catalog JW matching (~2-5x). Run once per cluster, then restart Python.
%pip install -q rapidfuzz

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Re-run after restart: repo path + malform dictionary (restart clears all variables).
import os
import sys

notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
for p in (
    "/Workspace" + notebook_path.rsplit("/", 1)[0]
    if notebook_path.startswith(("/Users/", "/Repos/"))
    else os.path.dirname(notebook_path),
    os.getcwd(),
):
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
        os.chdir(p)
        break

# COMMAND ----------

# MAGIC %run ./CharacterReplacementDictionary

# COMMAND ----------

from address_malform_recovery import RecoveryConfig
from address_recovery_benchmark import (
    BenchmarkTableConfig,
    aggregate_benchmark_rows,
    benchmark_summary_sql,
    run_recovery_benchmark_spark,
)

enabled = frozenset(
    s.strip() for s in dbutils.widgets.get("enabled_steps").split(",") if s.strip()
)
rcfg = RecoveryConfig(
    enabled_steps=enabled,
    min_jw_to_accept=float(dbutils.widgets.get("min_jw_to_accept")),
    min_jw_margin_over_t5=float(dbutils.widgets.get("min_jw_margin_over_t5")),
)

_phase = dbutils.widgets.get("run_phase")
bench = BenchmarkTableConfig(baseline_table=dbutils.widgets.get("baseline_table"))
out_table = dbutils.widgets.get("output_table").strip() or None

# Presets override limit/extra_where unless run_phase is custom
_lim_widget = dbutils.widgets.get("limit_rows").strip()
_where_widget = dbutils.widgets.get("extra_where").strip()
if _phase == "smoke_replace_char":
    limit = 500
    extra_where = "malformed_first_line_malform_steps = ', replace_char'"
elif _phase == "full_replace_char":
    limit = None
    extra_where = "malformed_first_line_malform_steps = ', replace_char'"
elif _phase == "full_all_steps_allowed":
    limit = None
    extra_where = None
else:
    limit = int(_lim_widget) if _lim_widget else None
    extra_where = _where_widget or None

print(f"Run phase: {_phase}")
print(f"Baseline: {bench.baseline_table}")
print(f"Enabled recovery steps: {sorted(enabled)}")
print(f"Output table: {out_table or bench.output_table()}")

# COMMAND ----------

enriched_sdf, report = run_recovery_benchmark_spark(
    spark,
    char_malform_dct,
    recovery_config=rcfg,
    bench=bench,
    output_table=out_table,
    extra_where=extra_where,
    limit=limit,
)

# COMMAND ----------

import pandas as pd

display(pd.DataFrame([report["overall"]]))
display(pd.DataFrame(report["by_malform_step"]).head(50))

# COMMAND ----------

eval_table = out_table or bench.output_table()
queries = benchmark_summary_sql(eval_table)

display(spark.sql(queries["overall"]))
display(spark.sql(queries["by_malform_step"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect wins / regressions vs T5 baseline

# COMMAND ----------

display(spark.sql(queries["wins"]))
display(spark.sql(queries["regressions"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outcome breakdown & still-unfixed rows
# MAGIC
# MAGIC - **outcome_breakdown**: already correct under T5, fixed by recovery, regressions, still wrong
# MAGIC - **still_unfixed**: rows where `hybrid_exact` is false (detail, worst JW first)
# MAGIC - **still_unfixed_breakdown**: why unfixed rows stayed wrong (`recovery_used`, `prediction_source`)

# COMMAND ----------

display(spark.sql(queries["outcome_breakdown"]))
display(spark.sql(queries["still_unfixed"]))
display(spark.sql(queries["still_unfixed_breakdown"]))

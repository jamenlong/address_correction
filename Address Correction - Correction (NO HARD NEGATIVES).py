# Databricks notebook source
# %pip install datasets
# dbutils.library.restartPython()

# COMMAND ----------

# %pip install transformers==4.29.2 datasets accelerate sentencepiece
%pip install jellyfish
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, StringType, FloatType
import jellyfish

def get_jw_score(str1, str2):
    """
    get_jw_score will calculate and return the Jaro-Winkler score between two strings.
    
    Parameters:
    str1 (str): The first string to compare.
    str2 (str): The second string to compare.
    
    Returns:
    float: The Jaro-Winkler score between the two strings.
    """
    try:
        jaro_winkler_score = jellyfish.jaro_winkler_similarity(str1, str2)

        return jaro_winkler_score
    
    except:
        return 9.9999

# Register the UDF
get_jw_score_udf = udf(get_jw_score, FloatType())

# Apply UDF
# result_df = df.withColumn("jw_score", get_jw_score_udf(df.string1, df.string2))

# COMMAND ----------

# Analysis without re-running slow test-set generation (~55 min):
#   analysis_mode = reload_saved  → load model_output.<run_name>_output (or parquet backup)
#   analysis_mode = run_full_prediction → run the generation loop below, then save
dbutils.widgets.dropdown(
    "analysis_mode",
    "reload_saved",
    ["reload_saved", "run_full_prediction"],
    "reload_saved skips model.generate on the full test set",
)
dbutils.widgets.text(
    "analysis_run_name",
    "",
    "Run name only, e.g. t5_product_corrector_training_03DEC2025_05_20_02 (empty = last run that finished saving predictions)",
)
dbutils.widgets.text(
    "analysis_predictions_table",
    "",
    "Optional full table name if not model_output.<run_name>_output (e.g. hive_metastore.model_output.my_output)",
)

# COMMAND ----------

from typing import List, Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# DBFS folder for parquet backups and last-run pointer (survives notebook restarts)
EVAL_ARTIFACT_DIR = "/dbfs/FileStore/address_correction_eval"
EVAL_ARTIFACT_DBFS = "dbfs:/FileStore/address_correction_eval"


def _eval_artifact_path_local(run_name: str) -> str:
    """Driver/local path (open(), pandas)."""
    return f"{EVAL_ARTIFACT_DIR}/{run_name}_test_predictions.parquet"


def _eval_artifact_path_spark(run_name: str) -> str:
    """Spark read/write path on Databricks."""
    return f"{EVAL_ARTIFACT_DBFS}/{run_name}_test_predictions.parquet"


def _last_run_name_path() -> str:
    return f"{EVAL_ARTIFACT_DIR}/last_run_name.txt"


def save_last_run_name(run_name: str) -> None:
    import os

    os.makedirs(EVAL_ARTIFACT_DIR, exist_ok=True)
    with open(_last_run_name_path(), "w", encoding="utf-8") as f:
        f.write(run_name)


def resolve_analysis_run_name(explicit: str = "") -> str:
    """Pick run_name for reload: widget → last *saved predictions* pointer → error."""
    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    try:
        with open(_last_run_name_path(), encoding="utf-8") as f:
            last = f.read().strip()
    except OSError:
        last = ""
    if last:
        print(f"Using last saved predictions run_name from {_last_run_name_path()}: {last}")
        return last
    raise ValueError(
        "Set the analysis_run_name widget to a run whose predictions were saved "
        "(see SHOW TABLES IN model_output LIKE '*_output'), or run once with "
        "analysis_mode=run_full_prediction through the save cell."
    )


def list_saved_prediction_tables() -> List[str]:
    """Return fully qualified table names under schema model_output ending in _output."""
    names: List[str] = []
    try:
        rows = spark.sql("SHOW TABLES IN model_output LIKE '*_output'").collect()
    except Exception:
        try:
            spark.sql("CREATE DATABASE IF NOT EXISTS model_output")
            rows = spark.sql("SHOW TABLES IN model_output LIKE '*_output'").collect()
        except Exception:
            return names
    for row in rows:
        t = getattr(row, "tableName", None) or row["tableName"]
        names.append(f"model_output.{t}")
    return sorted(names)


def _try_load_table(table_name: str) -> Optional[DataFrame]:
    try:
        sdf = spark.table(table_name)
        _ = sdf.limit(1).count()
        return sdf
    except Exception:
        return None


def _try_load_parquet(path: str) -> Optional[DataFrame]:
    try:
        sdf = spark.read.parquet(path)
        _ = sdf.limit(1).count()
        return sdf
    except Exception:
        return None


def find_prediction_sources(run_name: str, table_override: str = "") -> List[Tuple[str, str]]:
    """Return [(kind, location), ...] for existing artifacts, best match first."""
    found: List[Tuple[str, str]] = []
    override = (table_override or "").strip()
    if override:
        if _try_load_table(override) is not None:
            found.append(("delta", override))

    seen = {loc for _, loc in found}
    primary = f"model_output.{run_name}_output"
    for tbl in [primary, f"hive_metastore.{primary}"]:
        if tbl not in seen and _try_load_table(tbl) is not None:
            found.append(("delta", tbl))
            seen.add(tbl)

    for tbl in list_saved_prediction_tables():
        if tbl not in seen and run_name in tbl.split(".")[-1]:
            if _try_load_table(tbl) is not None:
                found.append(("delta", tbl))
                seen.add(tbl)

    for pq in (_eval_artifact_path_spark(run_name), _eval_artifact_path_local(run_name)):
        if pq not in seen and _try_load_parquet(pq) is not None:
            found.append(("parquet", pq))
            seen.add(pq)

    # Same session: generate finished but publish/save cell not run yet
    if "test_predictions" not in seen:
        try:
            sdf = spark.table("test_predictions")
            if "predicted" in sdf.columns:
                found.append(("temp_view", "test_predictions"))
        except Exception:
            pass

    return found


def add_jw_improvement_columns(sdf: DataFrame) -> DataFrame:
    """Add jw_input, jw_output, jw_delta, jw_relative_gain, jw_improved (and jw_score alias)."""
    if "jw_output" not in sdf.columns and "jw_score" in sdf.columns:
        sdf = sdf.withColumn("jw_output", F.col("jw_score"))
    if "jw_input" not in sdf.columns:
        sdf = sdf.withColumn(
            "jw_input",
            get_jw_score_udf(sdf.malformed_first_line, sdf.first_line),
        )
    if "jw_output" not in sdf.columns:
        sdf = sdf.withColumn(
            "jw_output",
            get_jw_score_udf(sdf.first_line, sdf.predicted),
        )
    sdf = sdf.withColumn("jw_score", F.col("jw_output"))
    if "jw_delta" not in sdf.columns:
        sdf = sdf.withColumn("jw_delta", F.col("jw_output") - F.col("jw_input"))
    if "jw_relative_gain" not in sdf.columns:
        sdf = sdf.withColumn(
            "jw_relative_gain",
            F.when(
                F.col("jw_input") < F.lit(1.0),
                (F.col("jw_output") - F.col("jw_input"))
                / (F.lit(1.0) - F.col("jw_input")),
            ),
        )
    if "jw_improved" not in sdf.columns:
        sdf = sdf.withColumn("jw_improved", F.col("jw_delta") > F.lit(0.0))
    return sdf


def load_saved_predictions(
    run_name: str,
    table_override: str = "",
) -> DataFrame:
    """Load predictions from Delta, parquet, or temp view; print hints if missing."""
    sources = find_prediction_sources(run_name, table_override=table_override)
    if not sources:
        available = list_saved_prediction_tables()
        msg = [
            f"No saved predictions for run_name={run_name!r}.",
            "",
            "Common causes:",
            "  • Training set last_run_name.txt but the slow prediction + SAVE cell never ran.",
            "  • Cluster/metastore changed (table exists in another workspace).",
            "",
            "Fix: set analysis_mode=run_full_prediction and run through the cell that",
            "writes model_output.<run_name>_output (or set analysis_predictions_table).",
            "",
            f"Expected Delta: model_output.{run_name}_output",
            f"Expected parquet: {_eval_artifact_path_spark(run_name)}",
        ]
        if available:
            msg.append("")
            msg.append("Tables found in model_output:")
            for t in available[-20:]:
                msg.append(f"  • {t}")
            msg.append(
                "Copy the matching run into the analysis_run_name widget "
                "(the part before _output)."
            )
        else:
            msg.append("")
            msg.append("No model_output.*_output tables found in this metastore.")
        raise RuntimeError("\n".join(msg))

    kind, loc = sources[0]
    if len(sources) > 1:
        print("Also found:", ", ".join(f"{k}={v}" for k, v in sources[1:]))
    if kind == "delta" or kind == "temp_view":
        sdf = spark.table(loc)
    else:
        sdf = spark.read.parquet(loc)
    n = sdf.count()
    print(f"Loaded {n} rows from {kind} ({loc})")
    return sdf


def publish_predictions_with_jw_metrics(
    sdf: DataFrame,
    run_name: str,
    *,
    write_delta: bool = True,
    write_parquet: bool = True,
) -> DataFrame:
    """Ensure JW columns exist, register temp view, optionally persist."""
    sdf = add_jw_improvement_columns(sdf)
    sdf.createOrReplaceTempView("test_predictions")
    table = f"model_output.{run_name}_output"
    if write_delta:
        sdf.write.mode("overwrite").saveAsTable(table)
        print(f"Updated Delta table {table}")
    if write_parquet:
        import os

        os.makedirs(EVAL_ARTIFACT_DIR, exist_ok=True)
        pq_spark = _eval_artifact_path_spark(run_name)
        sdf.write.mode("overwrite").parquet(pq_spark)
        print(f"Wrote parquet backup to {pq_spark}")
    save_last_run_name(run_name)
    return sdf

# COMMAND ----------

test_sdf = spark.sql("""
SELECT *
FROM smarty.smarty_malformed_1000_training
WHERE (LENGTH(malformed_first_line_malform_steps) - LENGTH(REPLACE(malformed_first_line_malform_steps, ',', ''))) = 1
-- SORT BY RAND(2112)
-- LIMIT 5000
""")

test_sdf.createOrReplaceTempView("test")

dist_sdf = spark.sql("""
SELECT malformed_first_line_malform_steps
     , COUNT(*) AS record_count
FROM test
GROUP BY 1
ORDER BY 2 DESC""")

n_orig_addrs_sdf = spark.sql("""
SELECT first_line
     , COUNT(*) AS n_records
FROM test
GROUP BY 1
ORDER BY 2 DESC                             
                             """)

print (f"N test rows: {test_sdf.count()}")
print (f"N original addresses: {n_orig_addrs_sdf.count()}")


display(dist_sdf)

test_pdf = test_sdf.toPandas()

display(test_sdf)
display(n_orig_addrs_sdf)

# COMMAND ----------

display(n_orig_addrs_sdf.sort('first_line'))

# COMMAND ----------

from datasets import Dataset

test_pdf_cleaned = test_pdf.convert_dtypes().infer_objects().copy()

test_hfdf = Dataset.from_pandas(test_pdf_cleaned, preserve_index=False)
dataset = test_hfdf.train_test_split(test_size=0.1, seed=2112)  # 90/10 split for training/validation


# COMMAND ----------

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")  

def preprocess(example, in_col_name = "malformed_first_line", out_col_name = "first_line"):
    input_enc = tokenizer(
        example[in_col_name], 
        truncation=True, 
        padding="max_length", 
        max_length=32
    )
    target_enc = tokenizer(
        example[out_col_name], 
        truncation=True, 
        padding="max_length", 
        max_length=32
    )

    input_enc["labels"] = target_enc["input_ids"]
    
    return input_enc

tokenized_dataset = dataset.map(preprocess, batched=True)


# COMMAND ----------

# Specifying which addresses can actually be predicted

valid_product_names = n_orig_addrs_sdf.select("first_line").distinct().toPandas()['first_line'].tolist()

# Pre-tokenize the allowed outputs
allowed_token_seqs = [tokenizer.encode(name, add_special_tokens=False) 
                      for name in valid_product_names]

def prefix_allowed_tokens_fn(batch_id, generated_ids):
    # Convert to 1D Python list
    if generated_ids.ndim == 0:
        generated = [generated_ids.item()]
    else:
        generated = generated_ids.tolist()

    # Skip T5's decoder_start_token_id (usually <pad>=0)
    if len(generated) > 0 and generated[0] == tokenizer.pad_token_id:
        effective_prefix = generated[1:]
    else:
        effective_prefix = generated

    allowed_next = set()

    for seq in allowed_token_seqs:
        # Compare only after skipping the initial <pad>
        if effective_prefix == seq[:len(effective_prefix)]:
            # Entire sequence matched
            if len(effective_prefix) == len(seq):
                allowed_next.add(tokenizer.eos_token_id)
            else:
                allowed_next.add(seq[len(effective_prefix)])

    # If no possible match, fallback to EOS
    if not allowed_next:
        return [tokenizer.eos_token_id]

    return list(allowed_next)


# COMMAND ----------

# Print valid product names for later investigations
valid_product_names

# COMMAND ----------

# Print test and train dataset sizes
print(f"Training dataset size: {tokenized_dataset['train'].shape}")
print(f"Test dataset size: {tokenized_dataset['test'].shape}")


# COMMAND ----------

def get_timestamp():
    
    from datetime import datetime
    
    month_dct = {
        "01": "JAN",
        "02": "FEB",
        "03": "MAR",
        "04": "APR",
        "05": "MAY",
        "06": "JUN",
        "07": "JUL",
        "08": "AUG",
        "09": "SEP",
        "10": "OCT",
        "11": "NOV",
        "12": "DEC"}
    
    timestamp = datetime.now().strftime("%d%m%Y_%H_%M_%S")
    timestamp = timestamp[:2] + month_dct[timestamp[2:4]] + timestamp[4:]

    return timestamp

# COMMAND ----------

!pip install mlflow

# COMMAND ----------

from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer, pipeline
import mlflow
import mlflow.transformers

model = T5ForConditionalGeneration.from_pretrained("t5-small")

import os
import transformers

print("TrainingArguments location:", TrainingArguments.__module__)
print("Transformers version:", transformers.__version__)

# HF Trainer checkpoints are large (~hundreds of MB per epoch). Writing under the
# Repo folder blows past Databricks Repos' ~1 GB Git working-tree limit and breaks
# git pull/push. Use DBFS instead (same idea as artifact_dir in Run_Databricks_Ranking).
HF_TRAINER_ROOT = "/dbfs/FileStore/address_correction_training"
os.makedirs(HF_TRAINER_ROOT, exist_ok=True)
_hf_output_dir = os.path.join(HF_TRAINER_ROOT, "product_corrector")
_hf_logging_dir = os.path.join(HF_TRAINER_ROOT, "logs")
print(f"Trainer output_dir: {_hf_output_dir}")

training_args = TrainingArguments(
    output_dir=_hf_output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir=_hf_logging_dir,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Set your experiment (see notes from above)
experiment_name = "/Users/jamenlong@yahoo.com/t5_experiments"

# Set MLFlow experiment
mlflow.set_experiment(experiment_name)


# Add identifying information to each run so they can be distinguished
from datetime import datetime
ts = get_timestamp()
run_name = f"t5_product_corrector_training_{ts}"
print(f"Run name: {run_name}")
print(
    "Note: predictions are saved under this run_name only after the slow "
    "prediction + publish cell (not at training time)."
)

# Add notes/tags for each run
dataset_version = "v0.1"
personal_notes = "Restarting this to get a cleaner comparison."
dataset_notes = "Testing on just single malform steps. Also testing on specifying which addresses can be used as prediction values. This is after adjusting some functions so that there are more malformations done with each malform function. In other words, in some cases, random.randint() returned 0, so nothing was malformed. These were changed to a minimum of 1."
model_notes = "model = T5ForConditionalGeneration.from_pretrained('t5-small')"
training_args = """output_dir='./product_corrector',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True"""

with mlflow.start_run(run_name=run_name) as run:
    # Add notes
    mlflow.set_tag("dataset_version", dataset_version)
    mlflow.set_tag("personal_notes", personal_notes)
    mlflow.set_tag("dataset_notes", dataset_notes)
    mlflow.set_tag("model_notes", model_notes)
    mlflow.set_tag("training_args", training_args)

    run_id = run.info.run_id
    print("Run ID:", run_id)
    
    # Train model
    trainer.train()

    # Log final metrics if you have them
    eval_results = trainer.evaluate()
    mlflow.log_metrics(eval_results)

    # ✅ Create a text2text-generation pipeline for MLflow
    pipe = pipeline("text2text-generation", model=trainer.model, tokenizer=tokenizer)

    # ✅ Log pipeline to MLflow and register it
    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path="model",
        registered_model_name=f"AddressCorrectorT5small_{ts}"
    )

print(f"✅ Training complete and model logged to MLflow successfully. Run name: {run_name}")


# COMMAND ----------

import mlflow

# See above for inputs
experiment = mlflow.get_experiment_by_name(experiment_name)

# Get latest run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

latest_run = runs.iloc[0]

run_id = latest_run["run_id"]
run_name = latest_run["tags.mlflow.runName"]

print("Run ID:", run_id)
print("Run Name:", run_name)

# COMMAND ----------

# Evaluate training. Using "try/except" to avoid downstream cancellations
try:
    results = trainer.evaluate()
    print("Evaluation metrics:", results)

except Exception as e:
    print("Error evaluating model:", e)    

# COMMAND ----------

# MAGIC %skip
# MAGIC import re
# MAGIC
# MAGIC def clean_string(input_string):
# MAGIC     
# MAGIC     """
# MAGIC     clean_string will preprocess the input string by doing a few things:
# MAGIC     1. Replace all special characters with a space
# MAGIC     2. Reduce multiple spaces to a single space
# MAGIC     3. Strip leading and trailing spaces
# MAGIC     4. Reduces all occurrences of more than two consecutive characters to two consecutive characters.
# MAGIC
# MAGIC     Args:
# MAGIC     addr (string): The address to be cleaned
# MAGIC
# MAGIC     Returns:
# MAGIC     string: The cleaned address
# MAGIC     """
# MAGIC
# MAGIC     # Replace special characters with a space
# MAGIC     spec_char_replaced_addr = re.sub(r'[^a-zA-Z0-9]', ' ', input_string)
# MAGIC
# MAGIC     # Reduce multiple spaces to a single space
# MAGIC     cleaned_spaces_string = re.sub(r'\s+', ' ', spec_char_replaced_addr).strip()  # .strip() removes leading/trailing spaces
# MAGIC
# MAGIC     # Reduce all occurrences of more than two consecutive characters to two consecutive characters.
# MAGIC     cleaned_string = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', cleaned_spaces_string, flags=re.IGNORECASE)
# MAGIC
# MAGIC     return cleaned_string

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 21 — Test-set predictions OR reload saved results
# MAGIC
# MAGIC **What this cell does**
# MAGIC - Branches on the **`analysis_mode`** widget (set near the top of the notebook).
# MAGIC - **`run_full_prediction` (slow, ~55 min):** Loops over every row in `dataset["test"]`, runs
# MAGIC   `model.generate` with **`prefix_allowed_tokens_fn`** so each prediction is constrained to a
# MAGIC   valid catalog `first_line`, and builds `model_evaluated_test_pdf`. Does **not** write Delta yet —
# MAGIC   run the **next cell** to join, compute JW metrics, and save.
# MAGIC - **`reload_saved` (fast):** Skips `model.generate`. Loads a prior run from
# MAGIC   `model_output.<run_name>_output` or `dbfs:/FileStore/address_correction_eval/...parquet`,
# MAGIC   adds JW columns if missing, and refreshes the table. Use this when you return later for analysis only.
# MAGIC
# MAGIC **Why two paths**
# MAGIC - Generation is expensive; JW/improvement analysis should not require re-running it.
# MAGIC - Reload only works after a successful slow path + save cell has created artifacts for that `run_name`.
# MAGIC
# MAGIC **Widgets**
# MAGIC - `analysis_run_name` — run suffix (e.g. `t5_product_corrector_training_18MAY2026_18_56_21`), or empty to use
# MAGIC   the last run that **finished saving** predictions (`last_run_name.txt` on DBFS).
# MAGIC - `analysis_predictions_table` — optional full table name override if your table is not under `model_output.*`.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Slow path: training cell completed (`model`, `tokenizer`, `dataset`, `prefix_allowed_tokens_fn`, `run_name`).
# MAGIC - Reload path: jellyfish UDF + helper cells only; **do not** need `model` loaded.
# MAGIC
# MAGIC **Fresh full notebook run:** set `analysis_mode` = **`run_full_prediction`**, then run this cell and the **next** cell.

# COMMAND ----------

# Cell 21 — see markdown above. Slow path: constrained generate on test split.
# Fast path: load_saved_predictions → publish_predictions_with_jw_metrics (JW + Delta + parquet).

import pandas as pd
import torch
from tqdm import tqdm

ANALYSIS_MODE = dbutils.widgets.get("analysis_mode").strip()
SKIP_SLOW_PREDICTION = ANALYSIS_MODE == "reload_saved"

if SKIP_SLOW_PREDICTION:
    run_name = resolve_analysis_run_name(dbutils.widgets.get("analysis_run_name"))
    table_override = dbutils.widgets.get("analysis_predictions_table").strip()
    print(f"reload_saved: skipping model.generate; run_name={run_name!r}")
    sources = find_prediction_sources(run_name, table_override=table_override)
    if sources:
        print("Found:", ", ".join(f"{k}={v}" for k, v in sources))
    else:
        print("No artifacts yet for this run_name. Available tables:")
        for t in list_saved_prediction_tables()[-15:]:
            print(f"  {t}")
    test_pdf_predictions_sdf = publish_predictions_with_jw_metrics(
        load_saved_predictions(run_name, table_override=table_override),
        run_name,
        write_delta=True,
        write_parquet=True,
    )
    display(test_pdf_predictions_sdf.limit(20))
else:
    # --- Slow path: generate predictions on the full test split ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_evaluated_test_pdf = pd.DataFrame(
        columns=["malformed_first_line", "predicted", "reference"]
    )
    eval_samples = dataset["test"]
    predictions = []
    references = []

    for example in tqdm(eval_samples):
        input_text = example["malformed_first_line"]
        ref_text = example["first_line"]
        input_ids = tokenizer.encode(
            input_text, return_tensors="pt", truncation=True, max_length=32
        ).to(device)
        output_ids = model.generate(
            input_ids,
            max_length=32,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred_text)
        references.append(ref_text)

    for malformed, pred, ref in zip(
        eval_samples["malformed_first_line"], predictions, references
    ):
        model_evaluated_test_pdf.loc[len(model_evaluated_test_pdf)] = [
            malformed,
            pred,
            ref,
        ]

    display(model_evaluated_test_pdf)

# COMMAND ----------

# Join + JW metrics + persist (slow path only; fast path already finished above)
if dbutils.widgets.get("analysis_mode").strip() != "reload_saved":
    if not globals().get("run_name"):
        raise RuntimeError(
            "run_name is not set. Run the training cell first, or set analysis_mode=reload_saved."
        )
    test_pdf = dataset["test"].to_pandas()
    test_pdf_predictions = test_pdf.merge(
        model_evaluated_test_pdf, on="malformed_first_line", how="left"
    )
    test_pdf_predictions_sdf = spark.createDataFrame(test_pdf_predictions)
    test_pdf_predictions_sdf = publish_predictions_with_jw_metrics(
        test_pdf_predictions_sdf,
        run_name,
        write_delta=True,
        write_parquet=True,
    )
    s3_path = (
        f"s3://jml-address-validation/tables/model_output/test/{run_name}/{run_name}_output"
    )
    print(f"Table model_output.{run_name}_output saved to {s3_path}")
    display(test_pdf_predictions_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improvement summary (input vs output JW)
# MAGIC **Fast path (no ~55 min generate):** set widget `analysis_mode` = `reload_saved`, set `analysis_run_name` (or leave empty to use last run), then run the **reload** cell above and **this section only** (skip training + generate).
# MAGIC - **jw_input**: similarity of malformed line to truth before the model.
# MAGIC - **jw_output** / **jw_score**: similarity of prediction to truth.
# MAGIC - **jw_delta**: `jw_output - jw_input` (positive = model moved closer to truth).
# MAGIC - **jw_relative_gain**: `jw_delta / (1 - jw_input)` when `jw_input < 1` — fraction of *remaining* error closed (1.0 = perfect correction to JW 1).

# COMMAND ----------

# Resolve run_name for SQL below (safe when re-opening notebook and jumping here)
if not globals().get("run_name"):
    run_name = resolve_analysis_run_name(dbutils.widgets.get("analysis_run_name"))
print(f"Analysis using run_name={run_name!r}  →  model_output.{run_name}_output")

# COMMAND ----------

# List saved prediction tables (pick a name for analysis_run_name widget)
display(
    spark.sql("SHOW TABLES IN model_output LIKE '*_output'").select(
        "tableName"
    )
)

# COMMAND ----------

# Overall improvement stats (non-trivial rows; exclude UDF error sentinel 9.9999)
jw_summary_sdf = spark.sql(f"""
SELECT COUNT(*) AS n_rows
     , ROUND(AVG(jw_input), 4) AS avg_jw_input
     , ROUND(AVG(jw_output), 4) AS avg_jw_output
     , ROUND(AVG(jw_delta), 4) AS avg_jw_delta
     , ROUND(AVG(jw_relative_gain), 4) AS avg_relative_gain
     , ROUND(100.0 * SUM(CASE WHEN jw_improved THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_improved
     , ROUND(100.0 * SUM(CASE WHEN jw_delta < 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_harmed
     , ROUND(100.0 * SUM(CASE WHEN jw_output = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_output_jw_exact
     , ROUND(100.0 * SUM(CASE WHEN jw_input = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_input_already_exact
FROM model_output.{run_name}_output
WHERE predicted IS NOT NULL
  AND malformed_first_line != first_line
  AND jw_input <= 1
  AND jw_output <= 1
""")

display(jw_summary_sdf)

# Relative gain by malform step (same filters)
jw_gain_by_step_sdf = spark.sql(f"""
SELECT malformed_first_line_malform_steps
     , COUNT(*) AS record_count
     , ROUND(AVG(jw_input), 4) AS avg_jw_input
     , ROUND(AVG(jw_output), 4) AS avg_jw_output
     , ROUND(AVG(jw_delta), 4) AS avg_jw_delta
     , ROUND(AVG(jw_relative_gain), 4) AS avg_relative_gain
     , ROUND(100.0 * SUM(CASE WHEN jw_improved THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_improved
     , ROUND(100.0 * SUM(CASE WHEN jw_output = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS perc_correct
FROM model_output.{run_name}_output
WHERE predicted IS NOT NULL
  AND malformed_first_line != first_line
  AND jw_input <= 1
  AND jw_output <= 1
GROUP BY 1
ORDER BY avg_jw_delta DESC
""")

display(jw_gain_by_step_sdf)

# COMMAND ----------

sdf = spark.sql("""
SELECT first_line
     , COUNT(*) As record_count
FROM test
GROUP BY 1           
ORDER BY 1
                """)

display(sdf) 

# COMMAND ----------

run_id

# COMMAND ----------

# Distribution of JW scores by malform steps

# run_name = 't5_product_corrector_training_03DEC2025_05_20_02'

distr_sdf = spark.sql(f"""
SELECT malformed_first_line_malform_steps
     , ROUND(AVG(jw_input), 4) AS avg_jw_input
     , ROUND(AVG(jw_output), 4) AS avg_jw_output
     , ROUND(AVG(jw_delta), 4) AS avg_jw_delta
     , ROUND(AVG(jw_relative_gain), 4) AS avg_relative_gain
     , COUNT(*) AS record_count
FROM model_output.{run_name}_output
WHERE jw_input <= 1
  AND jw_output <= 1
  AND malformed_first_line != first_line
GROUP BY 1
ORDER BY avg_jw_output DESC
""")

display(distr_sdf)

# COMMAND ----------


#  Which malform patterns are the hardest for the model to predict?

malform_pattern_dist = spark.sql(f"""
SELECT malformed_first_line_malform_steps
     , correct
     , total_count
     , ROUND(correct / total_count, 2) AS perc_correct
     , ROUND(avg_jw_input, 4) AS avg_jw_input
     , ROUND(avg_jw_output, 4) AS avg_jw_output
     , ROUND(avg_jw_delta, 4) AS avg_jw_delta
     , ROUND(avg_relative_gain, 4) AS avg_relative_gain
FROM (
        SELECT malformed_first_line_malform_steps
            , SUM(CASE WHEN jw_output = 1 THEN 1 ELSE 0 END) AS correct
            , COUNT(*) AS total_count
            , AVG(jw_input) AS avg_jw_input
            , AVG(jw_output) AS avg_jw_output
            , AVG(jw_delta) AS avg_jw_delta
            , AVG(jw_relative_gain) AS avg_relative_gain
        FROM model_output.{run_name}_output
        WHERE predicted IS NOT NULL
          AND malformed_first_line != first_line
          AND jw_input <= 1
          AND jw_output <= 1
        GROUP BY 1)
        """)

display(malform_pattern_dist)        

# COMMAND ----------

# Examples of difficult malform patterns

examples_sdf = spark.sql(f"""
SELECT first_line
     , malformed_first_line
     , malformed_first_line_malform_steps
     , predicted
     , reference
     , ROUND(jw_input, 4) AS jw_input
     , ROUND(jw_output, 4) AS jw_output
     , ROUND(jw_delta, 4) AS jw_delta
     , ROUND(jw_relative_gain, 4) AS jw_relative_gain
     , jw_improved
FROM model_output.{run_name}_output
WHERE malformed_first_line_malform_steps = ', add_special_characters'
  AND malformed_first_line != first_line
ORDER BY jw_delta
""")

display(examples_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ####TIMER

# COMMAND ----------

import time

start_time = time.time()
limit = 60
duration = 0
while duration < limit:
    duration = round((time.time() - start_time) / 60, 5)
    print (f"Elapsed time: {duration}")

# COMMAND ----------

###############################
#######  DO NOT DELETE  #######
###############################

# PRE-PROCESS BEFORE PREDICTION
# 1. Reduce repeats of characters from sequences of 3+ to just 2
# 2. Remove combining characters using this function:

import unicodedata

def remove_combining_characters(input_str):
    if input_str is None:
        return None
    # Normalize to NFD (Normalization Form Decomposed)
    normalized_str = unicodedata.normalize('NFD', input_str)
    
    # Filter out characters that are combining marks
    cleaned_str = ''.join(
        char for char in normalized_str 
        if unicodedata.combining(char) == 0
    )
    return cleaned_str

cs = remove_combining_characters("1̶̶̶̶̶̶01𝟴͎8͎ W 다섯222222NDDDD Pƚ UNIɬ too08")    
print (cs)


# Databricks notebook source
# NEXT STEPS:
# 1. Go through entire notebook and understand what it does. Left off at command 7.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

pip install -U accelerate

# COMMAND ----------

# MAGIC %sh
# MAGIC accelerate config

# COMMAND ----------

# Generate test dataset
# 1000 addresses that have been malformed 260 times each (260,000 rows)

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

# Convert test dataset above to pandas and then to a HuggingFace dataset

from datasets import Dataset

test_pdf_cleaned = test_pdf.convert_dtypes().infer_objects().copy()

test_hfdf = Dataset.from_pandas(test_pdf_cleaned, preserve_index=False)
dataset = test_hfdf.train_test_split(test_size=0.1, seed=2112)  # 90/10 split for training/validation


# COMMAND ----------

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")  

def preprocess(example, in_col_name = "malformed_first_line", out_col_name = "first_line"):
    """
    preprocess will take an example and tokenize the input column and the output column as well as create a labels column for the model to learn from.
    """
    
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


# COMMAND ----------

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Print test and train dataset sizes
print(f"Training dataset size: {tokenized_dataset['train'].shape}")
print(f"Test dataset size: {tokenized_dataset['test'].shape}")

# Specifying/separating train and test datasets
train_dataset = dataset["train"]
test_dataset = dataset["test"]


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

# MAGIC %md
# MAGIC ###Generate hard negatives

# COMMAND ----------

# Step 0: Load the tokenizer and model from initial model training

import mlflow
import torch

RUN_ID = "3af87f0aa32d4f7ab0f806aa69d6fb52"

pipe = mlflow.transformers.load_model(
    model_uri=f"runs:/{RUN_ID}/model"
)

type(pipe)

model = pipe.model
tokenizer = pipe.tokenizer

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Sanity checks
print(tokenizer.tokenize("yes"))
print(tokenizer.tokenize("no"))

from transformers import T5ForConditionalGeneration
assert isinstance(model, T5ForConditionalGeneration)



# COMMAND ----------

import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Step 0: Config
# -------------------------------
MAX_NEIGHBORS = 10               # number of nearest correct products to consider
MAX_NOISY_PER_NEIGHBOR = 5       # take at most 5 noisy variants per neighbor
MAX_HARD_NEGATIVES_PER_ROW = 20  # total hard negatives per row


# COMMAND ----------

# -------------------------------
# Step 1: Encode the catalog
# -------------------------------
def encode_catalog(model, tokenizer, catalog, batch_size=64, device="cuda"):
    all_embeddings = []
    for i in tqdm(range(0, len(catalog), batch_size), desc="Encoding catalog"):
        batch = catalog[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            enc_out = model.encoder(**inputs)
            emb = enc_out.last_hidden_state.mean(dim=1)  # mean pooling

        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)

catalog_embeddings = encode_catalog(
    model=model,
    tokenizer=tokenizer,
    catalog=valid_product_names,
    batch_size=128,
    device=device
)


# COMMAND ----------

# -------------------------------
# Step 2: Build nearest-neighbor index
# -------------------------------
nn = NearestNeighbors(n_neighbors=MAX_NEIGHBORS, metric='cosine')
nn.fit(catalog_embeddings)


# COMMAND ----------

# -------------------------------
# Step 3: Map true products to noisy variants
# -------------------------------
true_to_noisy_map = defaultdict(list)
for row in train_dataset:
    true_product = row["first_line"]
    noisy_input = row["malformed_first_line"]
    true_to_noisy_map[true_product].append(noisy_input)


# COMMAND ----------

# -------------------------------
# Step 4: Generate hard negatives
# -------------------------------
# Takes about 22 minutes on a single GPU

hard_negative_map = {}

for idx in tqdm(range(len(train_dataset)), desc="Generating hard negatives"):
    row = train_dataset[idx]
    true_product = row["first_line"]
    noisy_input = row["malformed_first_line"]

    # Step 4a: Nearest neighbors in embedding space
    true_idx = valid_product_names.index(true_product)
    true_emb = catalog_embeddings[true_idx].unsqueeze(0)  # shape [1, hidden_dim]
    distances, neighbor_indices = nn.kneighbors(true_emb)

    # Step 4b: Collect hard negatives
    hard_negatives = []
    for neighbor_idx in neighbor_indices[0]:
        neighbor_product = valid_product_names[neighbor_idx]
        if neighbor_product == true_product:
            continue
        neighbor_noisy_variants = true_to_noisy_map.get(neighbor_product, [])
        hard_negatives.extend(neighbor_noisy_variants[:MAX_NOISY_PER_NEIGHBOR])

    # Limit total hard negatives per row
    hard_negatives = hard_negatives[:MAX_HARD_NEGATIVES_PER_ROW]

    hard_negative_map[idx] = {
        "malformed_first_line": noisy_input,
        "first_line": true_product,
        "hard_negatives": hard_negatives
    }


# COMMAND ----------

# Save hard negative map

import os
import pickle

dbfs_path = "/dbfs/FileStore/hard_negative_map.pkl"

os.makedirs(os.path.dirname(dbfs_path), exist_ok=True)


with open(dbfs_path, "wb") as f:
    pickle.dump(hard_negative_map, f)

print(f"✅ Hard negatives saved to {dbfs_path}")


# COMMAND ----------

# Reload hard negatives
import os
import pickle

with open("/dbfs/FileStore/hard_negative_map.pkl", "rb") as f:
    hard_negative_map = pickle.load(f)

print(f"✅ Loaded {len(hard_negative_map)} rows")
print (f"Sample hard negative:")
print (hard_negative_map[0])


# COMMAND ----------

import time

timer = 20

begin = time.time()
duration = 0
while duration < timer:
    duration = (time.time() - begin)/60
    print (f"Elaped time: {round(duration/60, 5)} minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Retrain model

# COMMAND ----------

from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm

# =========================
# Step 5: Split original rows into train/test
# =========================
rows = list(hard_negative_map.values())

train_rows, test_rows = train_test_split(
    rows,
    test_size=0.1,
    random_state=42
)

print(f"Train rows: {len(train_rows)}, Test rows: {len(test_rows)}")


# COMMAND ----------

# =========================
# Step 6: Tokenization helper function
# =========================
def tokenize_row(row, tokenizer, max_input_len=128):
    tokenized_rows = []

    # Positive pair
    tokenized_rows.append(tokenizer(
        f"match product:\ninput: {row['malformed_first_line']}\ncandidate: {row['first_line']}",
        text_target="yes",
        padding="max_length",
        truncation=True,
        max_length=max_input_len
    ))

    # Hard negatives
    for neg in row["hard_negatives"]:
        tokenized_rows.append(tokenizer(
            f"match product:\ninput: {row['malformed_first_line']}\ncandidate: {neg}",
            text_target="no",
            padding="max_length",
            truncation=True,
            max_length=max_input_len
        ))

    return tokenized_rows


# COMMAND ----------

# =========================
# Step 7: Chunked, disk-persisted tokenization
# =========================

from datasets import Dataset, concatenate_datasets
import gc
from tqdm import tqdm
import os

# -----------------------
# Parameters
# -----------------------
chunk_size = 5000  # number of original rows per chunk
train_save_dir = "/dbfs/FileStore/tokenized_train_chunks"
test_save_dir  = "/dbfs/FileStore/tokenized_test_chunks"

os.makedirs(train_save_dir, exist_ok=True)
os.makedirs(test_save_dir, exist_ok=True)

# -----------------------
# Helper function to tokenize a batch of rows
# -----------------------
def tokenize_batch(rows, tokenizer):
    tokenized_list = []
    for row in rows:
        tokenized_rows = tokenize_row(row, tokenizer)  # your existing function
        tokenized_list.extend(tokenized_rows)
    return tokenized_list



# COMMAND ----------

# =========================
# Step 8: Tokenize test set
# =========================
# Runs in about 30 minutes on single GPU

# PARALLELIZED CODE FROM CHAT GPT
from pyspark.sql import SparkSession
from datasets import Dataset
from transformers import AutoTokenizer
import os
import uuid

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# Make sure output dir exists
os.makedirs(train_save_dir, exist_ok=True)

# Parallelize rows across the cluster
rdd = sc.parallelize(
    train_rows,
    numSlices=sc.defaultParallelism  # uses all cores across nodes
)

def tokenize_partition(rows_iter):
    """
    Runs on Spark workers (not the driver).
    Each partition tokenizes its slice and writes one chunk to disk.
    """
    from transformers import AutoTokenizer
    from datasets import Dataset
    import uuid
    import os

    rows = list(rows_iter)
    if not rows:
        return []

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    tokenized_list = tokenize_batch(rows, tokenizer)
    batch_dataset = Dataset.from_list(tokenized_list)

    chunk_id = str(uuid.uuid4())
    chunk_path = os.path.join(
        "/dbfs/FileStore/tokenized_train_chunks",
        f"train_chunk_{chunk_id}"
    )

    batch_dataset.save_to_disk(chunk_path)

    return [chunk_path]

# Execute distributed tokenization
train_chunk_paths = rdd.mapPartitions(tokenize_partition).collect()

print(f"✅ Tokenization complete. Saved {len(train_chunk_paths)} train chunks.")


# OLD CODE
# test_chunks = []
# num_test_chunks = (len(test_rows) + chunk_size - 1) // chunk_size

# for i in tqdm(range(num_test_chunks), desc="Tokenizing test dataset"):
#     batch_rows = test_rows[i*chunk_size : (i+1)*chunk_size]

#     tokenized_list = tokenize_batch(batch_rows, tokenizer)
#     batch_dataset = Dataset.from_list(tokenized_list)

#     chunk_path = os.path.join(test_save_dir, f"test_chunk_{i}")
#     batch_dataset.save_to_disk(chunk_path)

#     print(f"Saved test chunk {i} to {chunk_path}")

#     del batch_dataset, tokenized_list
#     gc.collect()



# COMMAND ----------

# =========================
# Reload tokenized datasets later (TRAIN and TEST) (runs in about 3 minutes)
# =========================
# Run command 22 first

from datasets import load_from_disk, concatenate_datasets
import os

# Reload train chunks
train_chunk_paths = sorted([os.path.join(train_save_dir, p) 
                            for p in os.listdir(train_save_dir)])
train_datasets = [load_from_disk(p) for p in train_chunk_paths]
tokenized_train_dataset = concatenate_datasets(train_datasets)

# Reload test chunks
test_chunk_paths = sorted([os.path.join(test_save_dir, p) 
                           for p in os.listdir(test_save_dir)])
test_datasets = [load_from_disk(p) for p in test_chunk_paths]
tokenized_test_dataset = concatenate_datasets(test_datasets)

print(f"✅ Loaded {len(tokenized_train_dataset)} train examples")
print(f"✅ Loaded {len(tokenized_test_dataset)} test examples")

# ✅ Loaded 4422600 train examples
# ✅ Loaded 491400 test examples


# COMMAND ----------

tokenized_train_dataset

# COMMAND ----------

# Step 8: Ready for trainer
# Run command 6 first (get timestamp function)
# Run command 8 first (load tokenizer)

from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, pipeline
import mlflow
import mlflow.transformers
from datasets import load_from_disk, concatenate_datasets
import os
from datetime import datetime
import gc
from accelerate import Accelerator

# -----------------------------
# Paths to saved tokenized chunks
# -----------------------------
train_save_dir = "/dbfs/FileStore/tokenized_train_chunks"
test_save_dir  = "/dbfs/FileStore/tokenized_test_chunks"

# -----------------------------
# Reload tokenized train dataset
# -----------------------------
train_chunk_paths = sorted([os.path.join(train_save_dir, p) 
                            for p in os.listdir(train_save_dir)])
train_datasets = [load_from_disk(p) for p in train_chunk_paths]
tokenized_train_dataset = concatenate_datasets(train_datasets)
del train_datasets
gc.collect()

# Reload tokenized test dataset
test_chunk_paths = sorted([os.path.join(test_save_dir, p) 
                           for p in os.listdir(test_save_dir)])
test_datasets = [load_from_disk(p) for p in test_chunk_paths]
tokenized_test_dataset = concatenate_datasets(test_datasets)
del test_datasets
gc.collect()

print(f"✅ Loaded train dataset: {len(tokenized_train_dataset)} examples")
print(f"✅ Loaded test dataset: {len(tokenized_test_dataset)} examples")

# -----------------------------
# Prepare dataset for accelerator
# -----------------------------
accelerator = Accelerator()
tokenized_train_dataset_accelerator, tokenized_test_dataset_accelerator = accelerator.prepare(
    tokenized_train_dataset,
    tokenized_test_dataset
)

# -----------------------------
# Load T5 model and prepare for accelerator
# -----------------------------

model = T5ForConditionalGeneration.from_pretrained("t5-small")
# model.gradient_checkpointing_enable()

model = accelerator.prepare(model)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Mixed precision (fp16) is set in TrainingArguments

# Timestamps / paths — keep HuggingFace Trainer output on DBFS, not under Git Repos (1GB limit).
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
HF_TRAINER_ROOT = "/dbfs/FileStore/address_correction_hf_trainer"
_run_output_dir = os.path.join(HF_TRAINER_ROOT, f"product_corrector_using_hardnegs_{ts}")

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=_run_output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,   # adjust for your GPU
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(_run_output_dir, "logs"),
    fp16=True,
    report_to="none",   # MLFlow logging handled manually
)

# -----------------------------
# Set MLflow experiment
# -----------------------------
# mlflow.set_experiment("/Users/you@example.com/t5_experiments")
run_name = f"t5_product_corrector_using_hardnegs_{ts}"

with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id
    print(f"MLflow run started. Run ID: {run_id}")

    # Add identifying tags
    mlflow.set_tag("dataset_version", "v0.1")
    mlflow.set_tag("dataset_notes", "Using hard negatives. ")
    mlflow.set_tag("model_type", "t5-small")
    mlflow.set_tag("training_notes", "Max neighbors: 10, Max noisy per neighbor: 5, Max hard negatives per row: 20")
    mlflow.set_tag("training_args", str(training_args))

    # -----------------------------
    # Instantiate Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset_accelerator,
        eval_dataset=tokenized_test_dataset_accelerator,
        tokenizer=tokenizer
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # Evaluate and log metrics
    # -----------------------------
    eval_results = trainer.evaluate()
    mlflow.log_metrics(eval_results)
    print("✅ Evaluation metrics logged to MLflow:", eval_results)

    # -----------------------------
    # Log final model + tokenizer
    # -----------------------------
    pipe = pipeline("text2text-generation", model=trainer.model, tokenizer=tokenizer)
    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path="model",
        registered_model_name=f"T5ProductCorrectorRanking_{ts}"
    )

    print(f"✅ Training complete and model logged to MLflow. Run ID: {run_id}")


# COMMAND ----------

# Train/test split for evaluation

from datasets import DatasetDict

# Example: 90% train / 10% test
split = tokenized_ranking_dataset.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
test_ds  = split["test"]

trainer.train_dataset = train_ds
trainer.eval_dataset  = test_ds


# COMMAND ----------

# MAGIC %md
# MAGIC ###Review test dataset

# COMMAND ----------

# Step 1: Prepare the candidate scoring function

import torch
from tqdm import tqdm

def score_candidates(model, tokenizer, noisy_input, candidates, device="cuda"):
    """
    Returns a dict mapping each candidate to the probability that the model outputs 'yes'.
    """
    model.eval()
    scores = {}

    # Batch the candidates for efficiency
    batch_size = 32
    for i in range(0, len(candidates), batch_size):
        batch_candidates = candidates[i:i+batch_size]
        inputs = [f"match product:\ninput: {noisy_input}\ncandidate: {c}" for c in batch_candidates]

        tokenized = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_length=3  # 'yes'/'no' is short
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for cand, pred in zip(batch_candidates, decoded):
            scores[cand] = pred.strip().lower()

    return scores


# COMMAND ----------

# Step 2: Run evaluation on a test dataset

test_results = []

for row in tqdm(test_dataset, desc="Evaluating test set"):
    noisy_input = row["malformed_first_line"]
    true_product = row["first_line"]

    scores = score_candidates(
        model=model,
        tokenizer=tokenizer,
        noisy_input=noisy_input,
        candidates=valid_product_names,  # full catalog
        device=device
    )

    # Pick the top 'yes' candidate
    predicted_product = max(scores.items(), key=lambda x: x[1] == "yes")[0] \
                        if "yes" in scores.values() else None

    test_results.append({
        "input_text": noisy_input,
        "true_text": true_product,
        "predicted_text": predicted_product,
        "is_correct": predicted_product == true_product
    })


# COMMAND ----------

# Step 3: Compute accuracy

num_correct = sum([r["is_correct"] for r in test_results])
accuracy = num_correct / len(test_results)

print(f"Test accuracy: {accuracy:.4f}")


# COMMAND ----------

# Step 4: Optional – Top-K Accuracy

def top_k_accuracy(results, k=3):
    count = 0
    for r in results:
        # Sort candidates: 'yes' first
        # If multiple candidates get 'yes', take top K arbitrarily
        yes_candidates = [c for c, v in r["scores"].items() if v == "yes"]
        if r["true_text"] in yes_candidates[:k]:
            count += 1
    return count / len(results)


# COMMAND ----------

# Join test dataset back to original dataset

# Step 1: Convert the Hugging Face test dataset to a Pandas DataFrame
test_pdf = dataset['test'].to_pandas()

# Step 2: Join model-evaluated test dataset back to original test_pdf dataset
test_pdf_predictions = test_pdf.merge(model_evaluated_test_pdf, on='malformed_first_line', how='left')

# Step 3: Add JW scores for evaluation
test_pdf_predictions_sdf = spark.createDataFrame(test_pdf_predictions)
test_pdf_predictions_sdf = test_pdf_predictions_sdf.withColumn("jw_score", get_jw_score_udf(test_pdf_predictions_sdf.first_line, test_pdf_predictions_sdf.predicted))

test_pdf_predictions_sdf.createOrReplaceTempView("test_predictions")

# Save as table for later querying
s3_path = f"s3://jml-address-validation/tables/model_output/test/{run_name}/{run_name}_output"
test_pdf_predictions_sdf.write.mode("overwrite").saveAsTable(f"model_output.{run_name}_output")
print(f"Table model_output.{run_name}_output saved to {s3_path}")

display(test_pdf_predictions_sdf)

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
     , AVG(jw_score) AS avg_jw
     , COUNT(*) AS record_count
FROM model_output.{run_name}_output
WHERE jw_score <= 1
  AND malformed_first_line != first_line
GROUP BY 1
ORDER BY 2 DESC""")

display(distr_sdf)

# COMMAND ----------


#  Which malform patterns are the hardest for the model to predict?

malform_pattern_dist = spark.sql(f"""
SELECT *
     , ROUND(correct / total_count, 2) AS perc_correct
FROM (     
        SELECT malformed_first_line_malform_steps
            , SUM(CASE WHEN jw_score =  1 THEN 1 ELSE 0 END) AS correct
            , COUNT(*) AS total_count
        FROM model_output.{run_name}_output
        WHERE predicted IS NOT NULL
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
     , jw_score
FROM model_output.{run_name}_output
WHERE malformed_first_line_malform_steps = ', add_special_characters'
ORDER BY jw_score
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


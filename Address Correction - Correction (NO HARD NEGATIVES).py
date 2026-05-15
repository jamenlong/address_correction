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

from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer, pipeline
import mlflow
import mlflow.transformers

model = T5ForConditionalGeneration.from_pretrained("t5-small")

import transformers
print("TrainingArguments location:", TrainingArguments.__module__)
print("Transformers version:", transformers.__version__)

training_args = TrainingArguments(
    output_dir="./product_corrector",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Set MLFlow experiment
mlflow.set_experiment("/Users/jamenlong@yahoo.com/t5_experiments")

# Add identifying information to each run so they can be distinguished
from datetime import datetime
ts = get_timestamp()
run_name = f"t5_product_corrector_training_{ts}"
print (f"Run name: {run_name}")

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
# MAGIC ###Review test dataset

# COMMAND ----------

# Review test dataset
import torch
import pandas as pd
from tqdm import tqdm

# Choose device automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model.to(device)

# Create empty DataFrame for evaluation
model_evaluated_test_pdf = pd.DataFrame(columns = ["malformed_first_line", "predicted", "reference"])

# Select a few examples from the validation set
# eval_samples = dataset["test"].select(range(1000))
eval_samples = dataset["test"]

# Generate predictions
predictions = []
references = []

for example in tqdm(eval_samples):
    input_text = example["malformed_first_line"]
    ref_text = example["first_line"]

    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=32)
    input_ids = input_ids.to(device)
    
    output_ids = model.generate(input_ids, max_length=32, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    predictions.append(pred_text)
    references.append(ref_text)

# Print side-by-side comparison
for malformed, pred, ref in zip(eval_samples["malformed_first_line"], predictions, references):
    # print(f"Malformed:  {malformed}")
    # print(f"Predicted:  {pred}")
    # print(f"Reference:  {ref}")
    # print("-" * 50)
    model_evaluated_test_pdf.loc[len(model_evaluated_test_pdf)] = [malformed, pred, ref]

model_evaluated_test_pdf

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


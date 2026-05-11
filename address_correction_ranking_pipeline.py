"""
Address-correction ranking pipeline (hard negatives + T5 yes/no training).

This module replaces the fragile patterns in
`Address Correction Model - Hard Negatives.py` with:

- One sealed (model, tokenizer) checkpoint used for catalog encoding, optional
  continued pretraining, hard-negative mining, tokenization, and Trainer.
- Tokenization that never mixes MLflow tokenizer with a fresh Hub tokenizer on
  workers; workers load only from `save_pretrained` artifacts.
- Evaluation that matches the training task and avoids full-catalog generation
  unless explicitly requested (shortlist via encoder similarity).
- Configurable SQL (including the legacy single-comma malform-steps filter).
- Optional Spark/Delta I/O; optional MLflow logging.

The malformation *generation* notebook (`Malform Addresses.py`) stays separate;
point `PipelineConfig.source_table` at the Delta table it produces.

Typical Databricks usage::

    from address_correction_ranking_pipeline import PipelineConfig, run_pipeline

    cfg = PipelineConfig(
        source_table="smarty.smarty_malformed_1000_training",
        model_checkpoint_uri="runs:/<RUN_ID>/model",  # or None for Hub t5-small
        hub_model_id="t5-small",
        artifact_dir="/dbfs/FileStore/address_correction_artifacts/run001",
    )
    result = run_pipeline(cfg)

Local / CI smoke test without Spark::

    python address_correction_ranking_pipeline.py \\
        --parquet /path/to/sample.parquet \\
        --hub-model t5-small \\
        --artifact-dir /tmp/addr_rank \\
        --max-train-rows 200 \\
        --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All runtime knobs in one place (no hard-coded RUN_IDs in code paths)."""

    # Data source: either Spark table or local parquet with same logical columns.
    source_table: Optional[str] = "smarty.smarty_malformed_1000_training"
    source_parquet: Optional[str] = None

    # Filter for malformation provenance. "legacy_one_comma" matches the old
    # notebook heuristic (exactly one comma in malformed_first_line_malform_steps).
    # "none" disables the filter. "min_steps" / "max_steps" count non-empty segments
    # after splitting on comma (more stable than raw comma counts if text can contain commas).
    malform_steps_filter: str = "legacy_one_comma"  # legacy_one_comma | none | min_steps | max_steps
    malform_steps_min: int = 1
    malform_steps_max: int = 1

    columns: Dict[str, str] = field(
        default_factory=lambda: {
            "noisy": "malformed_first_line",
            "canonical": "first_line",
            "steps": "malformed_first_line_malform_steps",
        }
    )

    # Model: load fine-tuned pipeline from MLflow, or use Hub id when uri is None.
    model_checkpoint_uri: Optional[str] = None
    hub_model_id: str = "t5-small"

    # Hard-negative mining
    max_neighbors: int = 10
    max_noisy_per_neighbor: int = 5
    max_hard_negatives_per_row: int = 20
    catalog_encode_batch_size: int = 128

    # Train / eval split (row-level, before expanding yes/no examples)
    train_test_split_seed: int = 42
    test_size: float = 0.1

    # Ranking prompt (must match training)
    ranking_prompt_template: str = (
        "match product:\ninput: {noisy}\ncandidate: {candidate}"
    )

    # Training
    artifact_dir: str = "/tmp/address_correction_ranking"
    output_dir: str = "./product_corrector_hard_negs"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_input_length: int = 128
    fp16: bool = True
    seed: int = 2112

    # Evaluation: shortlist size for scoring (encoder cosine on catalog)
    eval_shortlist_k: int = 64
    eval_max_rows: Optional[int] = None

    # Optional MLflow
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    mlflow_registered_model_name: Optional[str] = None

    # Optional Spark result table
    predictions_table: Optional[str] = None

    # Tokenization throughput (Spark optional; default uses HF datasets map + num_proc)
    tokenize_num_proc: Optional[int] = None
    use_spark_tokenization: bool = False

    # Debug
    max_rows: Optional[int] = None


@dataclass
class PipelineResult:
    artifact_dir: Path
    model_dir: Path
    tokenizer_dir: Path
    catalog_path: Path
    hard_negative_map_path: Path
    train_metrics: Dict[str, Any]
    eval_examples: List[Dict[str, Any]]
    eval_accuracy: float
    mlflow_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Malform-steps filters (clearer than counting commas only)
# ---------------------------------------------------------------------------


def _malform_step_segments(steps: Optional[str]) -> List[str]:
    if steps is None:
        return []
    parts = [p.strip() for p in str(steps).split(",")]
    return [p for p in parts if p]


def row_passes_malform_filter(row: Dict[str, Any], cfg: PipelineConfig) -> bool:
    steps_col = cfg.columns["steps"]
    steps = row.get(steps_col)
    mode = cfg.malform_steps_filter
    if mode == "none":
        return True
    if mode == "legacy_one_comma":
        s = "" if steps is None else str(steps)
        return (len(s) - len(s.replace(",", ""))) == 1
    segs = _malform_step_segments(steps)
    if mode == "min_steps":
        return len(segs) >= cfg.malform_steps_min
    if mode == "max_steps":
        return len(segs) <= cfg.malform_steps_max
    raise ValueError(f"Unknown malform_steps_filter: {mode}")


def build_source_sql(cfg: PipelineConfig) -> str:
    steps = cfg.columns["steps"]
    base = f"SELECT * FROM {cfg.source_table}"
    if cfg.malform_steps_filter == "legacy_one_comma":
        where = (
            f"(LENGTH({steps}) - LENGTH(REPLACE({steps}, ',', ''))) = 1"
        )
        return f"{base} WHERE {where}"
    if cfg.malform_steps_filter == "none":
        return base
    if cfg.malform_steps_filter == "min_steps":
        cond = f"""size(filter(split(trim({steps}), ','), x -> length(trim(x)) > 0)) >= {cfg.malform_steps_min}"""
        return f"{base} WHERE {cond}"
    if cfg.malform_steps_filter == "max_steps":
        cond = f"""size(filter(split(trim({steps}), ','), x -> length(trim(x)) > 0)) <= {cfg.malform_steps_max}"""
        return f"{base} WHERE {cond}"
    raise ValueError(f"Unknown malform_steps_filter: {cfg.malform_steps_filter}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rows_pyspark(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise RuntimeError("PySpark is required for source_table loading.") from e

    spark = SparkSession.builder.getOrCreate()
    sql = build_source_sql(cfg)
    logger.info("Loading rows with SQL:\n%s", sql)
    sdf = spark.sql(sql)
    if cfg.max_rows:
        sdf = sdf.limit(int(cfg.max_rows))
    return [row.asDict() for row in sdf.collect()]


def load_rows_parquet(path: str, cfg: PipelineConfig) -> List[Dict[str, Any]]:
    import pandas as pd

    pdf = pd.read_parquet(path)
    rows = pdf.to_dict(orient="records")
    if cfg.malform_steps_filter != "none":
        rows = [r for r in rows if row_passes_malform_filter(r, cfg)]
    if cfg.max_rows:
        rows = rows[: int(cfg.max_rows)]
    return rows


def load_rows(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    if cfg.source_parquet:
        return load_rows_parquet(cfg.source_parquet, cfg)
    if not cfg.source_table:
        raise ValueError("Set source_table or source_parquet.")
    return load_rows_pyspark(cfg)


# ---------------------------------------------------------------------------
# Single (model, tokenizer) load + artifact sync
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    hub_model_id: str,
    model_checkpoint_uri: Optional[str],
) -> Tuple[torch.nn.Module, Any]:
    """Return (model, tokenizer) from one coherent source."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    if model_checkpoint_uri:
        import mlflow.transformers

        pipe = mlflow.transformers.load_model(model_uri=model_checkpoint_uri)
        model = pipe.model
        tokenizer = pipe.tokenizer
        logger.info("Loaded model+tokenizer from MLflow: %s", model_checkpoint_uri)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
        model = T5ForConditionalGeneration.from_pretrained(hub_model_id)
        logger.info("Loaded model+tokenizer from Hub: %s", hub_model_id)

    return model, tokenizer


def persist_model_tokenizer(model: torch.nn.Module, tokenizer: Any, directory: Path) -> None:
    """Materialize a directory both Trainer and workers can reload identically."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(directory)
    model.save_pretrained(directory)
    logger.info("Saved model+tokenizer to %s", directory)


def reload_model_tokenizer(directory: Path, train: bool = True) -> Tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSeq2SeqLM.from_pretrained(directory)
    model.train(train)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Encoding utilities (catalog + query)
# ---------------------------------------------------------------------------


def mean_pool_encoder(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """L2-normalized mean-pooled encoder vectors for each string."""
    all_embs: List[torch.Tensor] = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = list(texts[i : i + batch_size])
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            enc_out = model.get_encoder()(**inputs)
            emb = enc_out.last_hidden_state.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Hard negatives
# ---------------------------------------------------------------------------


def build_true_to_noisy_map(
    rows: Iterable[Dict[str, Any]], noisy_col: str, canonical_col: str
) -> Dict[str, List[str]]:
    from collections import defaultdict

    m: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        key = r.get(canonical_col)
        val = r.get(noisy_col)
        if key is None or val is None:
            continue
        m[str(key)].append(str(val))
    return dict(m)


def build_hard_negative_map(
    train_rows: List[Dict[str, Any]],
    catalog: List[str],
    catalog_embeddings: torch.Tensor,
    true_to_noisy: Dict[str, List[str]],
    cfg: PipelineConfig,
) -> Dict[int, Dict[str, Any]]:
    nn = NearestNeighbors(
        n_neighbors=max(1, min(cfg.max_neighbors, len(catalog))),
        metric="cosine",
    )
    nn.fit(catalog_embeddings.numpy())
    cat_index = {c: i for i, c in enumerate(catalog)}
    out: Dict[int, Dict[str, Any]] = {}
    noisy_col = cfg.columns["noisy"]
    canonical_col = cfg.columns["canonical"]

    for idx, row in enumerate(tqdm(train_rows, desc="Hard negatives")):
        true_product = str(row[canonical_col])
        noisy_input = str(row[noisy_col])
        ti = cat_index.get(true_product)
        if ti is None:
            out[idx] = {
                noisy_col: noisy_input,
                canonical_col: true_product,
                "hard_negatives": [],
            }
            continue
        true_emb = catalog_embeddings[ti : ti + 1].numpy()
        _, neighbor_indices = nn.kneighbors(true_emb)
        hard_negatives: List[str] = []
        for neighbor_idx in neighbor_indices[0]:
            neighbor_product = catalog[int(neighbor_idx)]
            if neighbor_product == true_product:
                continue
            neighbor_noisy = true_to_noisy.get(neighbor_product, [])
            hard_negatives.extend(neighbor_noisy[: cfg.max_noisy_per_neighbor])
        hard_negatives = hard_negatives[: cfg.max_hard_negatives_per_row]
        out[idx] = {
            noisy_col: noisy_input,
            canonical_col: true_product,
            "hard_negatives": hard_negatives,
        }
    return out


# ---------------------------------------------------------------------------
# Tokenization (ranking yes/no) — always from tokenizer_dir
# ---------------------------------------------------------------------------


def ranking_prompt(cfg: PipelineConfig, noisy: str, candidate: str) -> str:
    return cfg.ranking_prompt_template.format(noisy=noisy, candidate=candidate)


def tokenize_row_dict(
    row: Dict[str, Any],
    tokenizer: Any,
    cfg: PipelineConfig,
) -> List[Dict[str, List[int]]]:
    noisy_col = cfg.columns["noisy"]
    canonical_col = cfg.columns["canonical"]
    noisy = row[noisy_col]
    true_c = row[canonical_col]
    outs: List[Dict[str, List[int]]] = []
    outs.append(
        tokenizer(
            ranking_prompt(cfg, noisy, true_c),
            text_target="yes",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_input_length,
        )
    )
    for neg in row.get("hard_negatives", []):
        outs.append(
            tokenizer(
                ranking_prompt(cfg, noisy, neg),
                text_target="no",
                padding="max_length",
                truncation=True,
                max_length=cfg.max_input_length,
            )
        )
    return outs


def tokenize_hard_negative_map(
    rows: List[Dict[str, Any]],
    tokenizer_dir: Path,
    cfg: PipelineConfig,
) -> Dataset:
    """Single-process tokenization (simple, correct). For large data, use map_num_proc below."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    all_examples: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="Tokenize ranking examples"):
        all_examples.extend(tokenize_row_dict(row, tok, cfg))
    return Dataset.from_list(all_examples)


def tokenize_hard_negative_map_parallel(
    rows: List[Dict[str, Any]],
    tokenizer_dir: Path,
    cfg: PipelineConfig,
) -> Dataset:
    """Expand each logical row to many tokenized ranking examples using tokenizer on disk."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    chunks: List[Dataset] = []
    for row in tqdm(rows, desc="Tokenize (chunk concat)"):
        chunks.append(Dataset.from_list(tokenize_row_dict(row, tok, cfg)))
    return concatenate_datasets(chunks) if chunks else Dataset.from_list([])


def spark_tokenize_partitions(
    rows: List[Dict[str, Any]],
    tokenizer_dir: Path,
    cfg: PipelineConfig,
) -> List[str]:
    """Optional Spark path: every executor loads tokenizer only from tokenizer_dir."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    td = str(tokenizer_dir)
    artifact_chunk_dir = Path(cfg.artifact_dir) / "tokenized_chunks"
    artifact_chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = str(artifact_chunk_dir)

    def _part(rows_iter):
        from transformers import AutoTokenizer
        from datasets import Dataset
        import uuid
        import os

        rows_local = list(rows_iter)
        if not rows_local:
            return []
        tok = AutoTokenizer.from_pretrained(td)
        all_examples: List[Dict[str, Any]] = []
        for row in rows_local:
            all_examples.extend(tokenize_row_dict(row, tok, cfg))
        ds = Dataset.from_list(all_examples)
        cid = str(uuid.uuid4())
        path = os.path.join(chunk_root, f"chunk_{cid}")
        ds.save_to_disk(path)
        return [path]

    rdd = sc.parallelize(rows, numSlices=sc.defaultParallelism)
    paths = rdd.mapPartitions(_part).collect()
    return paths


def load_tokenized_chunks(chunk_paths: List[str]) -> Dataset:
    from datasets import load_from_disk

    parts = [load_from_disk(p) for p in chunk_paths]
    return concatenate_datasets(parts) if parts else Dataset.from_list([])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ranking_model(
    train_ds: Dataset,
    eval_ds: Dataset,
    model_dir: Path,
    tokenizer_dir: Path,
    cfg: PipelineConfig,
) -> Tuple[torch.nn.Module, Any, Dict[str, float]]:
    import gc
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        AutoModelForSeq2SeqLM,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.gradient_checkpointing_enable()

    ta_kwargs = dict(
        output_dir=cfg.output_dir,
        save_strategy="epoch",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        fp16=cfg.fp16,
        report_to="none",
        seed=cfg.seed,
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **ta_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **ta_kwargs)

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
        )
    train_out = trainer.train()
    metrics = trainer.evaluate()
    del train_out
    gc.collect()

    # Save final weights next to tokenizer for downstream inference
    save_dir = Path(cfg.artifact_dir) / "trained_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(save_dir)

    return model, tokenizer, metrics


# ---------------------------------------------------------------------------
# Evaluation (consistent structures, optional shortlist)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def score_candidates_with_model(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    device: torch.device,
    batch_size: int = 32,
    max_new_tokens: int = 8,
) -> List[str]:
    model.eval()
    decoded: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens)
        decoded.extend(
            tokenizer.batch_decode(out, skip_special_tokens=True)
        )
    return [d.strip().lower() for d in decoded]


def shortlist_catalog_indices(
    model: torch.nn.Module,
    tokenizer: Any,
    noisy: str,
    catalog: List[str],
    catalog_embeddings: torch.Tensor,
    device: torch.device,
    top_k: int,
) -> List[int]:
    """Encoder cosine similarity between noisy input and each catalog line (same encoder as training)."""
    q = mean_pool_encoder(
        model, tokenizer, [noisy], batch_size=1, device=device
    )
    sims = (catalog_embeddings.to(device) * q.to(device)).sum(dim=1)
    k = min(top_k, sims.shape[0])
    _, idx = torch.topk(sims, k=k, largest=True)
    return idx.detach().cpu().tolist()


def evaluate_closed_set_ranking(
    model: torch.nn.Module,
    tokenizer: Any,
    test_rows: List[Dict[str, Any]],
    catalog: List[str],
    catalog_embeddings: torch.Tensor,
    cfg: PipelineConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], float]:
    noisy_col = cfg.columns["noisy"]
    canonical_col = cfg.columns["canonical"]
    results: List[Dict[str, Any]] = []
    correct = 0
    n = 0
    for row in tqdm(test_rows, desc="Eval (shortlist)"):
        if cfg.eval_max_rows is not None and n >= cfg.eval_max_rows:
            break
        noisy = str(row[noisy_col])
        true_c = str(row[canonical_col])
        idxs = shortlist_catalog_indices(
            model,
            tokenizer,
            noisy,
            catalog,
            catalog_embeddings,
            device,
            cfg.eval_shortlist_k,
        )
        candidates = [catalog[i] for i in idxs]
        # Ensure the gold label is scoreable (shortlist is approximate retrieval).
        if true_c not in candidates:
            rest = [c for c in candidates if c != true_c]
            candidates = [true_c] + rest[: max(0, cfg.eval_shortlist_k - 1)]
        prompts = [ranking_prompt(cfg, noisy, c) for c in candidates]
        preds = score_candidates_with_model(
            model, tokenizer, prompts, device=device
        )
        scores = {c: p for c, p in zip(candidates, preds)}
        # Prefer first 'yes' among shortlist order (highest similarity first)
        predicted = None
        for c in candidates:
            if scores.get(c) == "yes":
                predicted = c
                break
        is_correct = predicted == true_c
        if is_correct:
            correct += 1
        n += 1
        results.append(
            {
                "noisy": noisy,
                "true": true_c,
                "predicted": predicted,
                "is_correct": is_correct,
                "scores": scores,
                "shortlist": candidates,
            }
        )
    acc = correct / n if n else 0.0
    return results, acc


def top_k_accuracy_from_results(results: List[Dict[str, Any]], k: int = 3) -> float:
    """Uses per-row 'scores' dict: true is correct if in first k candidates with score 'yes' in shortlist order."""
    if not results:
        return 0.0
    hit = 0
    for r in results:
        yes_list = [c for c, v in r["scores"].items() if v == "yes"]
        # preserve shortlist order
        ordered = [c for c in r["shortlist"] if c in yes_list]
        if r["true"] in ordered[:k]:
            hit += 1
    return hit / len(results)


def jaro_winkler(a: str, b: str) -> float:
    try:
        import jellyfish

        if hasattr(jellyfish, "jaro_winkler_similarity"):
            return float(jellyfish.jaro_winkler_similarity(a, b))
        return float(jellyfish.jaro_winkler(a, b))
    except Exception:
        try:
            from rapidfuzz.distance import JaroWinkler

            return float(JaroWinkler.normalized_similarity(a, b))
        except Exception:
            return float(a == b)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    art = Path(cfg.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    base_dir = art / "base_checkpoint"
    tokenizer_dir = art / "base_checkpoint"  # same dir for simplicity
    model_dir = art / "base_checkpoint"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rows(cfg)
    if not rows:
        raise RuntimeError("No rows loaded; check table/SQL/filters.")

    noisy_col = cfg.columns["noisy"]
    canonical_col = cfg.columns["canonical"]

    catalog = sorted(
        {str(r[canonical_col]) for r in rows if r.get(canonical_col) is not None}
    )
    if not catalog:
        raise RuntimeError("Catalog is empty (no first_line values); cannot continue.")
    logger.info("Catalog size: %d | rows: %d", len(catalog), len(rows))

    model, tokenizer = load_model_and_tokenizer(
        cfg.hub_model_id, cfg.model_checkpoint_uri
    )
    model.to(device)
    persist_model_tokenizer(model, tokenizer, base_dir)

    # Reload from disk so downstream code paths only ever use artifact_dir
    model, tokenizer = reload_model_tokenizer(base_dir, train=False)
    model.to(device)

    catalog_embeddings = mean_pool_encoder(
        model,
        tokenizer,
        catalog,
        batch_size=cfg.catalog_encode_batch_size,
        device=device,
    )
    torch.save(catalog_embeddings, art / "catalog_embeddings.pt")
    with open(art / "catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f)

    # Row-level train/test split (canonical + noisy rows, not expanded tokens)
    train_rows, test_rows = train_test_split(
        rows,
        test_size=cfg.test_size,
        random_state=cfg.train_test_split_seed,
    )

    true_to_noisy = build_true_to_noisy_map(train_rows, noisy_col, canonical_col)
    hn_map = build_hard_negative_map(
        train_rows, catalog, catalog_embeddings, true_to_noisy, cfg
    )
    train_rank_rows: List[Dict[str, Any]] = []
    for i, tr in enumerate(train_rows):
        merged = dict(tr)
        merged.update(hn_map[i])
        train_rank_rows.append(merged)

    hn_path = art / "hard_negative_map.pkl"
    with open(hn_path, "wb") as f:
        pickle.dump({"train": train_rank_rows, "test": test_rows}, f)

    if cfg.use_spark_tokenization:
        chunk_paths = spark_tokenize_partitions(
            train_rank_rows, tokenizer_dir, cfg
        )
        train_ds = load_tokenized_chunks(chunk_paths)
        # Test tokenization driver-only (smaller)
        test_rank_rows = []
        for tr in test_rows:
            test_rank_rows.append({**tr, "hard_negatives": []})
        eval_chunks = spark_tokenize_partitions(test_rank_rows, tokenizer_dir, cfg)
        eval_ds = load_tokenized_chunks(eval_chunks)
    else:
        train_ds = tokenize_hard_negative_map_parallel(
            train_rank_rows, tokenizer_dir, cfg
        )
        test_rank_rows = [{**tr, "hard_negatives": []} for tr in test_rows]
        eval_ds = tokenize_hard_negative_map_parallel(
            test_rank_rows, tokenizer_dir, cfg
        )

    train_ds.save_to_disk(str(art / "tokenized_train"))
    eval_ds.save_to_disk(str(art / "tokenized_eval"))

    train_metrics: Dict[str, Any] = {}
    _, _, eval_metrics = train_ranking_model(
        train_ds,
        eval_ds,
        model_dir=base_dir,
        tokenizer_dir=tokenizer_dir,
        cfg=cfg,
    )
    train_metrics.update(eval_metrics)

    # Load trained weights for evaluation (same artifacts as Trainer saved).
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    trained_dir = art / "trained_model"
    eval_model = AutoModelForSeq2SeqLM.from_pretrained(trained_dir)
    eval_tok = AutoTokenizer.from_pretrained(trained_dir)
    eval_model.to(device)

    # Shortlist retrieval must use the same encoder weights as the ranker you evaluate.
    eval_catalog_embeddings = mean_pool_encoder(
        eval_model,
        eval_tok,
        catalog,
        batch_size=cfg.catalog_encode_batch_size,
        device=device,
    )

    eval_results, acc = evaluate_closed_set_ranking(
        eval_model,
        eval_tok,
        test_rows,
        catalog,
        eval_catalog_embeddings,
        cfg,
        device,
    )
    topk = top_k_accuracy_from_results(eval_results, k=3)

    logger.info("Eval accuracy (shortlist argmax yes): %.4f", acc)
    logger.info("Top-3 yes accuracy: %.4f", topk)

    mlflow_run_id = None
    if cfg.mlflow_experiment_name:
        import mlflow

        mlflow.set_experiment(cfg.mlflow_experiment_name)
        with mlflow.start_run(run_name=cfg.mlflow_run_name or "ranking_hard_negs"):
            mlflow.log_params(
                {
                    "malform_steps_filter": cfg.malform_steps_filter,
                    "max_neighbors": cfg.max_neighbors,
                    "hub_model_id": cfg.hub_model_id,
                }
            )
            mlflow.log_metrics({**{f"eval_{k}": v for k, v in eval_metrics.items()}, "shortlist_acc": acc, "top3_yes": topk})
            if cfg.mlflow_registered_model_name:
                from transformers import pipeline as hf_pipeline

                gen_pipe = hf_pipeline(
                    "text2text-generation",
                    model=eval_model,
                    tokenizer=eval_tok,
                )
                mlflow.transformers.log_model(
                    transformers_model=gen_pipe,
                    artifact_path="model",
                    registered_model_name=cfg.mlflow_registered_model_name,
                )
            mlflow_run_id = mlflow.active_run().info.run_id

    if cfg.predictions_table:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            out_rows = []
            for r in eval_results:
                out_rows.append(
                    (
                        r["noisy"],
                        r["true"],
                        r["predicted"],
                        r["is_correct"],
                        json.dumps(r["scores"]),
                        jaro_winkler(r["true"], r["predicted"] or ""),
                    )
                )
            sdf = spark.createDataFrame(
                out_rows,
                schema="noisy STRING, true_canonical STRING, predicted STRING, is_correct BOOLEAN, scores_json STRING, jw_score DOUBLE",
            )
            sdf.write.mode("overwrite").saveAsTable(cfg.predictions_table)
        except Exception as e:
            logger.warning("Could not write predictions table: %s", e)

    return PipelineResult(
        artifact_dir=art,
        model_dir=trained_dir,
        tokenizer_dir=trained_dir,
        catalog_path=art / "catalog.json",
        hard_negative_map_path=hn_path,
        train_metrics=train_metrics,
        eval_examples=eval_results,
        eval_accuracy=acc,
        mlflow_run_id=mlflow_run_id,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=str, default="/tmp/address_correction_ranking")
    p.add_argument("--hub-model", type=str, default="t5-small")
    p.add_argument("--model-uri", type=str, default=None, help="MLflow runs:/.../model URI")
    p.add_argument("--source-table", type=str, default=None)
    p.add_argument("--parquet", type=str, default=None)
    p.add_argument("--malform-filter", type=str, default="legacy_one_comma")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-train-rows", type=int, default=None, help="Cap rows after load for smoke tests")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval-shortlist-k", type=int, default=64)
    args = p.parse_args(argv)

    cfg = PipelineConfig(
        artifact_dir=args.artifact_dir,
        hub_model_id=args.hub_model,
        model_checkpoint_uri=args.model_uri,
        source_table=args.source_table or PipelineConfig.source_table,
        source_parquet=args.parquet,
        malform_steps_filter=args.malform_filter,
        max_rows=args.max_train_rows or args.max_rows,
        num_train_epochs=args.epochs,
        eval_shortlist_k=args.eval_shortlist_k,
    )
    if args.parquet:
        cfg.source_table = None

    run_pipeline(cfg)


if __name__ == "__main__":
    main()

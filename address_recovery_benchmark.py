"""
Compare malform recovery against a saved T5 benchmark table (model_output.*_output).

Use the same JW columns and filters as the correction notebook so you can see,
per row and per malform step, where hybrid recovery helps or hurts vs the baseline.

Typical baseline:
  model_output.t5_product_corrector_training_20may2026_16_07_06_output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from address_malform_recovery import (
    CatalogIndex,
    RecoveryConfig,
    apply_hybrid_override,
    build_inverse_char_map,
    jaro_winkler,
    normalize_address,
    recover_address,
)

# Default T5 benchmark from your 20 May 2026 run
DEFAULT_BASELINE_TABLE = (
    "model_output.t5_product_corrector_training_20may2026_16_07_06_output"
)

STANDARD_EVAL_WHERE = """
    predicted IS NOT NULL
    AND malformed_first_line != first_line
    AND jw_input <= 1
    AND jw_output <= 1
"""


@dataclass
class BenchmarkTableConfig:
    """Column names and table id for a model_output.*_output benchmark."""

    baseline_table: str = DEFAULT_BASELINE_TABLE
    malformed_col: str = "malformed_first_line"
    true_col: str = "first_line"
    t5_pred_col: str = "predicted"
    steps_col: str = "malformed_first_line_malform_steps"
    jw_input_col: str = "jw_input"
    jw_output_col: str = "jw_output"
    eval_where_sql: str = field(
        default_factory=lambda: STANDARD_EVAL_WHERE.strip().replace("\n", " ")
    )

    def output_table(self, suffix: str = "_recovery_eval") -> str:
        base = self.baseline_table.split(".")[-1]
        if base.endswith("_output"):
            base = base[: -len("_output")]
        return f"model_output.{base}{suffix}"

    def baseline_select_sql(
        self,
        extra_where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        where = self.eval_where_sql
        if extra_where:
            where = f"({where}) AND ({extra_where})"
        lim = f"\nLIMIT {int(limit)}" if limit else ""
        return f"""
SELECT *
FROM {self.baseline_table}
WHERE {where}{lim}
"""


def jw_improvement_metrics(jw_input: float, jw_output: float) -> Dict[str, Any]:
    """Mirror correction notebook: jw_delta, jw_relative_gain, jw_improved, exact."""
    jw_delta = float(jw_output) - float(jw_input)
    if jw_input < 1.0:
        relative_gain = jw_delta / (1.0 - float(jw_input))
    else:
        relative_gain = None
    return {
        "jw_delta": jw_delta,
        "jw_relative_gain": relative_gain,
        "jw_improved": jw_delta > 0.0,
        "jw_exact": float(jw_output) >= 1.0,
    }


def _jw_or_compute(
    a: str, b: str, existing: Optional[float] = None
) -> float:
    if existing is not None and existing <= 1.0:
        return float(existing)
    return jaro_winkler(normalize_address(a), normalize_address(b))


def enrich_row_with_recovery(
    row: Dict[str, Any],
    catalog: Sequence[str],
    char_malform_dct: Dict[str, Sequence[str]],
    recovery_config: Optional[RecoveryConfig] = None,
    bench: Optional[BenchmarkTableConfig] = None,
    *,
    catalog_index: Optional[CatalogIndex] = None,
    inverse_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Add recovery / hybrid JW metrics on top of baseline T5 columns in ``row``.
    """
    cfg_b = bench or BenchmarkTableConfig()
    rcfg = recovery_config or RecoveryConfig()

    malformed = str(row.get(cfg_b.malformed_col) or "")
    truth = str(row.get(cfg_b.true_col) or "")
    t5_pred = str(row.get(cfg_b.t5_pred_col) or "")
    steps = row.get(cfg_b.steps_col)

    jw_in = _jw_or_compute(
        malformed, truth, row.get(cfg_b.jw_input_col)
    )
    jw_t5 = _jw_or_compute(
        t5_pred, truth, row.get(cfg_b.jw_output_col)
    )
    t5_metrics = jw_improvement_metrics(jw_in, jw_t5)

    rec = recover_address(
        malformed,
        catalog,
        char_malform_dct,
        malform_steps=steps,
        config=rcfg,
        catalog_index=catalog_index,
        inverse_map=inverse_map,
    )
    hybrid_pred = apply_hybrid_override(t5_pred, truth, rec, rcfg)

    rec_match = rec.catalog_match or ""
    jw_rec_only = (
        jaro_winkler(normalize_address(rec_match), normalize_address(truth))
        if rec_match
        else 0.0
    )
    jw_hybrid = jaro_winkler(
        normalize_address(hybrid_pred), normalize_address(truth)
    )
    hybrid_metrics = jw_improvement_metrics(jw_in, jw_hybrid)

    t5_exact = normalize_address(t5_pred) == normalize_address(truth)
    rec_exact = normalize_address(rec_match) == normalize_address(truth)
    hybrid_exact = normalize_address(hybrid_pred) == normalize_address(truth)

    jw_delta_vs_t5 = jw_hybrid - jw_t5
    recovered_over_t5 = normalize_address(hybrid_pred) != normalize_address(t5_pred)

    out = dict(row)
    out.update(
        {
            "recovery_enabled_steps": ",".join(sorted(rcfg.enabled_steps)),
            "recovery_recovered_text": rec.recovered_text,
            "recovery_catalog_match": rec_match,
            "recovery_catalog_jw": rec.catalog_jw,
            "recovery_used": rec.used_recovery,
            "recovery_applied_steps": ",".join(rec.applied_steps),
            "hybrid_predicted": hybrid_pred,
            "jw_output_recovery_only": jw_rec_only,
            "jw_output_hybrid": jw_hybrid,
            "jw_delta_hybrid": hybrid_metrics["jw_delta"],
            "jw_relative_gain_hybrid": hybrid_metrics["jw_relative_gain"],
            "jw_improved_hybrid": hybrid_metrics["jw_improved"],
            "jw_exact_hybrid": hybrid_exact,
            "jw_delta_vs_t5": jw_delta_vs_t5,
            "jw_improved_vs_t5": jw_delta_vs_t5 > 0.0,
            "t5_exact": t5_exact,
            "recovery_exact": rec_exact,
            "hybrid_exact": hybrid_exact,
            "recovery_fixed_t5_miss": hybrid_exact and not t5_exact,
            "recovery_broke_t5_hit": t5_exact and not hybrid_exact,
            "recovery_changed_prediction": normalize_address(hybrid_pred)
            != normalize_address(t5_pred),
            "prediction_source": "recovery"
            if recovered_over_t5
            else "t5",
            # Baseline snapshot (explicit names for downstream SQL)
            "benchmark_jw_input": jw_in,
            "benchmark_jw_output_t5": jw_t5,
            "benchmark_jw_delta_t5": t5_metrics["jw_delta"],
        }
    )
    return out


def aggregate_benchmark_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    group_col: str = "malformed_first_line_malform_steps",
) -> Dict[str, Any]:
    """
    Overall + per-malform-step summary vs T5 baseline (hybrid = recovery + override).
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    all_rows: List[Dict[str, Any]] = []
    for row in rows:
        all_rows.append(row)
        key = str(row.get(group_col) or "")
        buckets.setdefault(key, []).append(row)

    def _agg(name: str, subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {"group": name, "n": 0}

        def avg(field: str) -> float:
            vals = [float(r[field]) for r in subset if r.get(field) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        def pct(flag: str) -> float:
            return 100.0 * sum(1 for r in subset if r.get(flag)) / n

        return {
            "group": name,
            "n": n,
            "avg_jw_input": round(avg("benchmark_jw_input"), 4),
            "avg_jw_output_t5": round(avg("benchmark_jw_output_t5"), 4),
            "avg_jw_output_hybrid": round(avg("jw_output_hybrid"), 4),
            "avg_jw_delta_t5": round(avg("benchmark_jw_delta_t5"), 4),
            "avg_jw_delta_hybrid": round(avg("jw_delta_hybrid"), 4),
            "avg_jw_delta_vs_t5": round(avg("jw_delta_vs_t5"), 4),
            "pct_improved_vs_t5": round(pct("jw_improved_vs_t5"), 2),
            "pct_exact_t5": round(pct("t5_exact"), 2),
            "pct_exact_hybrid": round(pct("hybrid_exact"), 2),
            "pct_exact_gain": round(
                pct("hybrid_exact") - pct("t5_exact"), 2
            ),
            "recovery_fixed_t5_miss": sum(
                1 for r in subset if r.get("recovery_fixed_t5_miss")
            ),
            "recovery_broke_t5_hit": sum(
                1 for r in subset if r.get("recovery_broke_t5_hit")
            ),
            "recovery_changed_prediction": sum(
                1 for r in subset if r.get("recovery_changed_prediction")
            ),
        }

    overall = _agg("__overall__", all_rows)
    by_step = [_agg(k, v) for k, v in sorted(buckets.items(), key=lambda x: -len(x[1]))]
    return {"overall": overall, "by_malform_step": by_step}


def benchmark_summary_sql(
    eval_table: str,
    *,
    only_enabled_steps: bool = False,
) -> Dict[str, str]:
    """
  SQL snippets for Databricks display (same spirit as correction notebook JW cells).
    """
    step_filter = ""
    if only_enabled_steps:
        step_filter = " AND recovery_applied_steps != ''"

    overall = f"""
SELECT COUNT(*) AS n_rows
     , ROUND(AVG(benchmark_jw_input), 4) AS avg_jw_input
     , ROUND(AVG(benchmark_jw_output_t5), 4) AS avg_jw_output_t5
     , ROUND(AVG(jw_output_hybrid), 4) AS avg_jw_output_hybrid
     , ROUND(AVG(benchmark_jw_delta_t5), 4) AS avg_jw_delta_t5
     , ROUND(AVG(jw_delta_hybrid), 4) AS avg_jw_delta_hybrid
     , ROUND(AVG(jw_delta_vs_t5), 4) AS avg_jw_delta_vs_t5
     , ROUND(100.0 * SUM(CASE WHEN jw_improved_hybrid THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_improved_hybrid
     , ROUND(100.0 * SUM(CASE WHEN jw_improved_vs_t5 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_improved_vs_t5
     , ROUND(100.0 * SUM(CASE WHEN hybrid_exact THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_exact_hybrid
     , ROUND(100.0 * SUM(CASE WHEN t5_exact THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_exact_t5
     , SUM(CASE WHEN recovery_fixed_t5_miss THEN 1 ELSE 0 END) AS fixed_t5_miss
     , SUM(CASE WHEN recovery_broke_t5_hit THEN 1 ELSE 0 END) AS broke_t5_hit
FROM {eval_table}
WHERE 1=1{step_filter}
"""

    by_step = f"""
SELECT malformed_first_line_malform_steps
     , COUNT(*) AS n_rows
     , ROUND(AVG(benchmark_jw_output_t5), 4) AS avg_jw_output_t5
     , ROUND(AVG(jw_output_hybrid), 4) AS avg_jw_output_hybrid
     , ROUND(AVG(jw_delta_vs_t5), 4) AS avg_jw_delta_vs_t5
     , ROUND(100.0 * SUM(CASE WHEN t5_exact THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_exact_t5
     , ROUND(100.0 * SUM(CASE WHEN hybrid_exact THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_exact_hybrid
     , ROUND(
           100.0 * SUM(CASE WHEN hybrid_exact THEN 1 ELSE 0 END) / COUNT(*)
         - 100.0 * SUM(CASE WHEN t5_exact THEN 1 ELSE 0 END) / COUNT(*),
           2
       ) AS pct_exact_gain
     , SUM(CASE WHEN recovery_fixed_t5_miss THEN 1 ELSE 0 END) AS fixed_t5_miss
     , SUM(CASE WHEN recovery_broke_t5_hit THEN 1 ELSE 0 END) AS broke_t5_hit
FROM {eval_table}
WHERE 1=1{step_filter}
GROUP BY 1
ORDER BY avg_jw_delta_vs_t5 DESC
"""

    regressions = f"""
SELECT malformed_first_line
     , first_line
     , predicted AS t5_predicted
     , hybrid_predicted
     , malformed_first_line_malform_steps
     , benchmark_jw_output_t5
     , jw_output_hybrid
     , jw_delta_vs_t5
FROM {eval_table}
WHERE recovery_broke_t5_hit
ORDER BY jw_delta_vs_t5 ASC
LIMIT 200
"""

    wins = f"""
SELECT malformed_first_line
     , first_line
     , predicted AS t5_predicted
     , hybrid_predicted
     , malformed_first_line_malform_steps
     , benchmark_jw_output_t5
     , jw_output_hybrid
     , jw_delta_vs_t5
FROM {eval_table}
WHERE recovery_fixed_t5_miss
ORDER BY jw_delta_vs_t5 DESC
LIMIT 200
"""

    return {
        "overall": overall,
        "by_malform_step": by_step,
        "wins": wins,
        "regressions": regressions,
    }


def run_recovery_benchmark_pandas(
    baseline_rows: List[Dict[str, Any]],
    catalog: Sequence[str],
    char_malform_dct: Dict[str, Sequence[str]],
    recovery_config: Optional[RecoveryConfig] = None,
    bench: Optional[BenchmarkTableConfig] = None,
    progress_every: int = 50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Enrich baseline rows and return (enriched_rows, aggregate_report)."""
    import time

    n_rows = len(baseline_rows)
    print(f"  building catalog index ({len(catalog)} lines)...")
    catalog_index = CatalogIndex.build(catalog)
    print(f"  building inverse char map...")
    inverse_map = build_inverse_char_map(char_malform_dct)

    enriched: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, row in enumerate(baseline_rows):
        enriched.append(
            enrich_row_with_recovery(
                row,
                catalog,
                char_malform_dct,
                recovery_config=recovery_config,
                bench=bench,
                catalog_index=catalog_index,
                inverse_map=inverse_map,
            )
        )
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_rows - i - 1) / rate if rate > 0 else 0.0
            print(
                f"  enriched {i + 1}/{n_rows} rows "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )
    report = aggregate_benchmark_rows(enriched)
    return enriched, report


def run_recovery_benchmark_spark(
    spark,
    char_malform_dct: Dict[str, Sequence[str]],
    recovery_config: Optional[RecoveryConfig] = None,
    bench: Optional[BenchmarkTableConfig] = None,
    *,
    output_table: Optional[str] = None,
    extra_where: Optional[str] = None,
    limit: Optional[int] = None,
    catalog: Optional[Sequence[str]] = None,
    write_mode: str = "overwrite",
) -> Any:
    """
    Load T5 benchmark Delta table, run recovery, write eval table, return Spark DF.

    Requires PySpark (Databricks). Catalog defaults to distinct ``first_line`` in
    the baseline table (same pool T5 was constrained to during eval).
    """
    cfg = bench or BenchmarkTableConfig()
    rcfg = recovery_config or RecoveryConfig()
    out_table = output_table or cfg.output_table()

    sql = cfg.baseline_select_sql(extra_where=extra_where, limit=limit)
    baseline_sdf = spark.sql(sql)
    n = baseline_sdf.count()
    print(f"Baseline rows (eval filter): {n} from {cfg.baseline_table}")

    if catalog is None:
        catalog = [
            r[0]
            for r in spark.sql(
                f"""
                SELECT DISTINCT {cfg.true_col}
                FROM {cfg.baseline_table}
                WHERE {cfg.eval_where_sql}
                """
            ).collect()
        ]
    print(f"Catalog size: {len(catalog)} distinct {cfg.true_col}")

    baseline_pdf = baseline_sdf.toPandas()
    rows = baseline_pdf.to_dict("records")
    enriched, report = run_recovery_benchmark_pandas(
        rows,
        catalog,
        char_malform_dct,
        recovery_config=rcfg,
        bench=cfg,
    )

    print("Overall vs T5 baseline:")
    for k, v in report["overall"].items():
        print(f"  {k}: {v}")

    spark = baseline_sdf.sparkSession
    enriched_sdf = spark.createDataFrame(enriched)
    enriched_sdf.write.mode(write_mode).saveAsTable(out_table)
    print(f"Wrote recovery eval → {out_table}")
    return enriched_sdf, report

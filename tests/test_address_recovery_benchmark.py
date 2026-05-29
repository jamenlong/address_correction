import unittest

from address_malform_recovery import MALFORM_STEP_REPLACE_CHAR, RecoveryConfig
from address_recovery_benchmark import (
    BenchmarkTableConfig,
    aggregate_benchmark_rows,
    enrich_row_with_recovery,
    jw_improvement_metrics,
)
from tests.test_address_malform_recovery import MINI_CHAR_MALFORM_DCT


class TestRecoveryBenchmark(unittest.TestCase):
    def test_jw_improvement_metrics(self):
        m = jw_improvement_metrics(0.8, 0.9)
        self.assertAlmostEqual(m["jw_delta"], 0.1)
        self.assertAlmostEqual(m["jw_relative_gain"], 0.5)
        self.assertTrue(m["jw_improved"])

    def test_enrich_row_vs_baseline(self):
        row = {
            "malformed_first_line": "4o85 MAIN ST APT 103",
            "first_line": "4085 MAIN ST APT 103",
            "predicted": "4585 YUKON CT",
            "malformed_first_line_malform_steps": ", replace_char",
            "jw_input": 0.85,
            "jw_output": 0.4,
        }
        catalog = ["4085 MAIN ST APT 103", "4585 YUKON CT"]
        rcfg = RecoveryConfig(
            enabled_steps=frozenset({MALFORM_STEP_REPLACE_CHAR}),
            min_jw_to_accept=0.85,
            min_jw_margin_over_t5=0.01,
        )
        out = enrich_row_with_recovery(
            row, catalog, MINI_CHAR_MALFORM_DCT, recovery_config=rcfg
        )
        self.assertFalse(out["t5_exact"])
        self.assertGreater(out["jw_delta_vs_t5"], 0.0)
        self.assertEqual(out["prediction_source"], "recovery")
        self.assertTrue(
            out["recovery_fixed_t5_miss"] or out["hybrid_exact"],
            "hybrid should beat wrong T5 prediction on this example",
        )

    def test_aggregate_by_step(self):
        rows = [
            {
                "malformed_first_line_malform_steps": ", replace_char",
                "benchmark_jw_input": 0.9,
                "benchmark_jw_output_t5": 0.5,
                "benchmark_jw_delta_t5": -0.4,
                "jw_output_hybrid": 1.0,
                "jw_delta_hybrid": 0.1,
                "jw_delta_vs_t5": 0.5,
                "jw_improved_vs_t5": True,
                "t5_exact": False,
                "hybrid_exact": True,
                "recovery_fixed_t5_miss": True,
                "recovery_broke_t5_hit": False,
                "recovery_changed_prediction": True,
            }
        ]
        report = aggregate_benchmark_rows(rows)
        self.assertEqual(report["overall"]["n"], 1)
        self.assertEqual(report["overall"]["recovery_fixed_t5_miss"], 1)
        self.assertEqual(report["by_malform_step"][0]["group"], ", replace_char")

    def test_default_baseline_table(self):
        cfg = BenchmarkTableConfig()
        self.assertIn("20may2026", cfg.baseline_table)
        self.assertTrue(cfg.output_table().endswith("_recovery_eval"))

    def test_benchmark_summary_sql_includes_unfixed_queries(self):
        from address_recovery_benchmark import benchmark_summary_sql

        q = benchmark_summary_sql("model_output.test_recovery_eval")
        for key in (
            "outcome_breakdown",
            "still_unfixed",
            "still_unfixed_breakdown",
        ):
            self.assertIn(key, q)
            self.assertIn("model_output.test_recovery_eval", q[key])


if __name__ == "__main__":
    unittest.main()

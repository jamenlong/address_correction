import unittest

from address_malform_recovery import (
    MALFORM_STEP_REPLACE_CHAR,
    RecoveryConfig,
    build_inverse_char_map_multi,
    is_allowed_address_text,
    normalize_address,
    parse_malform_steps,
    recover_address,
    undo_replace_char_beam,
    undo_replace_char_greedy,
)


# Minimal dict mirroring training malformations.
# Keep small: wide inverse maps false-positive on normal street-name letters.
MINI_CHAR_MALFORM_DCT = {
    "0": ["o", "O"],
    "8": ["B", "b"],
}

# Mirrors the real char_malform_dct structure: two separate canonical keys
# whose alt lists overlap.  '0' (digit) can be replaced by 'O'/'o', and
# 'O' (letter) can be replaced by '0'/'o'.  So a malformed 'o' or 'O' is
# genuinely ambiguous — it could have come from the digit 0 OR the letter O.
ZERO_O_DCT = {
    "0": ["O", "o"],   # digit zero replaced by letter O or lowercase o
    "O": ["0", "o"],   # letter O replaced by digit 0 or lowercase o
}


class TestAddressMalformRecovery(unittest.TestCase):
    def test_parse_steps(self):
        self.assertEqual(parse_malform_steps(", replace_char"), ["replace_char"])
        self.assertEqual(
            parse_malform_steps(", replace_char, add_special_characters"),
            ["replace_char", "add_special_characters"],
        )

    def test_undo_replace_char_greedy(self):
        inv = build_inverse_char_map_multi(MINI_CHAR_MALFORM_DCT)
        self.assertEqual(undo_replace_char_greedy("4o85", inv), "4085")
        self.assertEqual(undo_replace_char_greedy("B0", inv), "80")

    # ------------------------------------------------------------------
    # Beam search tests
    # ------------------------------------------------------------------

    def test_beam_no_ambiguity_returns_greedy(self):
        """When nothing is ambiguous the beam should match greedy."""
        inv = build_inverse_char_map_multi(MINI_CHAR_MALFORM_DCT)
        # "4o85" has only one canonical for 'o' -> '0'
        beam = undo_replace_char_beam("4o85", inv)
        self.assertIn("4085", beam)
        # Greedy result must be present
        greedy = undo_replace_char_greedy("4o85", inv)
        self.assertIn(greedy, beam)

    def test_beam_generates_both_0_and_O_candidates(self):
        """
        With 0/O symmetry, a malformed 'o' should produce candidates where
        that position is both the digit 0 and the letter O.
        For an address like 'FOO ST' where original could be 'F00 ST' or 'FOO ST',
        the beam should generate both variants so catalog scoring can pick.
        """
        inv = build_inverse_char_map_multi(ZERO_O_DCT)
        # Malformed: 'o' at position 1 could be digit 0 or letter O
        candidates = undo_replace_char_beam("Foo", inv, max_beam_positions=6, beam_width=32)
        normalized = [normalize_address(c) for c in candidates]
        # We expect at least one candidate with digit 0 and one with letter O
        has_zero = any("0" in c for c in normalized)
        has_letter_o = any(c.replace("0", "") != c.replace("O", "") or "O" in c for c in normalized)
        self.assertTrue(
            len(candidates) > 1,
            f"Expected multiple candidates for ambiguous 0/O, got: {candidates}",
        )
        self.assertTrue(
            has_zero or has_letter_o,
            f"Expected 0/O variants in candidates, got: {candidates}",
        )

    def test_beam_width_caps_candidates(self):
        """beam_width must hard-cap the number of returned strings."""
        inv = build_inverse_char_map_multi(ZERO_O_DCT)
        # 10 characters all ambiguous -> would be huge without cap
        text = "o" * 10
        candidates = undo_replace_char_beam(text, inv, max_beam_positions=10, beam_width=8)
        self.assertLessEqual(len(candidates), 8)

    def test_beam_max_positions_limits_branching(self):
        """max_beam_positions=1 should only branch on the first ambiguous spot."""
        inv = build_inverse_char_map_multi(ZERO_O_DCT)
        # Three ambiguous positions
        text = "ooo"
        # With max_beam_positions=1 only the first 'o' branches
        candidates_1 = undo_replace_char_beam(text, inv, max_beam_positions=1, beam_width=64)
        candidates_3 = undo_replace_char_beam(text, inv, max_beam_positions=3, beam_width=64)
        self.assertLessEqual(len(candidates_1), len(candidates_3))

    def test_beam_catalog_picks_correct_address(self):
        """
        Catalog has F00 ST and FOO ST.  Malformed input uses digit 0 and letter O
        interchangeably.  The beam should generate candidates that include perfect
        JW=1 matches for both, and recovery should accept one of the two correct
        catalog entries (not an unrelated one like FAR AVE).
        """
        catalog = ["F00 ST", "FOO ST", "FAR AVE"]
        cfg = RecoveryConfig(
            enabled_steps=frozenset({MALFORM_STEP_REPLACE_CHAR}),
            min_jw_to_accept=0.5,
            max_beam_positions=6,
            beam_width=32,
        )
        # f00 st: both F00 ST and FOO ST score JW=1 among candidates; either is correct
        res_00 = recover_address(
            "f00 st", catalog, ZERO_O_DCT, malform_steps=", replace_char", config=cfg,
        )
        self.assertIn(
            normalize_address(res_00.catalog_match),
            {normalize_address("F00 ST"), normalize_address("FOO ST")},
            f"f00 st: expected F00 ST or FOO ST, got {res_00.catalog_match}",
        )
        self.assertGreaterEqual(res_00.catalog_jw, 0.99)

        # foo st: same — beam covers both variants, either catalog entry is valid
        res_oo = recover_address(
            "foo st", catalog, ZERO_O_DCT, malform_steps=", replace_char", config=cfg,
        )
        self.assertIn(
            normalize_address(res_oo.catalog_match),
            {normalize_address("F00 ST"), normalize_address("FOO ST")},
            f"foo st: expected F00 ST or FOO ST, got {res_oo.catalog_match}",
        )
        self.assertGreaterEqual(res_oo.catalog_jw, 0.99)

        # far ave should never win for either input
        self.assertNotEqual(normalize_address(res_00.catalog_match), normalize_address("FAR AVE"))
        self.assertNotEqual(normalize_address(res_oo.catalog_match), normalize_address("FAR AVE"))

    # ------------------------------------------------------------------
    # Existing integration tests (unchanged behaviour)
    # ------------------------------------------------------------------

    def test_recovery_only_when_step_enabled(self):
        catalog = ["4085 MAIN ST APT 103", "4585 YUKON CT"]
        cfg = RecoveryConfig(
            enabled_steps=frozenset({MALFORM_STEP_REPLACE_CHAR}),
            min_jw_to_accept=0.85,
        )
        res = recover_address(
            "4o85 MAIN ST APT 103",
            catalog,
            MINI_CHAR_MALFORM_DCT,
            malform_steps=", replace_char",
            config=cfg,
        )
        self.assertTrue(res.used_recovery)
        self.assertIn("4085", normalize_address(res.recovered_text))

        res_off = recover_address(
            "4o85 MAIN ST APT 103",
            catalog,
            MINI_CHAR_MALFORM_DCT,
            malform_steps=", add_special_characters",
            config=cfg,
        )
        self.assertFalse(res_off.used_recovery)
        self.assertEqual(res_off.applied_steps, ())

    def test_allowed_address_charset(self):
        self.assertTrue(is_allowed_address_text("4085 MAIN ST APT 103"))
        self.assertTrue(is_allowed_address_text("123-45 N. AVE, STE 2"))
        self.assertFalse(is_allowed_address_text("4085\u0305 MAIN"))

    def test_strict_charset_blocks_recovery(self):
        catalog = ["4085 MAIN ST"]
        cfg = RecoveryConfig(
            enabled_steps=frozenset({MALFORM_STEP_REPLACE_CHAR}),
            min_jw_to_accept=0.5,
            strict_replace_char_undo=True,
        )
        res = recover_address(
            "4085\u0305 MAIN ST",
            catalog,
            MINI_CHAR_MALFORM_DCT,
            malform_steps=", replace_char",
            config=cfg,
        )
        self.assertFalse(res.used_recovery)
        self.assertFalse(res.charset_clean)
        self.assertEqual(res.debug, "charset_not_clean")

    def test_incremental_config_only_replace_char(self):
        catalog = ["4085 MAIN ST APT 103"]
        cfg = RecoveryConfig(
            enabled_steps=frozenset({MALFORM_STEP_REPLACE_CHAR}),
            min_jw_to_accept=0.85,
        )
        res = recover_address(
            "4o85 MAIN ST APT 103",
            catalog,
            MINI_CHAR_MALFORM_DCT,
            malform_steps=", replace_char",
            config=cfg,
        )
        self.assertEqual(
            normalize_address(res.catalog_match),
            normalize_address("4085 MAIN ST APT 103"),
        )


if __name__ == "__main__":
    unittest.main()

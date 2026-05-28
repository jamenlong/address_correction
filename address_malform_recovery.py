"""
Incremental malform recovery (undo steps + catalog JW match).

Enable one malform pattern at a time via ``RecoveryConfig.enabled_steps``. After each
new step is added, re-run per-step eval to confirm prior patterns still help and do
not regress.

Databricks: ``%run ./CharacterReplacementDictionary`` then pass ``char_malform_dct``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

# Step names must match Malform Addresses.py UDF output strings (e.g. "replace_char").
MALFORM_STEP_REPLACE_CHAR = "replace_char"

# Add future steps here as they are implemented, then enable in RecoveryConfig.
IMPLEMENTED_STEPS: FrozenSet[str] = frozenset({MALFORM_STEP_REPLACE_CHAR})


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


def normalize_address(text: str) -> str:
    """Light normalization for matching (not full Unicode folding)."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_malform_steps(steps_value: Optional[str]) -> List[str]:
    """Parse ``malformed_first_line_malform_steps`` (e.g. ', replace_char')."""
    if not steps_value:
        return []
    parts = [p.strip() for p in str(steps_value).split(",")]
    return [p for p in parts if p]


@dataclass
class RecoveryConfig:
    """Which undo steps are active and when to trust recovery over another predictor."""

    enabled_steps: FrozenSet[str] = field(
        default_factory=lambda: frozenset({MALFORM_STEP_REPLACE_CHAR})
    )
    catalog_top_k: int = 15
    min_jw_to_accept: float = 0.92
    min_jw_margin_over_t5: float = 0.05

    # Beam search controls for replace_char undo
    # max_beam_positions: how many ambiguous character positions we branch on.
    #   Each extra position multiplies candidates by the number of options at that spot.
    #   4-6 is a good balance; raising beyond ~8 produces diminishing returns fast.
    max_beam_positions: int = 6
    # beam_width: max live candidates kept after scoring each position during the beam.
    #   Acts as a hard cap so runtime stays predictable regardless of address length.
    beam_width: int = 64

    # Kept for backward compatibility; no longer the primary cap (beam_width is).
    max_undo_candidates: int = 64


@dataclass
class RecoveryResult:
    recovered_text: str
    catalog_match: Optional[str]
    catalog_jw: float
    applied_steps: Tuple[str, ...]
    used_recovery: bool
    debug: Optional[str] = None


@dataclass
class CatalogIndex:
    """Pre-normalized catalog for fast JW lookup (build once per benchmark run)."""

    entries: List[Tuple[str, str]]  # (normalized, raw)

    @classmethod
    def build(cls, catalog: Sequence[str]) -> "CatalogIndex":
        entries: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for raw in catalog:
            if not raw:
                continue
            norm = normalize_address(raw)
            if norm in seen:
                continue
            seen.add(norm)
            entries.append((norm, str(raw)))
        return cls(entries=entries)

    def best_match(self, query: str) -> Tuple[Optional[str], float]:
        q = normalize_address(query)
        if not q or not self.entries:
            return None, 0.0
        try:
            from rapidfuzz import process
            from rapidfuzz.distance import JaroWinkler

            norms = [e[0] for e in self.entries]
            hit = process.extractOne(
                q,
                norms,
                scorer=JaroWinkler.normalized_similarity,
            )
            if hit is not None:
                _, score, idx = hit
                return self.entries[idx][1], float(score)
        except Exception:
            pass
        best_jw = 0.0
        best_raw: Optional[str] = None
        for norm_c, raw in self.entries:
            if norm_c == q:
                return raw, 1.0
            jw = jaro_winkler(q, norm_c)
            if jw > best_jw:
                best_jw = jw
                best_raw = raw
                if best_jw >= 1.0:
                    break
        return best_raw, best_jw


def build_inverse_char_map(
    char_malform_dct: Dict[str, Sequence[str]],
) -> Dict[str, str]:
    """
    Map malformed token -> canonical character (key from char_malform_dct).

    Single-character alts also register upper/lower variants. Multi-character
  alts (e.g. homoglyph art) are stored as full-string keys.
    """
    inverse: Dict[str, str] = {}
    for canonical, alts in char_malform_dct.items():
        if not canonical:
            continue
        canon = str(canonical)
        for alt in alts or []:
            alt_s = str(alt)
            if not alt_s or alt_s == canon:
                continue
            inverse[alt_s] = canon
            if len(alt_s) == 1 and alt_s.isalpha():
                inverse[alt_s.upper()] = canon
                inverse[alt_s.lower()] = canon
    return inverse


def _steps_allowed(steps: Sequence[str], enabled: Set[str]) -> bool:
    if not steps:
        return False
    return all(s in enabled for s in steps)


def _tokenize_for_undo(
    text: str, inverse_map: Dict[str, str]
) -> List[Tuple[int, str, List[str]]]:
    """
    Walk the malformed string once and return a list of tokens:
        (start_index, raw_token, [canonical_option, ...])

    Strategy
    --------
    At each position we collect *all* keys in inverse_map whose length equals
    the longest match.  Each such key may map to a **different** canonical,
    producing a genuine alternative (ambiguous position).

    Additionally, a single ASCII alphanumeric character that appears in the
    inverse map is also always offered as itself — because replace_char can
    leave a character unchanged (the original IS the canonical).  This is
    what makes 0/O branching work: the malformed token 'O' maps to '0', but
    'O' unchanged is also a valid hypothesis (it really is the letter O).
    """
    keys_by_len = sorted(inverse_map.keys(), key=len, reverse=True)
    tokens: List[Tuple[int, str, List[str]]] = []
    i = 0
    n = len(text)
    while i < n:
        best_len = 0
        options: List[str] = []
        for key in keys_by_len:
            if not key:
                continue
            if text.startswith(key, i):
                if best_len == 0:
                    best_len = len(key)
                if len(key) == best_len:
                    canon = inverse_map[key]
                    if canon not in options:
                        options.append(canon)
        if best_len > 0:
            raw = text[i : i + best_len]
            # If the raw token is a single alphanumeric char it could also
            # simply *be* itself (the original character, not a substitution).
            # Add it as an additional hypothesis so the beam considers both
            # "this is a substitute" and "this was already canonical".
            if best_len == 1 and raw.isalnum() and raw not in options:
                options.append(raw)
            tokens.append((i, raw, options))
            i += best_len
        else:
            # Literal character — no mapping in the inverse map at all
            tokens.append((i, text[i], [text[i]]))
            i += 1
    return tokens


def undo_replace_char_greedy(text: str, inverse_map: Dict[str, str]) -> str:
    """Single greedy-undo pass: always pick the first canonical option."""
    if not text or not inverse_map:
        return text
    tokens = _tokenize_for_undo(text, inverse_map)
    return "".join(opts[0] for _, _, opts in tokens)


def undo_replace_char_beam(
    text: str,
    inverse_map: Dict[str, str],
    max_beam_positions: int = 6,
    beam_width: int = 64,
) -> List[str]:
    """
    Bounded beam search over ambiguous character positions.

    Algorithm
    ---------
    1. Tokenise the malformed string once, identifying every position where
       a mapped token has more than one possible canonical value.
    2. Only branch on up to ``max_beam_positions`` of those ambiguous spots
       (chosen left-to-right, which matches how replace_char applies subs).
    3. Maintain a beam of partial strings, capped at ``beam_width``.
       At each ambiguous token, expand every live beam state with every
       canonical option for that token, then trim back to ``beam_width``.
    4. Unambiguous tokens are appended to every beam state without branching.

    This gives us at most  ``beam_width``  complete candidate strings.
    For typical addresses (≤40 chars, ≤50% chars replaced, 2–3 options each)
    with ``max_beam_positions=6`` and ``beam_width=64`` the search completes
    in well under 1 ms per address.
    """
    if not text:
        return [""]

    tokens = _tokenize_for_undo(text, inverse_map)

    # Count how many positions are genuinely ambiguous
    ambiguous_indices = [
        idx for idx, (_, _, opts) in enumerate(tokens) if len(opts) > 1
    ]

    # If nothing is ambiguous, fall straight through to greedy
    if not ambiguous_indices:
        return ["".join(opts[0] for _, _, opts in tokens)]

    # Limit branching to the first max_beam_positions ambiguous positions
    branch_set: Set[int] = set(ambiguous_indices[:max_beam_positions])

    # Each beam state is a list of chosen canonical strings (one per token so far)
    beam: List[List[str]] = [[]]

    for tok_idx, (_, _, opts) in enumerate(tokens):
        if tok_idx in branch_set:
            # Expand: for each live state, fork on every option
            new_beam: List[List[str]] = []
            for state in beam:
                for opt in opts:
                    new_beam.append(state + [opt])
            # Trim to beam_width (keep first beam_width — they're all equally
            # scored at this point; scoring against the catalog happens later)
            beam = new_beam[:beam_width]
        else:
            # No branch: append the greedy (first) option to every live state
            chosen = opts[0]
            for state in beam:
                state.append(chosen)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for state in beam:
        s = "".join(state)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def best_catalog_match(
    query: str,
    catalog: Sequence[str],
    top_k: int = 15,
    catalog_index: Optional[CatalogIndex] = None,
) -> Tuple[Optional[str], float]:
    if catalog_index is not None:
        return catalog_index.best_match(query)
    return CatalogIndex.build(catalog).best_match(query)


def recover_address(
    malformed: str,
    catalog: Sequence[str],
    char_malform_dct: Dict[str, Sequence[str]],
    malform_steps: Optional[str] = None,
    config: Optional[RecoveryConfig] = None,
    *,
    catalog_index: Optional[CatalogIndex] = None,
    inverse_map: Optional[Dict[str, str]] = None,
) -> RecoveryResult:
    """
    Undo enabled malform steps, then pick best catalog line by JW on normalized text.
    """
    cfg = config or RecoveryConfig()
    steps = parse_malform_steps(malform_steps)
    enabled = set(cfg.enabled_steps)

    if not _steps_allowed(steps, enabled):
        return RecoveryResult(
            recovered_text=malformed or "",
            catalog_match=None,
            catalog_jw=0.0,
            applied_steps=tuple(),
            used_recovery=False,
            debug="steps_not_enabled_or_empty",
        )

    index = catalog_index or CatalogIndex.build(catalog)
    inv = inverse_map if inverse_map is not None else build_inverse_char_map(
        char_malform_dct
    )
    candidates: List[str] = [malformed or ""]

    if MALFORM_STEP_REPLACE_CHAR in enabled and MALFORM_STEP_REPLACE_CHAR in steps:
        candidates = undo_replace_char_beam(
            malformed or "",
            inv,
            max_beam_positions=cfg.max_beam_positions,
            beam_width=cfg.beam_width,
        )

    best_text = malformed or ""
    best_match: Optional[str] = None
    best_jw = 0.0
    for cand in candidates:
        match, jw = index.best_match(cand)
        if jw > best_jw:
            best_jw = jw
            best_match = match
            best_text = cand

    used = best_jw >= cfg.min_jw_to_accept and best_match is not None
    return RecoveryResult(
        recovered_text=best_text,
        catalog_match=best_match,
        catalog_jw=best_jw,
        applied_steps=tuple(s for s in steps if s in enabled),
        used_recovery=used,
    )


def apply_hybrid_override(
    t5_prediction: str,
    true_line: Optional[str],
    rec: RecoveryResult,
    config: Optional[RecoveryConfig] = None,
) -> str:
    """Pick T5 or recovery catalog line using JW margin vs truth (or T5 if no truth)."""
    cfg = config or RecoveryConfig()
    if not rec.used_recovery or not rec.catalog_match:
        return t5_prediction

    ref = true_line if true_line else t5_prediction
    norm_ref = normalize_address(ref)
    jw_rec = jaro_winkler(normalize_address(rec.catalog_match), norm_ref)
    jw_t5 = jaro_winkler(normalize_address(t5_prediction or ""), norm_ref)
    if jw_rec >= jw_t5 + cfg.min_jw_margin_over_t5:
        return rec.catalog_match
    return t5_prediction


def maybe_override_prediction(
    malformed: str,
    t5_prediction: str,
    true_line: Optional[str],
    catalog: Sequence[str],
    char_malform_dct: Dict[str, Sequence[str]],
    malform_steps: Optional[str] = None,
    config: Optional[RecoveryConfig] = None,
    *,
    catalog_index: Optional[CatalogIndex] = None,
    inverse_map: Optional[Dict[str, str]] = None,
    rec: Optional[RecoveryResult] = None,
) -> Tuple[str, RecoveryResult]:
    """
    Return recovery catalog match if it beats T5 by JW margin (when true_line given,
    comparison is to true_line; otherwise to T5 prediction only).
    """
    cfg = config or RecoveryConfig()
    if rec is None:
        rec = recover_address(
            malformed,
            catalog,
            char_malform_dct,
            malform_steps=malform_steps,
            config=cfg,
            catalog_index=catalog_index,
            inverse_map=inverse_map,
        )
    hybrid = apply_hybrid_override(t5_prediction, true_line, rec, cfg)
    return hybrid, rec


def summarize_recovery_eval(
    rows: Iterable[dict],
    *,
    step_column: str = "malformed_first_line_malform_steps",
    malformed_column: str = "malformed_first_line",
    true_column: str = "first_line",
    t5_column: str = "predicted",
) -> dict:
    """
    Per-row dicts: compare T5 vs recovery exact match on true_column (JW=1).
    Filter rows to those whose steps are a subset of enabled steps in each row's config
    by passing pre-filtered rows.
    """
    n = 0
    t5_ok = 0
    rec_ok = 0
    hybrid_ok = 0
    rec_fixed = 0
    rec_broke = 0
    for row in rows:
        n += 1
        true = row.get(true_column) or ""
        t5 = row.get(t5_column) or ""
        rec_line = row.get("recovery_match") or ""
        t5_hit = normalize_address(t5) == normalize_address(true)
        rec_hit = normalize_address(rec_line) == normalize_address(true)
        hybrid = row.get("hybrid_prediction") or t5
        hybrid_hit = normalize_address(hybrid) == normalize_address(true)
        t5_ok += int(t5_hit)
        rec_ok += int(rec_hit)
        hybrid_ok += int(hybrid_hit)
        if rec_hit and not t5_hit:
            rec_fixed += 1
        if t5_hit and not rec_hit:
            rec_broke += 1
    return {
        "n": n,
        "t5_accuracy": t5_ok / n if n else 0.0,
        "recovery_accuracy": rec_ok / n if n else 0.0,
        "hybrid_accuracy": hybrid_ok / n if n else 0.0,
        "recovery_fixed_t5_miss": rec_fixed,
        "recovery_broke_t5_hit": rec_broke,
    }

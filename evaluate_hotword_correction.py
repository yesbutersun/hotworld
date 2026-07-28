#!/usr/bin/env python3
"""Evaluate hotword recovery against a reference transcript."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_hotwords(path: Path) -> list[str]:
    seen = set()
    terms = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        term = raw.strip()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def count_occurrences(text: str, term: str) -> int:
    count = 0
    start = 0
    while True:
        pos = text.find(term, start)
        if pos < 0:
            return count
        count += 1
        start = pos + len(term)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hotword correction by exact hotword occurrence recovery.")
    parser.add_argument("--hotwords", default="hot-world.txt")
    parser.add_argument("--reference", default="standard.txt")
    parser.add_argument("--before", required=True, help="ASR text before correction.")
    parser.add_argument("--after", required=True, help="Text after correction.")
    args = parser.parse_args()

    hotwords = read_hotwords(Path(args.hotwords))
    reference = Path(args.reference).read_text(encoding="utf-8")
    before = Path(args.before).read_text(encoding="utf-8")
    after = Path(args.after).read_text(encoding="utf-8")

    total_expected = 0
    before_hits = 0
    after_hits = 0
    changed_expected = 0
    changed_fixed = 0
    changed_lost = 0
    rows = []

    for term in hotwords:
        expected = count_occurrences(reference, term)
        if expected == 0:
            continue
        before_count = count_occurrences(before, term)
        after_count = count_occurrences(after, term)
        before_hit = min(before_count, expected)
        after_hit = min(after_count, expected)

        total_expected += expected
        before_hits += before_hit
        after_hits += after_hit
        if before_hit < expected:
            changed_expected += expected - before_hit
            changed_fixed += max(0, after_hit - before_hit)
        if after_hit < before_hit:
            changed_lost += before_hit - after_hit

        if expected != before_count or expected != after_count:
            rows.append((term, expected, before_count, after_count, after_hit - before_hit))

    recall_before = before_hits / total_expected if total_expected else 0.0
    recall_after = after_hits / total_expected if total_expected else 0.0
    fix_rate = changed_fixed / changed_expected if changed_expected else 0.0

    print(f"expected_hotword_occurrences={total_expected}")
    print(f"before_hits={before_hits}")
    print(f"after_hits={after_hits}")
    print(f"before_recall={recall_before:.4f}")
    print(f"after_recall={recall_after:.4f}")
    print(f"absolute_recall_gain={recall_after - recall_before:.4f}")
    print(f"missed_occurrences_before={changed_expected}")
    print(f"fixed_missed_occurrences={changed_fixed}")
    print(f"fix_rate_among_before_misses={fix_rate:.4f}")
    print(f"lost_previous_hits={changed_lost}")
    print()
    print("top_changed_terms:")
    for term, expected, before_count, after_count, delta in sorted(rows, key=lambda x: abs(x[4]), reverse=True)[:40]:
        print(f"{term}\texpected={expected}\tbefore={before_count}\tafter={after_count}\tdelta={delta:+d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect hotwords still missing after correction."""

from __future__ import annotations

import argparse
import difflib
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


def snippets(text: str, term: str, radius: int = 28) -> list[str]:
    out = []
    start = 0
    while True:
        pos = text.find(term, start)
        if pos < 0:
            return out
        out.append(text[max(0, pos - radius) : pos + len(term) + radius].replace("\n", "\\n"))
        start = pos + len(term)


def best_near_snippet(reference_snippet: str, text: str, radius: int = 45) -> str:
    compact_text = text.replace("\n", "")
    ref = reference_snippet.replace("\\n", "")
    best_score = -1.0
    best = ""
    step = 12
    window = max(len(ref) + 18, radius * 2)
    for start in range(0, max(1, len(compact_text) - window + 1), step):
        candidate = compact_text[start : start + window]
        score = difflib.SequenceMatcher(None, ref, candidate).ratio()
        if score > best_score:
            best_score = score
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect hotwords missed after correction.")
    parser.add_argument("--hotwords", default="hot-world.txt")
    parser.add_argument("--reference", default="standard.txt")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    hotwords = read_hotwords(Path(args.hotwords))
    reference = Path(args.reference).read_text(encoding="utf-8")
    before = Path(args.before).read_text(encoding="utf-8")
    after = Path(args.after).read_text(encoding="utf-8")

    misses = []
    for term in hotwords:
        expected = count_occurrences(reference, term)
        after_count = count_occurrences(after, term)
        if expected > after_count:
            before_count = count_occurrences(before, term)
            misses.append((term, expected, before_count, after_count, expected - after_count))

    print(f"missed_terms={len(misses)}")
    print(f"missed_occurrences={sum(item[4] for item in misses)}")
    print()
    for term, expected, before_count, after_count, miss_count in misses[: args.limit]:
        print(f"TERM {term} expected={expected} before={before_count} after={after_count} missing={miss_count}")
        refs = snippets(reference, term)
        for ref in refs[:2]:
            print(f"  REF   {ref}")
            print(f"  ASR   {best_near_snippet(ref, before)}")
            print(f"  AFTER {best_near_snippet(ref, after)}")
        print()


if __name__ == "__main__":
    main()

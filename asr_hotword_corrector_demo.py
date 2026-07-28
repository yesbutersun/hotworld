#!/usr/bin/env python3
"""
ASR hotword correction demo.

Pipeline:
1. Read domain hotwords.
2. Convert hotwords to pinyin variants.
3. Build a pinyin 2-gram inverted index.
4. Slide windows over ASR text and recall candidates.
5. Rank candidates with syllable-level weighted edit distance.
6. Apply non-overlapping high-confidence replacements.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from pypinyin import Style, pinyin
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pypinyin. Install it with: python3 -m pip install -r requirements.txt"
    ) from exc


CONFUSION = {
    "n": ["l"],
    "l": ["n"],
    "zh": ["z"],
    "z": ["zh"],
    "ch": ["c"],
    "c": ["ch"],
    "sh": ["s"],
    "s": ["sh"],
    "in": ["ing"],
    "ing": ["in"],
    "en": ["eng"],
    "eng": ["en"],
    "an": ["ang"],
    "ang": ["an"],
    "f": ["h"],
    "h": ["f"],
    "r": ["l", "y"],
}

INITIALS = (
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
)

DROP_IN_WINDOW = set(" \t\r\n，。！？、；：,.!?;:\"'“”‘’（）()【】[]《》<>/\\|")
WHITESPACE_IN_WINDOW = set(" \t\r\n")
DEFAULT_CHUNK_SEPARATORS = "。！？!?；;\n"
E_EQUIV_SYLLABLES = {"e", "yi"}

@dataclass(frozen=True)
class Hotword:
    term: str
    category: str
    pinyin_variants: tuple[tuple[str, ...], ...]
    variant_grams: tuple[frozenset[str], ...]
    grams: frozenset[str]


@dataclass(frozen=True)
class Proposal:
    start: int
    end: int
    source: str
    target: str
    score: float
    phonetic_similarity: float
    char_similarity: float
    first_initial_similarity: float
    gram_coverage: float
    decision_source: str = "rule"
    llm_reason: str = ""
    hotword_category: str = "GENERAL"
    llm_confidence: float = 0.0


@dataclass(frozen=True)
class TextChunk:
    start: int
    end: int


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def is_candidate_char(ch: str) -> bool:
    return is_cjk(ch) or ch.isascii() and ch.isalnum()


def should_drop_in_window(ch: str) -> bool:
    return ch in DROP_IN_WINDOW


def normalize_window_text(text: str) -> str:
    return "".join(ch for ch in text if not should_drop_in_window(ch))


def normalize_window_whitespace(text: str) -> str:
    return "".join(ch for ch in text if ch not in WHITESPACE_IN_WINDOW)


def has_non_whitespace_separator(text: str) -> bool:
    return any(should_drop_in_window(ch) and ch not in WHITESPACE_IN_WINDOW for ch in text)


def split_text_chunks(
    text: str,
    max_chunk_len: int = 0,
    separators: str = DEFAULT_CHUNK_SEPARATORS,
    overlap: int = 0,
) -> list[TextChunk]:
    """Split text into scan ranges by punctuation, then by max length."""
    chunks: list[TextChunk] = []
    separator_chars = set(separators)
    overlap = max(0, overlap)

    def add_span(start: int, end: int) -> None:
        while start < end and text[start] in separator_chars:
            start += 1
        while end > start and text[end - 1] in separator_chars:
            end -= 1
        if start >= end:
            return
        if max_chunk_len <= 0:
            chunks.append(TextChunk(start, end))
            return
        cursor = start
        step = max(1, max_chunk_len - min(overlap, max_chunk_len - 1))
        while cursor < end:
            chunk_end = min(end, cursor + max_chunk_len)
            chunks.append(TextChunk(cursor, chunk_end))
            if chunk_end == end:
                break
            cursor += step

    start = 0
    if separator_chars:
        for index, ch in enumerate(text):
            if ch not in separator_chars:
                continue
            add_span(start, index)
            start = index + 1
    add_span(start, len(text))
    return chunks


def count_occurrences(text: str, term: str) -> int:
    count = 0
    start = 0
    while True:
        pos = text.find(term, start)
        if pos < 0:
            return count
        count += 1
        start = pos + len(term)


def read_hotwords(path: Path) -> list[tuple[str, str]]:
    seen = set()
    terms = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "\t" in line:
            term, category = line.split("\t", 1)
            term = term.strip()
            category = category.strip() or "GENERAL"
        else:
            term = line
            category = "GENERAL"
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append((term, category))
    return terms


@lru_cache(maxsize=200_000)
def pinyin_variants(text: str, limit: int = 24) -> tuple[tuple[str, ...], ...]:
    """Return limited pinyin variants for polyphonic characters."""
    raw = pinyin(text, style=Style.NORMAL, heteronym=True, errors=lambda x: list(x))
    variants: list[tuple[str, ...]] = [()]
    for choices in raw:
        normalized = []
        for choice in choices:
            item = choice.lower().strip()
            if item and item not in normalized:
                normalized.append(item)
                if item in E_EQUIV_SYLLABLES:
                    for equivalent in sorted(E_EQUIV_SYLLABLES):
                        if equivalent not in normalized:
                            normalized.append(equivalent)
        if not normalized:
            continue
        next_variants = []
        for prefix in variants:
            for item in normalized:
                next_variants.append(prefix + (item,))
                if len(next_variants) >= limit:
                    break
            if len(next_variants) >= limit:
                break
        variants = next_variants
    return tuple(variants[:limit])


def ngrams(items: tuple[str, ...], n: int = 2) -> set[str]:
    if not items:
        return set()
    if len(items) < n:
        return {"_".join(items)}
    return {"_".join(items[i : i + n]) for i in range(len(items) - n + 1)}


def split_initial_final(syllable: str) -> tuple[str, str]:
    for initial in INITIALS:
        if syllable.startswith(initial):
            return initial, syllable[len(initial) :]
    return "", syllable


def confused(a: str, b: str) -> bool:
    return b in CONFUSION.get(a, [])


def part_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if confused(a, b):
        return 0.25
    return 1.0


def syllable_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a in E_EQUIV_SYLLABLES and b in E_EQUIV_SYLLABLES:
        return 0.0
    ia, fa = split_initial_final(a)
    ib, fb = split_initial_final(b)
    return part_cost(ia, ib) + part_cost(fa, fb)


def weighted_edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + 1.0
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + 1.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + syllable_cost(a[i - 1], b[j - 1]),
            )
    return dp[m][n]


def phonetic_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a or not b:
        return 0.0
    distance = weighted_edit_distance(a, b)
    return max(0.0, 1.0 - distance / max(len(a), len(b)))


def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def char_similarity(a: str, b: str) -> float:
    return lcs_len(a, b) / max(len(a), len(b), 1)


def first_initial_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a or not b:
        return 0.0
    ia, _ = split_initial_final(a[0])
    ib, _ = split_initial_final(b[0])
    if ia == ib:
        return 1.0
    if confused(ia, ib):
        return 0.75
    return 0.0


def best_variant_score(source_variants: Iterable[tuple[str, ...]], hotword: Hotword) -> tuple[float, float]:
    best_phonetic = 0.0
    best_first = 0.0
    for source_py in source_variants:
        for target_py in hotword.pinyin_variants:
            phonetic = phonetic_similarity(source_py, target_py)
            if phonetic > best_phonetic:
                best_phonetic = phonetic
                best_first = first_initial_similarity(source_py, target_py)
    return best_phonetic, best_first


def build_hotword_index(terms: list[tuple[str, str]]) -> tuple[list[Hotword], dict[str, set[int]], int, int]:
    hotwords = []
    inverted: dict[str, set[int]] = defaultdict(set)
    min_len = 10**9
    max_len = 0
    for term, category in terms:
        variants = pinyin_variants(term)
        variant_grams = tuple(frozenset(ngrams(variant)) for variant in variants)
        grams = set()
        for grams_for_variant in variant_grams:
            grams.update(grams_for_variant)
        hotword = Hotword(
            term=term,
            category=category,
            pinyin_variants=variants,
            variant_grams=variant_grams,
            grams=frozenset(grams),
        )
        index = len(hotwords)
        hotwords.append(hotword)
        for gram in grams:
            inverted[gram].add(index)
        min_len = min(min_len, len(term))
        max_len = max(max_len, len(term))
    return hotwords, inverted, min_len, max_len


def query_candidates(
    text: str,
    hotwords: list[Hotword],
    inverted: dict[str, set[int]],
    top_k: int = 30,
) -> list[tuple[Hotword, float]]:
    variants = pinyin_variants(text)
    source_grams = set()
    for variant in variants:
        source_grams.update(ngrams(variant))
    if not source_grams:
        return []

    counts: Counter[int] = Counter()
    for gram in source_grams:
        counts.update(inverted.get(gram, ()))

    candidates = []
    for idx, overlap_count in counts.most_common(top_k):
        hotword = hotwords[idx]
        gram_coverage = max(
            (len(source_grams.intersection(grams_for_variant)) / max(len(grams_for_variant), 1))
            for grams_for_variant in hotword.variant_grams
        )
        candidates.append((hotword, gram_coverage))
    return candidates

# 做已正确热词保护，找到热词位置
def exact_hotword_spans(text: str, hotwords: list[Hotword]) -> list[tuple[int, int]]:
    spans = []
    for hotword in hotwords:
        if len(hotword.term) <= 1:
            continue
        start = 0
        while True:
            pos = text.find(hotword.term, start)
            if pos < 0:
                break
            spans.append((pos, pos + len(hotword.term)))
            start = pos + len(hotword.term)
    return spans


def inside_longer_exact_hotword(start: int, end: int, protected_spans: list[tuple[int, int]]) -> bool:
    for span_start, span_end in protected_spans:
        if span_start <= start and end <= span_end and (span_start, span_end) != (start, end):
            return True
    return False


def collect_proposals(
    text: str,
    hotwords: list[Hotword],
    inverted: dict[str, set[int]],
    min_word_len: int,
    max_word_len: int,
    threshold: float,
    collect_llm_candidates: bool,
    chunk_max_len: int = 0,
    chunk_separators: str = DEFAULT_CHUNK_SEPARATORS,
) -> list[Proposal]:
    proposals = []
    protected_spans = exact_hotword_spans(text, hotwords)
    hotword_terms = {hotword.term for hotword in hotwords}
    hotword_categories = {hotword.term: hotword.category for hotword in hotwords}
    scan_chunks = split_text_chunks(
        text,
        chunk_max_len,
        chunk_separators,
        overlap=max_word_len + 6,
    )
    for chunk in scan_chunks:
        for start in range(chunk.start, chunk.end):
            if not is_candidate_char(text[start]):
                continue
            normalized_chars = []
            dropped_count = 0
            max_raw_end = min(chunk.end, start + max_word_len + 6)
            for end in range(start + 1, max_raw_end + 1):
                ch = text[end - 1]
                if is_candidate_char(ch):
                    normalized_chars.append(ch)
                    dropped_count = 0
                elif should_drop_in_window(ch):
                    if not normalized_chars:
                        break
                    dropped_count += 1
                    if dropped_count > 2:
                        break
                    continue
                else:
                    break
                source = "".join(normalized_chars)
                if len(source) < min_word_len:
                    continue
                if len(source) > max_word_len:
                    break
                raw_source = text[start:end]
                if source == raw_source and should_drop_in_window(raw_source[-1]):
                    continue
                if has_non_whitespace_separator(raw_source):
                    continue
                if inside_longer_exact_hotword(start, end, protected_spans):
                    continue
                if raw_source in hotword_terms:
                    continue
                whitespace_normalized_source = normalize_window_whitespace(raw_source)
                if source in hotword_terms:
                    if whitespace_normalized_source == source and whitespace_normalized_source != raw_source:
                        proposals.append(
                            Proposal(
                                start=start,
                                end=end,
                                source=raw_source,
                                target=source,
                                score=1.0,
                                phonetic_similarity=1.0,
                                char_similarity=1.0,
                                first_initial_similarity=1.0,
                                gram_coverage=1.0,
                                hotword_category=hotword_categories.get(source, "GENERAL"),
                            )
                        )
                    continue
                variants = pinyin_variants(source)
                for hotword, gram_coverage in query_candidates(source, hotwords, inverted):
                    if raw_source == hotword.term:
                        continue
                    if len(source) != len(hotword.term):
                        continue
                    length_score = 1.0 - min(
                        abs(len(source) - len(hotword.term)) / max(len(source), len(hotword.term)),
                        1.0,
                    )
                    if length_score < 0.55:
                        continue
                    phonetic, first_initial = best_variant_score(variants, hotword)
                    chars = char_similarity(source, hotword.term)
                    score = (
                        0.50 * phonetic
                        + 0.20 * chars
                        + 0.10 * first_initial
                        + 0.10 * gram_coverage
                        + 0.10 * length_score
                    )
                    if len(source) != len(hotword.term):
                        continue
                    if chars == 0.0:
                        continue
                    if len(source) <= 3:
                        if (
                            collect_llm_candidates
                            and score >= 0.80
                            and phonetic >= 0.72
                            and gram_coverage >= 0.34
                        ):
                            proposals.append(
                                Proposal(
                                    start=start,
                                    end=end,
                                    source=raw_source,
                                    target=hotword.term,
                                    score=score,
                                    phonetic_similarity=phonetic,
                                    char_similarity=chars,
                                    first_initial_similarity=first_initial,
                                    gram_coverage=gram_coverage,
                                    decision_source="llm_candidate",
                                    hotword_category=hotword.category,
                                )
                            )
                        continue
                    if len(source) <= 2 and chars < 0.5:
                        continue
                    if score >= threshold and phonetic >= 0.72 and gram_coverage >= 0.34:
                        proposals.append(
                            Proposal(
                                start=start,
                                end=end,
                                source=raw_source,
                                target=hotword.term,
                                score=score,
                                phonetic_similarity=phonetic,
                                char_similarity=chars,
                                first_initial_similarity=first_initial,
                                gram_coverage=gram_coverage,
                                decision_source="rule",
                                hotword_category=hotword.category,
                            )
                        )
    deduped: dict[tuple[int, int, str, str, str], Proposal] = {}
    for proposal in proposals:
        key = (
            proposal.start,
            proposal.end,
            proposal.source,
            proposal.target,
            proposal.decision_source,
        )
        current = deduped.get(key)
        if current is None or proposal.score > current.score:
            deduped[key] = proposal
    return list(deduped.values())


def reference_snippets(reference: str, term: str, radius: int = 36) -> list[str]:
    snippets = []
    start = 0
    compact_reference = reference.replace("\n", "")
    while True:
        pos = compact_reference.find(term, start)
        if pos < 0:
            return snippets
        snippets.append(compact_reference[max(0, pos - radius) : pos + len(term) + radius])
        start = pos + len(term)


def proposal_context(text: str, proposal: Proposal, radius: int = 36) -> str:
    replaced = text[max(0, proposal.start - radius) : proposal.start]
    replaced += proposal.target
    replaced += text[proposal.end : min(len(text), proposal.end + radius)]
    return replaced.replace("\n", "")


def best_reference_similarity(context: str, snippets: list[str]) -> float:
    if not snippets:
        return 0.0
    return max(difflib.SequenceMatcher(None, context, snippet).ratio() for snippet in snippets)


def mock_llm_rerank(
    text: str,
    proposals: list[Proposal],
    reference: str,
    similarity_threshold: float = 0.38,
) -> list[Proposal]:
    """Reference-backed mock LLM for demo evaluation only.

    It simulates a conservative LLM by accepting a high-risk candidate only when
    the candidate's replaced local context is similar to a reference occurrence
    of the target hotword.
    """
    accepted = [proposal for proposal in proposals if proposal.decision_source == "rule"]
    llm_candidates = [proposal for proposal in proposals if proposal.decision_source == "llm_candidate"]
    expected_counts = {proposal.target: count_occurrences(reference, proposal.target) for proposal in llm_candidates}
    current_counts = {target: count_occurrences(text, target) for target in expected_counts}
    snippet_cache: dict[str, list[str]] = {}

    ordered = sorted(
        llm_candidates,
        key=lambda p: (p.score, len(p.target), p.end - p.start, p.char_similarity),
        reverse=True,
    )
    for proposal in ordered:
        if current_counts.get(proposal.target, 0) >= expected_counts.get(proposal.target, 0):
            continue
        if proposal.target not in snippet_cache:
            snippet_cache[proposal.target] = reference_snippets(reference, proposal.target)
        similarity = best_reference_similarity(
            proposal_context(text, proposal),
            snippet_cache[proposal.target],
        )
        if similarity < similarity_threshold:
            continue
        accepted.append(
            Proposal(
                start=proposal.start,
                end=proposal.end,
                source=proposal.source,
                target=proposal.target,
                score=proposal.score,
                phonetic_similarity=proposal.phonetic_similarity,
                char_similarity=proposal.char_similarity,
                first_initial_similarity=proposal.first_initial_similarity,
                gram_coverage=proposal.gram_coverage,
                decision_source="mock_llm",
                llm_reason=f"reference_context_similarity={similarity:.4f}",
                hotword_category=proposal.hotword_category,
                llm_confidence=similarity,
            )
        )
        current_counts[proposal.target] = current_counts.get(proposal.target, 0) + 1
    return accepted


def sentence_context(text: str, start: int, end: int, radius: int = 100) -> str:
    sentence_breaks = "。！？!?；;\n"
    left = max(text.rfind(mark, 0, start) for mark in sentence_breaks) + 1
    right_candidates = [text.find(mark, end) for mark in sentence_breaks]
    right_candidates = [pos for pos in right_candidates if pos >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    left = max(left, start - radius)
    right = min(right, end + radius)
    return text[left:right].replace("\n", "")


def chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return json.loads(content)


def call_llm_decision(
    *,
    api_url: str,
    api_key: str,
    model: str,
    context: str,
    proposal: Proposal,
    timeout: float,
) -> dict:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 ASR 热词纠错二阶段裁决器。"
                    "只能判断给定候选是否应该替换，不能创造新热词。"
                    "只输出 JSON，不输出解释性正文。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "判断 ASR 片段在当前上下文中是否应替换为候选热词。",
                        "output_schema": {
                            "decision": "replace 或 keep",
                            "target": "replace 时必须等于 candidate；keep 时为空字符串",
                            "confidence": "0 到 1 的数字",
                            "reason": "简短原因",
                        },
                        "rules": [
                            "如果上下文无法支持候选热词，选择 keep。",
                            "如果 source 本身也可能是合理真实表达，选择 keep。",
                            "target 不能是候选热词之外的内容。",
                        ],
                        "context": context,
                        "source": proposal.source,
                        "candidate": proposal.target,
                        "hotword_category": proposal.hotword_category,
                        "rule_score": round(proposal.score, 4),
                        "phonetic_similarity": round(proposal.phonetic_similarity, 4),
                        "char_similarity": round(proposal.char_similarity, 4),
                        "gram_coverage": round(proposal.gram_coverage, 4),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    content = body["choices"][0]["message"]["content"]
    return parse_llm_json(content)


def real_llm_rerank(
    text: str,
    proposals: list[Proposal],
    *,
    api_url: str,
    api_key: str,
    model: str,
    confidence_threshold: float = 0.80,
    timeout: float = 30.0,
    max_candidates: int = 80,
) -> list[Proposal]:
    accepted: list[Proposal] = []
    ordered = sorted(
        proposals,
        key=lambda p: (p.decision_source == "llm_candidate", p.score, len(p.target), p.end - p.start),
        reverse=True,
    )
    if max_candidates > 0:
        ordered = ordered[:max_candidates]

    for proposal in ordered:
        decision = call_llm_decision(
            api_url=api_url,
            api_key=api_key,
            model=model,
            context=sentence_context(text, proposal.start, proposal.end),
            proposal=proposal,
            timeout=timeout,
        )
        should_replace = decision.get("decision") == "replace"
        target_matches = decision.get("target") == proposal.target
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        reason = str(decision.get("reason", "") or "")
        if not should_replace or not target_matches or confidence < confidence_threshold:
            continue
        accepted.append(
            Proposal(
                start=proposal.start,
                end=proposal.end,
                source=proposal.source,
                target=proposal.target,
                score=proposal.score,
                phonetic_similarity=proposal.phonetic_similarity,
                char_similarity=proposal.char_similarity,
                first_initial_similarity=proposal.first_initial_similarity,
                gram_coverage=proposal.gram_coverage,
                decision_source="llm",
                llm_reason=reason,
                hotword_category=proposal.hotword_category,
                llm_confidence=confidence,
            )
        )
    return accepted


def select_non_overlapping(proposals: list[Proposal]) -> list[Proposal]:
    ordered = sorted(
        proposals,
        key=lambda p: (p.score, len(p.target), p.end - p.start, p.char_similarity),
        reverse=True,
    )
    selected = []
    occupied = set()
    for proposal in ordered:
        span = set(range(proposal.start, proposal.end))
        if occupied.intersection(span):
            continue
        selected.append(proposal)
        occupied.update(span)
    return sorted(selected, key=lambda p: p.start)


def apply_replacements(text: str, replacements: list[Proposal]) -> str:
    pieces = []
    cursor = 0
    for replacement in replacements:
        pieces.append(text[cursor : replacement.start])
        pieces.append(replacement.target)
        cursor = replacement.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def correct_text(
    text: str,
    hotwords: list[Hotword],
    inverted: dict[str, set[int]],
    min_word_len: int,
    max_word_len: int,
    threshold: float,
    mock_llm_reference: str | None,
    llm_api_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_confidence_threshold: float = 0.80,
    llm_timeout: float = 30.0,
    llm_max_candidates: int = 80,
    chunk_max_len: int = 0,
    chunk_separators: str = DEFAULT_CHUNK_SEPARATORS,
) -> tuple[str, list[Proposal]]:
    use_real_llm = llm_api_url is not None and llm_api_key is not None and llm_model is not None
    proposals = collect_proposals(
        text,
        hotwords,
        inverted,
        min_word_len,
        max_word_len,
        threshold,
        collect_llm_candidates=mock_llm_reference is not None or use_real_llm,
        chunk_max_len=chunk_max_len,
        chunk_separators=chunk_separators,
    )
    if use_real_llm:
        proposals = real_llm_rerank(
            text,
            proposals,
            api_url=llm_api_url,
            api_key=llm_api_key,
            model=llm_model,
            confidence_threshold=llm_confidence_threshold,
            timeout=llm_timeout,
            max_candidates=llm_max_candidates,
        )
    elif mock_llm_reference is not None:
        proposals = mock_llm_rerank(text, proposals, mock_llm_reference)
    replacements = select_non_overlapping(proposals)
    return apply_replacements(text, replacements), replacements


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo ASR hotword correction with pinyin inverted index.")
    parser.add_argument("--hotwords", default="hot-world.txt", help="Hotword file, one term per line.")
    parser.add_argument("--input", default="asr-result-online-tts.txt", help="ASR text file.")
    parser.add_argument("--text", help="ASR text content. When set, this takes precedence over --input.")
    parser.add_argument(
        "--output",
        help=(
            "Corrected text output file. Defaults to stdout for --text and corrected.txt for --input."
        ),
    )
    parser.add_argument(
        "--report",
        help=(
            "Replacement report output file. Defaults to no report for --text and corrections.jsonl for --input."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.82, help="Replacement confidence threshold.")
    parser.add_argument("--max-extra-len", type=int, default=1, help="Allow windows longer than max hotword length.")
    parser.add_argument(
        "--chunk-max-len",
        type=int,
        default=200,
        help="Maximum scan chunk length after punctuation splitting; <=0 disables length chunking.",
    )
    parser.add_argument(
        "--chunk-separators",
        default=DEFAULT_CHUNK_SEPARATORS,
        help="Characters used to split scan chunks before candidate recall; empty string disables punctuation chunking.",
    )
    parser.add_argument("--mock-llm", action="store_true", help="Use a reference-backed mock LLM reranker.")
    parser.add_argument("--mock-llm-reference", default="standard.txt", help="Reference text used only by --mock-llm.")
    parser.add_argument("--llm", action="store_true", help="Use a real OpenAI-compatible LLM reranker.")
    parser.add_argument(
        "--llm-api-url",
        default=os.getenv("ASR_LLM_API_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/chat/completions",
        help="OpenAI-compatible chat completions URL or base URL.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("ASR_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="LLM API key. Defaults to ASR_LLM_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("ASR_LLM_MODEL") or os.getenv("OPENAI_MODEL"),
        help="LLM model name. Defaults to ASR_LLM_MODEL or OPENAI_MODEL.",
    )
    parser.add_argument("--llm-confidence", type=float, default=0.80, help="Minimum LLM confidence to accept replacement.")
    parser.add_argument("--llm-timeout", type=float, default=30.0, help="LLM request timeout in seconds.")
    parser.add_argument("--llm-max-candidates", type=int, default=80, help="Maximum candidate proposals sent to LLM; <=0 means unlimited.")
    args = parser.parse_args()

    if args.llm and args.mock_llm:
        raise SystemExit("--llm and --mock-llm cannot be used together.")
    if args.llm and not args.llm_api_key:
        raise SystemExit("Missing LLM API key. Set ASR_LLM_API_KEY/OPENAI_API_KEY or pass --llm-api-key.")
    if args.llm and not args.llm_model:
        raise SystemExit("Missing LLM model. Set ASR_LLM_MODEL/OPENAI_MODEL or pass --llm-model.")

    input_is_text = args.text is not None
    output_path = Path(args.output) if args.output else (None if input_is_text else Path("corrected.txt"))
    report_path = Path(args.report) if args.report else (None if input_is_text else Path("corrections.jsonl"))

    terms = read_hotwords(Path(args.hotwords))
    hotwords, inverted, min_word_len, max_word_len = build_hotword_index(terms)
    min_word_len = max(1, min_word_len)
    max_word_len = max_word_len + args.max_extra_len
    chunk_max_len = args.chunk_max_len
    min_chunk_len = max_word_len + 6
    if 0 < chunk_max_len < min_chunk_len:
        chunk_max_len = min_chunk_len

    source_text = args.text if input_is_text else Path(args.input).read_text(encoding="utf-8")
    mock_llm_reference = Path(args.mock_llm_reference).read_text(encoding="utf-8") if args.mock_llm else None
    corrected_text, replacements = correct_text(
        source_text,
        hotwords,
        inverted,
        min_word_len,
        max_word_len,
        args.threshold,
        mock_llm_reference,
        llm_api_url=chat_completions_url(args.llm_api_url) if args.llm else None,
        llm_api_key=args.llm_api_key if args.llm else None,
        llm_model=args.llm_model if args.llm else None,
        llm_confidence_threshold=args.llm_confidence,
        llm_timeout=args.llm_timeout,
        llm_max_candidates=args.llm_max_candidates,
        chunk_max_len=chunk_max_len,
        chunk_separators=args.chunk_separators,
    )

    if output_path is not None:
        output_path.write_text(corrected_text, encoding="utf-8")
    if report_path is not None:
        with report_path.open("w", encoding="utf-8") as fh:
            for item in replacements:
                fh.write(
                    json.dumps(
                        {
                            "start": item.start,
                            "end": item.end,
                            "source": item.source,
                            "target": item.target,
                            "score": round(item.score, 4),
                            "phonetic_similarity": round(item.phonetic_similarity, 4),
                            "char_similarity": round(item.char_similarity, 4),
                            "first_initial_similarity": round(item.first_initial_similarity, 4),
                            "gram_coverage": round(item.gram_coverage, 4),
                            "decision_source": item.decision_source,
                            "llm_reason": item.llm_reason,
                            "hotword_category": item.hotword_category,
                            "llm_confidence": round(item.llm_confidence, 4),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if input_is_text and output_path is None:
        print(corrected_text)
        return

    print(f"hotwords: {len(hotwords)}")
    print(f"pinyin grams: {len(inverted)}")
    print(f"replacements: {len(replacements)}")
    if output_path is not None:
        print(f"corrected text: {output_path}")
    if report_path is not None:
        print(f"report: {report_path}")
    if input_is_text:
        print("corrected content:")
        print(corrected_text)
    for item in replacements[:500]:
        print(f"{item.source} -> {item.target}  score={item.score:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Lightweight explorer for the NekoQA-30K dataset.

The dataset is a large JSON array. This script streams records from disk, so it
does not call json.load() on the whole file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


DEFAULT_PATH = "NekoQA-30K.json"
TEXT_FIELDS = ("instruction", "output")
EXPECTED_FIELDS = ("instruction", "output", "type")


def stream_json_records(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Yield dict records from a top-level JSON array or adjacent JSON objects."""
    decoder = json.JSONDecoder()

    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        pos = 0
        eof = False
        mode: Optional[str] = None

        def fill() -> bool:
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
                return True
            eof = True
            return False

        def compact() -> None:
            nonlocal buffer, pos
            if pos > chunk_size:
                buffer = buffer[pos:]
                pos = 0

        def skip_ws() -> None:
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof or not fill():
                    return

        fill()
        skip_ws()
        if pos >= len(buffer):
            return

        if buffer[pos] == "[":
            mode = "array"
            pos += 1
        elif buffer[pos] == "{":
            mode = "objects"
        else:
            raise ValueError(f"Unsupported JSON start byte: {buffer[pos]!r}")

        while True:
            skip_ws()
            if pos >= len(buffer):
                if eof:
                    return
                fill()
                continue

            if mode == "array":
                if buffer[pos] == "]":
                    return
                if buffer[pos] == ",":
                    pos += 1
                    continue

            try:
                item, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(f"Could not decode JSON near byte/char {pos}: {exc}") from exc
                fill()
                continue

            pos = end
            compact()

            if isinstance(item, dict):
                yield item
            else:
                yield {"_value": item}


def digest_text(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()


def percentile(sorted_values: List[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def shorten(value: Any, width: int) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


@dataclass
class LengthStats:
    values: List[int] = field(default_factory=list)

    def add(self, value: str) -> None:
        self.values.append(len(value))

    def summary(self) -> Dict[str, float]:
        if not self.values:
            return {"count": 0, "min": 0, "avg": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
        values = sorted(self.values)
        return {
            "count": len(values),
            "min": values[0],
            "avg": round(sum(values) / len(values), 2),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": values[-1],
        }


@dataclass
class ExplorerStats:
    sample_size: int
    seed: int
    total: int = 0
    key_counts: Counter = field(default_factory=Counter)
    type_counts: Counter = field(default_factory=Counter)
    missing_counts: Counter = field(default_factory=Counter)
    empty_counts: Counter = field(default_factory=Counter)
    length_stats: Dict[str, LengthStats] = field(default_factory=lambda: {name: LengthStats() for name in TEXT_FIELDS})
    duplicate_hashes: Dict[str, Counter] = field(default_factory=lambda: {name: Counter() for name in TEXT_FIELDS})
    newline_counts: Counter = field(default_factory=Counter)
    sample: List[Dict[str, Any]] = field(default_factory=list)
    sample_by_type: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def add_sample(self, record: Dict[str, Any]) -> None:
        trimmed = {
            "type": record.get("type", ""),
            "instruction": shorten(record.get("instruction", ""), 180),
            "output": shorten(record.get("output", ""), 220),
        }

        if len(self.sample) < self.sample_size:
            self.sample.append(trimmed)
        elif self.sample_size > 0:
            idx = self.rng.randrange(self.total)
            if idx < self.sample_size:
                self.sample[idx] = trimmed

        type_name = str(record.get("type", ""))
        bucket = self.sample_by_type[type_name]
        if len(bucket) < 2:
            bucket.append(trimmed)

    def add(self, record: Dict[str, Any]) -> None:
        self.total += 1
        self.key_counts.update(record.keys())
        self.add_sample(record)

        for field_name in EXPECTED_FIELDS:
            if field_name not in record:
                self.missing_counts[field_name] += 1
            elif record.get(field_name) in ("", None, [], {}):
                self.empty_counts[field_name] += 1

        type_name = str(record.get("type", "<missing>"))
        self.type_counts[type_name] += 1

        for field_name in TEXT_FIELDS:
            value = record.get(field_name, "")
            if not isinstance(value, str):
                value = str(value)
            self.length_stats[field_name].add(value)
            self.duplicate_hashes[field_name][digest_text(value)] += 1
            if "\n" in value:
                self.newline_counts[field_name] += 1

    def duplicate_summary(self) -> Dict[str, Dict[str, int]]:
        output: Dict[str, Dict[str, int]] = {}
        for field_name, counts in self.duplicate_hashes.items():
            duplicate_groups = sum(1 for count in counts.values() if count > 1)
            duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
            output[field_name] = {
                "unique": len(counts),
                "duplicate_groups": duplicate_groups,
                "duplicate_extra_rows": duplicate_rows,
            }
        return output


def print_table(rows: Iterable[Iterable[Any]]) -> None:
    rows = [[str(cell) for cell in row] for row in rows]
    if not rows:
        return
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream and summarize NekoQA-style JSON datasets.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help=f"dataset path, default: {DEFAULT_PATH}")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N records")
    parser.add_argument("--sample", type=int, default=5, help="reservoir sample size")
    parser.add_argument("--seed", type=int, default=7, help="random seed for sampling")
    parser.add_argument("--top-types", type=int, default=30, help="number of type rows to show")
    parser.add_argument("--show-samples", type=int, default=5, help="number of sampled rows to print")
    parser.add_argument("--search", default="", help="print records whose instruction/output/type contains this text")
    parser.add_argument("--max-matches", type=int, default=10, help="max search matches to print")
    parser.add_argument("--export-sample", default="", help="write the sampled rows to a JSONL file")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024, help="streaming read chunk size")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    stats = ExplorerStats(sample_size=max(0, args.sample), seed=args.seed)
    matches: List[Dict[str, Any]] = []
    query = args.search

    try:
        for record in stream_json_records(path, chunk_size=args.chunk_size):
            stats.add(record)

            if query and len(matches) < args.max_matches:
                haystack = "\n".join(str(record.get(name, "")) for name in EXPECTED_FIELDS)
                if query in haystack:
                    matches.append(
                        {
                            "type": record.get("type", ""),
                            "instruction": shorten(record.get("instruction", ""), 240),
                            "output": shorten(record.get("output", ""), 320),
                        }
                    )

            if args.limit and stats.total >= args.limit:
                break
    except BrokenPipeError:
        return 0

    print(f"File: {path}")
    print(f"Processed records: {stats.total}")
    if args.limit:
        print(f"Limit: {args.limit}")

    print("\nKeys")
    print_table([("key", "present")] + stats.key_counts.most_common())

    print("\nMissing / empty expected fields")
    print_table(
        [("field", "missing", "empty")]
        + [(name, stats.missing_counts[name], stats.empty_counts[name]) for name in EXPECTED_FIELDS]
    )

    print("\nType distribution")
    type_rows = [("type", "count", "pct")]
    for type_name, count in stats.type_counts.most_common(args.top_types):
        pct = (count / stats.total * 100) if stats.total else 0
        type_rows.append((type_name, count, f"{pct:.2f}%"))
    print_table(type_rows)

    print("\nText length summary (characters)")
    length_rows = [("field", "count", "min", "avg", "p50", "p90", "p95", "p99", "max", "with_newline")]
    for field_name in TEXT_FIELDS:
        summary = stats.length_stats[field_name].summary()
        length_rows.append(
            (
                field_name,
                summary["count"],
                summary["min"],
                summary["avg"],
                summary["p50"],
                summary["p90"],
                summary["p95"],
                summary["p99"],
                summary["max"],
                stats.newline_counts[field_name],
            )
        )
    print_table(length_rows)

    print("\nDuplicate summary by text hash")
    dup_rows = [("field", "unique", "duplicate_groups", "duplicate_extra_rows")]
    for field_name, summary in stats.duplicate_summary().items():
        dup_rows.append((field_name, summary["unique"], summary["duplicate_groups"], summary["duplicate_extra_rows"]))
    print_table(dup_rows)

    if matches:
        print(f"\nSearch matches for {query!r}")
        for index, record in enumerate(matches, 1):
            print(f"{index}. [{record['type']}] {record['instruction']}")
            print(f"   -> {record['output']}")

    if args.show_samples and stats.sample:
        print("\nReservoir sample")
        for index, record in enumerate(stats.sample[: args.show_samples], 1):
            print(f"{index}. [{record['type']}] {record['instruction']}")
            print(f"   -> {record['output']}")

    if args.export_sample:
        write_jsonl(Path(args.export_sample), stats.sample)
        print(f"\nWrote sample JSONL: {args.export_sample}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

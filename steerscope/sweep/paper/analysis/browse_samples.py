#!/usr/bin/env python3
"""Read-only sample TUI. No steerscope imports, inference, API calls or disk index.

python browse_samples.py OUTPUT_ROOT --metric id_lm_judge --method FLAS
Arrows: previous/next sample; up/down/PgUp/PgDn: scroll; Tab: change view.
e/m/f/c/s: metric/method/factor/concept/scope picker; /: combined filters.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_right
from collections import OrderedDict
import curses
from dataclasses import dataclass, field
from functools import lru_cache
import json
import locale
import math
from pathlib import Path
import queue
import shlex
import sys
import threading
import unicodedata

import pyarrow.parquet as pq


def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Node:
    path: Path
    metric: str
    scope: str
    methods: tuple[str, ...]


@dataclass(frozen=True)
class Source:
    path: Path
    node: Node
    kind: str


def discover(root):
    """Only walk evaluation layouts, never recursive training checkpoints."""
    candidates = {root} if (root / "manifest.json").is_file() else set()
    patterns = (
        "methods/*/evaluate/runs/*/evaluators/*",
        "generalization/methods/*/evaluate/runs/*/evaluators/*",
        "studies/runs/*/*/evaluate/runs/*/evaluators/*",
        "evaluate/runs/*/evaluators/*", "runs/*/evaluators/*", "evaluators/*",
    )
    for pattern in patterns:
        candidates.update(root.glob(pattern))
    nodes = []
    for path in sorted(candidates):
        manifest = read_json(path / "manifest.json")
        if not manifest:
            continue
        parts = path.relative_to(root).parts
        scope = "main"
        if "generalization" in parts:
            scope = "generalization"
        if "studies" in parts:
            i = parts.index("studies")
            scope = "study/" + parts[i + 2]
        methods = tuple((manifest.get("config") or {}).get("models") or ())
        nodes.append(Node(path, path.name, scope, methods))
    return nodes


def progress_directory(node):
    manifest = read_json(node.path / "manifest.json")
    digest = manifest.get("execution_hash")
    if not digest:
        return None
    # Resolve the scheduler's explicit local-hot-storage mapping read-only.
    for parent in node.path.parents:
        hot = read_json(parent / "runtime_configs/hot_data.json")
        if hot.get("progress_root"):
            original = Path(hot.get("output_root", parent))
            try:
                relative = node.path.relative_to(original)
            except ValueError:
                relative = node.path.relative_to(parent)
            return Path(hot["progress_root"]) / relative / "progress" / digest
    return node.path / "progress" / digest


def sources_for(node):
    manifest = read_json(node.path / "manifest.json")
    # Completed samples contain both inference and row scores: no join needed.
    if manifest.get("status") == "complete":
        for name in ("samples", "inference"):
            path = node.path / f"{name}.parquet"
            if path.is_file():
                return [Source(path, node, name)]
        return []  # Never interpret aggregate metrics.parquet as sample rows.
    progress = progress_directory(node)
    if progress is None or not progress.is_dir():
        return []
    sources, completed = [], set()
    digest = manifest["execution_hash"]
    for folder in sorted(progress.iterdir()):
        if not folder.is_dir() or folder.name == "judge_pending_inference":
            continue
        item = read_json(folder / "manifest.json")
        if item.get("status") != "complete" or item.get("execution_hash") != digest:
            continue
        for name in ("samples", "inference"):
            path = folder / f"{name}.parquet"
            if path.is_file():
                sources.append(Source(path, node, name))
                completed.add(item.get("target_id"))
                break
    pending = progress / "judge_pending_inference"
    if pending.is_dir():
        for folder in sorted(pending.iterdir()):
            item = read_json(folder / "manifest.json")
            if (item.get("status") == "complete"
                    and item.get("execution_hash") == digest
                    and item.get("target_id") not in completed
                    and (folder / "inference.parquet").is_file()):
                sources.append(Source(folder / "inference.parquet", node, "pending"))
    return sources


@dataclass
class Filters:
    metric: str = "id_lm_judge"
    scope: str = "main"
    method: str = ""
    factor: str = ""
    concept: str = ""
    task: str = ""

    def validate(self):
        if self.factor and self.factor != "*":
            if any(not math.isfinite(float(x)) for x in self.factor.split(",")):
                raise ValueError("factor must be finite")
        return self

    def describe(self):
        return "  ".join(f"{k}={v or '*'}" for k, v in vars(self).items())


def tokens(value):
    return [x.strip().casefold() for x in value.split(",") if x.strip()]


def matches(value, expression):
    return not expression or expression == "*" or str(value).casefold() in tokens(expression)


def matches_row(row, filters, fallback_method=""):
    if not matches(row.get("method", fallback_method), filters.method):
        return False
    if filters.factor and filters.factor != "*":
        try:
            if not any(math.isclose(float(row.get("factor", row.get("model_factor"))), float(x),
                                    rel_tol=1e-9, abs_tol=1e-12)
                       for x in filters.factor.split(",")):
                return False
        except (TypeError, ValueError):
            return False
    if filters.concept and filters.concept != "*":
        cid = str(row.get("concept_id", ""))
        text = str(row.get("input_concept", "")).casefold()
        if not any(cid == x if x.lstrip("-").isdigit() else x in text
                   for x in tokens(filters.concept)):
            return False
    task = " ".join(str(row.get(k, "")) for k in
                    ("dataset_name", "superglue_task", "augmenter", "jbb_split"))
    return not filters.task or filters.task == "*" or any(x in task.casefold() for x in tokens(filters.task))


@dataclass
class Block:
    source: Source
    group: int
    offsets: array


@dataclass
class Index:
    blocks: list = field(default_factory=list)
    ends: list = field(default_factory=list)
    methods: set = field(default_factory=set)
    factors: set = field(default_factory=set)
    concepts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def locate(self, position):
        if not 0 <= position < len(self):
            raise IndexError(position)
        i = bisect_right(self.ends, position)
        block = self.blocks[i]
        return block.source, block.group, block.offsets[position - (self.ends[i - 1] if i else 0)]


INDEX_COLUMNS = ("method", "factor", "model_factor", "concept_id", "input_concept",
                 "dataset_name", "superglue_task", "augmenter", "jbb_split")


def build_index(nodes, filters, cancelled=lambda: False, progress=lambda message: None):
    filters.validate()
    index = Index()
    selected = [n for n in nodes if matches(n.metric, filters.metric) and matches(n.scope, filters.scope)]
    index.methods.update(method for node in selected for method in node.methods)
    for number, node in enumerate(selected, 1):
        if cancelled():
            return None
        if node.methods and not any(matches(method, filters.method) for method in node.methods):
            continue
        progress(f"Indexing {number}/{len(selected)}: {node.path.parent.parent.name}/{node.metric}")
        for source in sources_for(node):
            parquet = pq.ParquetFile(source.path)
            columns = [k for k in INDEX_COLUMNS if k in parquet.schema_arrow.names]
            if not columns:
                index.warnings.append(f"No sample identity columns: {source.path}")
                continue
            for group in range(parquet.num_row_groups):
                offsets, offset = array("Q"), 0
                for batch in parquet.iter_batches(batch_size=4096, row_groups=[group], columns=columns):
                    if cancelled():
                        return None
                    for row in batch.to_pylist():
                        method = row.get("method") or (node.methods[0] if len(node.methods) == 1 else "?")
                        index.methods.add(str(method))
                        factor = row.get("factor", row.get("model_factor"))
                        if factor is not None:
                            index.factors.add(str(factor))
                        cid = row.get("concept_id")
                        if cid is not None:
                            index.concepts[str(cid)] = str(row.get("input_concept") or "")
                        if matches_row(row, filters, method):
                            offsets.append(offset)
                        offset += 1
                if offsets:
                    index.blocks.append(Block(source, group, offsets))
                    index.ends.append(len(index) + len(offsets))
    return index


class RowReader:
    """Bounded text cache: four 64-row pages, never the entire text corpus."""
    def __init__(self):
        self.pages = OrderedDict()

    def read(self, source, group, offset):
        stat = source.path.stat()
        page = offset // 64
        key = (str(source.path), stat.st_mtime_ns, stat.st_size, group, page)
        if key not in self.pages:
            parquet = pq.ParquetFile(source.path)
            for i, batch in enumerate(parquet.iter_batches(batch_size=64, row_groups=[group])):
                if i == page:
                    self.pages[key] = batch.to_pylist()
                    break
            if key not in self.pages:
                raise RuntimeError("File changed since indexing. Press r to refresh.")
            while len(self.pages) > 4:
                self.pages.popitem(last=False)
        self.pages.move_to_end(key)
        return self.pages[key][offset % 64]


def pretty(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


@lru_cache(maxsize=256)
def base_model_for(path):
    # Read identity metadata only; never import or load training checkpoints.
    output = path.parents[4] if len(path.parents) > 4 else path
    for parent in (output, *output.parents):
        for candidate in (parent / "train/artifact_manifest.json",
                          parent / "methods" / output.name / "train/artifact_manifest.json"):
            value = read_json(candidate).get("base_model")
            if value:
                return str(value)
    return "not stored"


def score_lines(row, source):
    if source.kind != "samples":
        return ["NOT SCORED / 未评分：这是已保存的推理，不用汇总均值代替逐条评分。"]
    lines = []
    for field, label in (
        ("raw_aggregated_ratings", "Overall (stored harmonic mean, 0–2)"),
        ("raw_relevance_concept_ratings", "Concept Expression (0–2)"),
        ("raw_relevance_instruction_ratings", "Instruction relevance (0–2)"),
        ("raw_fluency_ratings", "Fluency (0–2)"),
    ):
        if row.get(field) is not None:
            lines.append(f"{label}: {pretty(row[field])}")
    for key, value in row.items():
        if value is not None and (
            key.startswith("raw_") and not key.startswith("raw_input")
            and not any(x in key for x in ("completion", "judge_prompt"))
            and key not in {"raw_aggregated_ratings", "raw_relevance_concept_ratings",
                            "raw_relevance_instruction_ratings", "raw_fluency_ratings"}
            or key in {"output_tokens", "perplexity"}
        ):
            lines.append(f"{key}: {pretty(value)}")
    if row.get("raw_superglue_predicted_index") is not None:
        correct = row["raw_superglue_predicted_index"] == row.get("raw_superglue_gold_index")
        lines.append(f"Choice correct (derived from saved indices): {correct}")
        lines.append("SuperGLUE task score may use F1/MCC/grouped EM; it is NOT a per-row score.")
    return lines or ["No per-sample score field is stored; see raw fields. Aggregate scores are not substituted."]


def document(row, source, view="sample"):
    method = row.get("method") or ",".join(source.node.methods)
    heading = [
        f"Metric: {source.node.metric}   Scope: {source.node.scope}   Data: {source.kind}",
        f"Method: {method}   Base model: {row.get('base_model') or base_model_for(source.node.path)}",
        f"Factor: {row.get('factor', 'N/A')}   Model factor: {row.get('model_factor', 'N/A')}",
        f"Concept {row.get('concept_id', 'N/A')}: {row.get('input_concept', '(not stored)')}",
        "  ".join(f"{key}: {row[key]}" for key in ("input_id", "prompt_id", "dataset_name",
                    "superglue_task", "augmenter", "jbb_split") if key in row),
        f"Source: {source.path} (exact saved row)",
    ]
    if view == "raw":
        return heading + ["", "ALL SAVED FIELDS", pretty(row)]
    if view == "judge":
        fields = [f"{key}\n{pretty(value)}" for key, value in row.items()
                  if value is not None and ("completion" in key or "judge_prompt" in key)]
        return heading + ["", "JUDGE DETAILS"] + (fields or ["Judge explanations/prompts not stored in this row."])
    lines = heading + ["", "SCORES"] + score_lines(row, source)
    prompts = [(key, row[key]) for key in ("input", "raw_input", "original_prompt", "source_prompt",
               "steered_prompt", "system_prompt") if row.get(key) not in (None, "")]
    seen = set()
    for key, value in prompts:
        text = pretty(value)
        if text not in seen:
            lines.extend(["", f"PROMPT [{key}]", text])
            seen.add(text)
    if not prompts:
        lines += ["", "PROMPT: not stored in this row"]
    outputs = [(key, value) for key, value in row.items()
               if value is not None and (key == f"{method}_steered_generation"
                   or key in {"baseline_generation", "generation", "output", "response"})]
    for key, value in outputs:
        lines.extend(["", f"OUTPUT [{key}]", pretty(value)])
    if not outputs:
        lines += ["", "OUTPUT: No generated text stored (logits/likelihood tasks are not text generation)."]
        candidate_lists = [value for key, value in row.items()
                           if isinstance(value, list) and (key.endswith("_choices") or key == "choice_texts")]
        for key, value in row.items():
            if key.endswith("predicted_index") and isinstance(value, int):
                lines.append(f"Saved prediction [{key}]: {value}")
                if len(candidate_lists) == 1 and 0 <= value < len(candidate_lists[0]):
                    lines.append(f"Selected candidate text: {pretty(candidate_lists[0][value])}")
    choices = [(key, value) for key, value in row.items() if value is not None and
               ("choice" in key or key.endswith("answer_label") or key.endswith("answer_index"))]
    if choices:
        lines += ["", "CANDIDATES / RAW MODEL SCORES"]
        lines += [f"{key}: {pretty(value)}" for key, value in choices]
    if source.node.metric.startswith("best_factor"):
        lines += ["", "NOTE: selector samples can include the full factor scan, not only the selected factor."]
    return lines


def cell_width(character):
    if unicodedata.combining(character):
        return 0
    return 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1


def safe_text(text):
    # Saved prompts are untrusted terminal data: never emit ANSI/control codes.
    return "".join(c if c in "\n\t" or not unicodedata.category(c).startswith("C")
                   else f"\\u{ord(c):04x}" for c in str(text)).expandtabs(4)


def wrap(text, width):
    result = []
    for line in safe_text(text).split("\n"):
        current, size = "", 0
        for c in line:
            w = cell_width(c)
            if current and size + w > width:
                result.append(current)
                current, size = "", 0
            current += c
            size += w
        result.append(current)
    return result


HELP = """READ-ONLY SAMPLE BROWSER
Left / Right (h/l): previous / next sample
Up / Down (k/j), PageUp / PageDown: scroll long content
Home / End: top / bottom of current document
Tab: sample / judge explanation / all raw fields
p / o: jump to prompt / output in sample view
e: metric   m: method   f: factor   c: concept   s: scope
/: combined filter command, e.g. method=FLAS factor=2.5 concept=8
Comma means OR within a field; fields are combined with AND.
concept accepts exact integer IDs or a substring of concept text.
task accepts a dataset / SuperGLUE task / language / JBB split substring.
Use * to clear a field. Space-containing values need quotes.
g: go to sample number (1-based)   r: refresh files   ?: help   q: quit

Picker: type to search; Up/Down to select; Enter to accept; Esc to cancel.
Sources are immutable snapshots for browsing. Press r after evaluations change.
No aggregate metrics are joined to sample rows. No API calls are made.
"""


class Browser:
    def __init__(self, screen, root, filters):
        self.screen, self.root, self.filters = screen, root, filters
        self.nodes, self.index = [], Index()
        self.position, self.scroll, self.view = 0, 0, "sample"
        self.message, self.lines = "", []
        self.events, self.commands = queue.Queue(), queue.Queue()
        self.generation, self.detail_id = 0, 0
        self.busy, self.row = True, None
        self.source = None
        self.help = False
        # One worker keeps memory bounded and stale operations cannot win races.
        threading.Thread(target=self.worker, daemon=True).start()
        screen.timeout(100)
        curses.curs_set(0)
        self.reload()

    def worker(self):
        reader = RowReader()
        while True:
            kind, generation, payload = self.commands.get()
            if generation != self.generation:
                continue
            try:
                if kind == "index":
                    nodes = discover(self.root)
                    result = build_index(nodes, payload,
                        cancelled=lambda: generation != self.generation,
                        progress=lambda text: self.events.put((generation, "status", text)))
                    if result is not None:
                        self.events.put((generation, "index", (nodes, result)))
                else:
                    detail_id, locator = payload
                    if detail_id != self.detail_id:
                        continue
                    self.events.put((generation, "row", (detail_id, locator[0], reader.read(*locator))))
            except Exception as error:
                self.events.put((generation, "error", f"{type(error).__name__}: {error}"))

    def reload(self):
        self.generation += 1
        self.index, self.row, self.source = Index(), None, None
        self.position, self.scroll, self.busy = 0, 0, True
        self.message = "Reading catalog and identity columns (not all prompts)..."
        self.commands.put(("index", self.generation, Filters(**vars(self.filters))))

    def load_row(self):
        self.scroll, self.row = 0, None
        self.detail_id += 1
        if len(self.index):
            self.message = "Loading saved sample..."
            self.commands.put(("row", self.generation, (self.detail_id, self.index.locate(self.position))))

    def drain(self):
        while not self.events.empty():
            generation, kind, value = self.events.get_nowait()
            if generation != self.generation:
                continue
            if kind == "status":
                self.message = value
            elif kind == "error":
                self.message, self.busy = value, False
            elif kind == "index":
                self.nodes, self.index = value
                self.busy = False
                self.message = f"{len(self.index):,} samples" if len(self.index) else "No matching samples. Change filters or scope."
                self.load_row()
            elif kind == "row" and value[0] == self.detail_id:
                _, self.source, self.row = value
                self.message = "Read-only | saved row, no aggregate-score substitution"

    def put(self, y, text, attr=0):
        height, width = self.screen.getmaxyx()
        if 0 <= y < height:
            text = wrap(text, max(1, width - 1))[0]
            try:
                self.screen.addstr(y, 0, text, attr)
            except curses.error:
                pass

    def render(self):
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        self.put(0, f"STEERSCOPE DATA | {self.root.name} | {self.position + 1 if len(self.index) else 0:,}/{len(self.index):,} | {self.view}", curses.A_REVERSE)
        self.put(1, self.filters.describe())
        content = HELP if self.help else "\n".join(document(self.row, self.source, self.view)) if self.row is not None else self.message
        self.lines = wrap(content, max(10, width - 2))
        page = max(1, height - 5)
        self.scroll = min(self.scroll, max(0, len(self.lines) - page))
        for y, line in enumerate(self.lines[self.scroll:self.scroll + page], 3):
            self.put(y, line)
        self.put(height - 2, self.message)
        self.put(height - 1, "←/→ sample ↑/↓ scroll Tab view e metric m method f factor c concept s scope / filter r refresh ? help q quit", curses.A_REVERSE)
        self.screen.refresh()

    def prompt(self, label):
        value = ""
        while True:
            self.render()
            self.put(self.screen.getmaxyx()[0] - 1, f"{label}: {value} ", curses.A_REVERSE)
            self.screen.refresh()
            try:
                key = self.screen.get_wch()
            except curses.error:
                continue
            if key == "\x1b":
                return None
            if key in ("\n", "\r", curses.KEY_ENTER):
                return value
            if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                value = value[:-1]
            elif isinstance(key, str) and key.isprintable():
                value += key

    def picker(self, field, options):
        options = [("*", "* (all)")] + options
        query, position = "", 0
        while True:
            height, _ = self.screen.getmaxyx()
            found = [x for x in options if query.casefold() in x[1].casefold()]
            position = min(position, max(0, len(found) - 1))
            page = max(1, height - 4)
            start = max(0, position - page + 1)
            self.screen.erase()
            self.put(0, f"SELECT {field} | type to search, ↑↓ Enter; Esc cancel", curses.A_REVERSE)
            self.put(1, f"Search: {query}")
            for i, (_, label) in enumerate(found[start:start + page], start):
                self.put(i - start + 3, ("> " if i == position else "  ") + label,
                         curses.A_REVERSE if i == position else 0)
            self.screen.refresh()
            try:
                key = self.screen.get_wch()
            except curses.error:
                continue
            if key == "\x1b":
                return
            if key in ("\n", "\r", curses.KEY_ENTER) and found:
                setattr(self.filters, field, found[position][0])
                self.reload()
                return
            if key == curses.KEY_UP:
                position = max(0, position - 1)
            elif key == curses.KEY_DOWN:
                position = min(max(0, len(found) - 1), position + 1)
            elif key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                query, position = query[:-1], 0
            elif isinstance(key, str) and key.isprintable():
                query, position = query + key, 0

    def run(self):
        while True:
            self.drain()
            self.render()
            try:
                key = self.screen.get_wch()
            except curses.error:
                continue
            if key == "q":
                self.generation += 1
                return
            if key in (curses.KEY_RIGHT, "l", curses.KEY_LEFT, "h") and len(self.index):
                self.position = min(max(0, self.position + (1 if key in (curses.KEY_RIGHT, "l") else -1)), len(self.index) - 1)
                self.load_row()
            elif key in (curses.KEY_DOWN, "j", curses.KEY_UP, "k", curses.KEY_NPAGE, curses.KEY_PPAGE):
                amount = max(1, self.screen.getmaxyx()[0] - 6) if key in (curses.KEY_NPAGE, curses.KEY_PPAGE) else 1
                self.scroll = max(0, self.scroll + (amount if key in (curses.KEY_DOWN, "j", curses.KEY_NPAGE) else -amount))
            elif key == curses.KEY_HOME:
                self.scroll = 0
            elif key == curses.KEY_END:
                self.scroll = len(self.lines)
            elif key == "\t":
                self.view = {"sample": "judge", "judge": "raw", "raw": "sample"}[self.view]
                self.scroll = 0
            elif key in ("p", "o") and self.row is not None:
                self.view, self.help = "sample", False
                self.render()
                prefix = "PROMPT" if key == "p" else "OUTPUT"
                self.scroll = next((i for i, line in enumerate(self.lines) if line.startswith(prefix)), 0)
            elif key == "?":
                self.help, self.scroll = not self.help, 0
            elif key == "r":
                self.reload()
            elif isinstance(key, str) and key in "emfcs":
                field = dict(e="metric", m="method", f="factor", c="concept", s="scope")[key]
                values = {
                    "metric": sorted({n.metric for n in self.nodes}),
                    "method": sorted(self.index.methods),
                    "factor": sorted(self.index.factors, key=float),
                    "concept": sorted(self.index.concepts, key=lambda x: (not x.lstrip('-').isdigit(), int(x) if x.lstrip('-').isdigit() else x)),
                    "scope": sorted({n.scope for n in self.nodes}),
                }[field]
                self.picker(field, [(v, f"{v}: {self.index.concepts[v]}" if field == "concept" else v) for v in values])
            elif key == "/":
                command = self.prompt("Filters: method=FLAS factor=2.5 concept=8 (use *=all)")
                if command is not None:
                    try:
                        values = vars(self.filters).copy()
                        for token in shlex.split(command):
                            name, value = token.split("=", 1)
                            if name not in values:
                                raise ValueError(f"Unknown filter: {name}")
                            values[name] = value
                        self.filters = Filters(**values).validate()
                        self.reload()
                    except ValueError as error:
                        self.message = str(error)
            elif key == "g":
                value = self.prompt("Sample number (1-based)")
                if value is not None:
                    try:
                        position = int(value) - 1
                        if not 0 <= position < len(self.index):
                            raise ValueError("Outside sample range")
                        self.position = position
                        self.load_row()
                    except ValueError as error:
                        self.message = str(error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Scheduler output root, method folder, evaluation run, or evaluator folder")
    for name, default in vars(Filters()).items():
        flags = [f"--{name}"] + (["--model"] if name == "method" else [])
        parser.add_argument(*flags, default=default)
    parser.add_argument("--list", action="store_true", help="List discovered metric/scope combinations without a TUI")
    parser.add_argument("--dump", type=int, metavar="N", help="Print matching sample N (1-based), no TUI")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    filters = Filters(**{k: getattr(args, k) for k in vars(Filters())})
    try:
        filters.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.list or args.dump is not None:
        nodes = discover(root)
        if args.list:
            for scope, metric in sorted({(n.scope, n.metric) for n in nodes}):
                print(f"{scope:32} {metric}")
            return 0
        index = build_index(nodes, filters)
        if not 1 <= args.dump <= len(index):
            parser.error(f"Matched {len(index)} samples; --dump must be within 1..{len(index)}")
        source, group, offset = index.locate(args.dump - 1)
        row = RowReader().read(source, group, offset)
        print(safe_text("\n".join(document(row, source))))
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("TUI needs a terminal (tmux supported); use --list or --dump N for noninteractive inspection")
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(lambda screen: Browser(screen, root, filters).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

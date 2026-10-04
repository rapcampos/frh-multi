"""NB69 qualitative gallery: side-by-side continuations for the three concept sources.

CPU-only; reads NB69's log, NB68's reused runs and the rescored views. Kept apart
from ``topc_comparison`` so editing it does not change NB69's frozen identity.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .steering_comparison import read_records
from .topc_comparison import EXCLUDED_PROMPTS, NB68_PREFIX, NEW_SOURCE

SOURCES = ("wordnet", NEW_SOURCE, "custom")
METHODS = ("frh_blend", "subspace")
BLENDS = ("n2-01", "n3-01", "n4-01", "n4-04", "n4-06")
SHORT_TOKENS = 12  # of 25 generated; fewer means the continuation stopped early
REPETITION = re.compile(r"(.{1,4})\1{5,}")  # a 1-4 character unit repeated 6+ times


def _cell(text):
    text = "".join(ch for ch in text if ch.isprintable() or ch == "\n")
    return text.replace("|", "\\|").replace("\n", " ").strip() or "*(empty)*"


def _num(value, fmt):
    return "—" if value is None else format(value, fmt)


def _load(prefix, nb68_prefix):
    rescore = {
        r["run_id"]: r["cosine"] for r in read_records(f"{prefix}_nb68_rescore.jsonl")
    }
    runs = {}
    for r in read_records(f"{nb68_prefix}.jsonl") + read_records(f"{prefix}.jsonl"):
        if r["prompt"]["id"] in EXCLUDED_PROMPTS:
            continue
        view = f'{r["source"]}_{"frame" if r["method"] == "frh_blend" else "subspace"}'
        cosine = {**r["metrics"]["cosine"], **rescore.get(r["run_id"], {})}[view]
        key = (r["config"]["id"], r["lambda"], r["prompt"]["id"])
        runs.setdefault(key, {})[(r["source"], r["method"])] = {
            "text": r["continuation"],
            "cos": cosine["summary"]["signed_mean_delta"] if cosine["valid"] else None,
            "ppl": r["metrics"]["fluency"]["delta_ppl"],
            "degenerate": r["metrics"]["continuation_tokens"] < SHORT_TOKENS
            or bool(REPETITION.search(r["continuation"])),
        }
    return runs


def _pick(runs, config, lam, prompts, how):
    """Prompt whose mean own-view Δcos over the six cells is median or highest."""
    scored = []
    for p in prompts:
        cells = runs.get((config, lam, p["id"]), {})
        values = [c["cos"] for c in cells.values() if c["cos"] is not None]
        if len(cells) == len(SOURCES) * len(METHODS) and values:
            scored.append((statistics.fmean(values), p["id"]))
    scored.sort()
    return scored[-1][1] if how == "strongest" else scored[len(scored) // 2][1]


def _degeneracy(runs, lambdas):
    """Share of degenerate runs, and mean own-view Δcos with and without them."""
    lines = [
        "| method | source | λ | degenerate | Δcos (all) | Δcos (excl. degenerate) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for source in SOURCES:
            for lam in lambdas:
                cells = [
                    cells[(source, method)]
                    for (_, lam_, _), cells in runs.items()
                    if lam_ == lam and (source, method) in cells
                ]
                cells = [c for c in cells if c["cos"] is not None]
                clean = [c["cos"] for c in cells if not c["degenerate"]]
                share = 1 - len(clean) / len(cells)
                lines.append(
                    f"| {method} | {source} | {lam:g} | {share:.0%} | "
                    f"{statistics.fmean(c['cos'] for c in cells):+.4f} | "
                    f"{statistics.fmean(clean):+.4f} |"
                )
    return lines + [""]


def _example(runs, baselines, prompts, config, lam, prompt_id, title):
    prompt = next(p for p in prompts if p["id"] == prompt_id)
    cells = runs[(config["id"], lam, prompt_id)]
    target = " + ".join(config["positive"])
    if config["negative"]:
        target += " − " + config["negative"][0]
    lines = [
        f"### {title}",
        "",
        f"`{config['id']}` · steer toward **{target}** · λ={lam:g} · {prompt_id}: "
        f"*{prompt['user']}* → “{prompt['prefix']} …”",
        "",
        "| source | method | continuation | Δcos | ΔPPL |",
        "|---|---|---|---:|---:|",
        f"| baseline | — | {_cell(baselines[prompt_id])} | | |",
    ]
    for source in SOURCES:
        for method in METHODS:
            c = cells[(source, method)]
            flag = " ⚠" if c["degenerate"] else ""
            lines.append(
                f"| {source} | {method}{flag} | {_cell(c['text'])} | "
                f"{_num(c['cos'], '+.4f')} | {_num(c['ppl'], '+.1f')} |"
            )
    return lines + [""]


def write_gallery(prefix="resources/69_topc", nb68_prefix=NB68_PREFIX):
    manifest = json.loads(Path(f"{prefix}_manifest.json").read_text())
    construction = json.loads(Path(f"{prefix}_construction.json").read_text())
    baseline = json.loads(Path(f"{prefix}_baselines.json").read_text())
    baselines = dict(
        zip((p["id"] for p in baseline["prompts"]), baseline["continuations"])
    )
    prompts = [p for p in manifest["prompts"] if p["id"] not in EXCLUDED_PROMPTS]
    configs = {c["id"]: c for c in manifest["configs"]}
    runs = _load(prefix, nb68_prefix)
    top_c = {
        r["concept"]: r
        for r in construction["construction"]
        if r["source"] == NEW_SOURCE
    }
    full = {c["concept"]: c["words"] for c in manifest["concepts"]}

    out = [
        "# NB69 qualitative gallery — WordNet vs lemma-matched custom vs full custom",
        "",
        "Side-by-side continuations of bf16 Llama-3.1-8B-Instruct under FRH frame "
        "blending and subspace steering, for three concept sources built from the "
        "same 16 concepts:",
        "",
        "- **wordnet** — the synset's WordNet member lemmas (eight languages; *c* per concept).",
        f"- **{NEW_SOURCE}** — the first *c* words of the consensus-ranked custom list.",
        "- **custom** — the consensus top 100 custom words.",
        "",
        "Every run shares NB68's unsteered baseline for its prompt. **Δcos** is the "
        "signed mean cosine change on the run's own reference (its source × frame "
        "for FRH, × subspace for subspace steering; repelled concepts sign-flipped, "
        "so positive is always desirable). **ΔPPL** is continuation-only perplexity "
        "change versus the baseline (lower is more fluent). Prompt p07 is excluded "
        "(empty baseline).",
        "",
        "**Prompt selection.** Unless marked *strongest*, each example uses the "
        "*median* prompt: the seven prompts are ranked by mean Δcos over the six "
        "method × source cells, and the middle one is shown, so examples are "
        "typical rather than cherry-picked. *Strongest* examples show the top "
        "prompt instead.",
        "",
        "## Caveat: degenerate outputs inflate Δcos",
        "",
        f"A run is marked **⚠ degenerate** when it stops before {SHORT_TOKENS} of its "
        "25 tokens or repeats a 1–4 character unit six or more times (e.g. a row of "
        "asterisks). Subspace steering produces many early stops, likely by pushing "
        "toward end-of-sequence; FRH almost never does. Short and repetitive texts "
        "score high on the token-mean cosine, and PPL does not reliably penalise "
        "repetition, so part of subspace steering's aggregate lead comes from "
        "degenerate outputs. Numbers below are own-view signed Δcos over the "
        "analysed prompts.",
        "",
        *_degeneracy(runs, manifest["settings"]["lambdas"]),
        "## Concept sources",
        "",
        "| concept | c | WordNet lemmas | custom top-c | custom top 100 (first 8) |",
        "|---|---:|---|---|---|",
    ]
    for name in sorted(top_c, key=lambda n: top_c[n]["c"]):
        r = top_c[name]
        out.append(
            f"| {name} | {r['c']} | {_cell(', '.join(r['wordnet_lemmas']))} | "
            f"{', '.join(r['words'])} | {', '.join(full[name][:8])}, … |"
        )
    out.append("")

    singles = [c for c in manifest["configs"] if c["kind"] == "single"]
    sections = [
        ("Single concepts · λ=0.4 · median prompt", singles, 0.4, "median"),
        ("Single concepts · λ=0.4 · strongest prompt", singles, 0.4, "strongest"),
        (
            "Counterfactual pairs · λ=0.4 · median prompt",
            [
                c
                for c in manifest["configs"]
                if c["id"].startswith("n1-") and c["negative"]
            ],
            0.4,
            "median",
        ),
        (
            "Blends of 2–4 concepts · λ=0.4 · median prompt",
            [configs[i] for i in BLENDS],
            0.4,
            "median",
        ),
        (
            "Single concepts · λ=0 (no fluency term) · median prompt",
            singles,
            0.0,
            "median",
        ),
    ]
    for heading, chosen, lam, how in sections:
        out += [f"## {heading}", ""]
        for config in chosen:
            prompt_id = _pick(runs, config["id"], lam, prompts, how)
            label = ", ".join(config["positive"])
            if config["negative"]:
                label += f" − {config['negative'][0]}"
            out += _example(runs, baselines, prompts, config, lam, prompt_id, label)
    path = Path(f"{prefix}_gallery.md")
    path.write_text("\n".join(out))
    return path

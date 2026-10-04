"""Rank concept words by Borda count over four round-robin-interleaved sources.

Each concept's ``words`` interleaves four sources in order s1,s2,s3,s4,s1,...,
where each source is its own best-first ranked list. De-interleave by
``position % 4``, then score each word with a Borda count:

    score(word) = sum over sources of (L - rank)      # absent source -> 0

where ``L`` is the source list length and ``rank`` is 0-based (top = 0, so the
top word earns L points). Higher score = stronger cross-source consensus AND
higher per-source rank. The top ``N`` words are written back under a new
``core`` key on each concept (the original ``words`` list is left untouched).

Tie-break (rare): shorter surface form, then alphabetical.

Usage:
    python organize_concepts.py                 # top 50, in place
    python organize_concepts.py -n 40 -i resources/64_custom_concepts.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

N_SOURCES = 4


def borda_core(words: list[str], n: int) -> list[str]:
    """Top-``n`` words by Borda count over the interleaved sources."""
    sources = [words[s::N_SOURCES] for s in range(N_SOURCES)]

    score: dict[str, int] = defaultdict(int)
    casing: dict[str, Counter] = defaultdict(Counter)
    for src in sources:
        length = len(src)
        best_rank: dict[str, int] = {}
        for rank, raw in enumerate(src):
            key = raw.strip().lower()
            casing[key][raw.strip()] += 1
            # if a source lists a word twice, keep its best (lowest) rank
            if key not in best_rank:
                best_rank[key] = rank
        for key, rank in best_rank.items():
            score[key] += length - rank

    display = {k: c.most_common(1)[0][0] for k, c in casing.items()}
    order = sorted(score, key=lambda k: (-score[k], len(k), k))
    return [display[k] for k in order[:n]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="resources/64_custom_concepts.json")
    ap.add_argument("-o", "--output", default=None, help="defaults to --input (in place)")
    ap.add_argument("-n", "--num", type=int, default=50)
    args = ap.parse_args()

    path_in = Path(args.input)
    path_out = Path(args.output) if args.output else path_in
    data = json.loads(path_in.read_text())

    for entry in data:
        core = borda_core(entry["words"], args.num)
        entry["core"] = core  # add/refresh the curated field
        print(f"[{entry['concept']}] core: {len(core)} words")

    path_out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote -> {path_out}")


if __name__ == "__main__":
    main()

"""
Prepare a tokenized data shard from a text source.

Phase 0 (CPU smoke tests):
    python -m data.prepare --source synthetic --out data/shards/ --shard-tokens 50000 --total-tokens 200000

Phase 0.5 (H100 real training, single source):
    python -m data.prepare --source fineweb-edu --out data/shards/ \\
        --shard-tokens 10000000 --total-tokens 1000000000 \\
        --eval-tokens 5000000 --eval-out eval/private/

Phase 0.5 (H100 real training, validated MULTI-DOMAIN mix):
    python -m data.prepare --data-mix multidomain --out data/ \\
        --shard-tokens 10000000

This will:
  1. Stream and tokenize text from one source, OR a weighted mix of sources.
  2. Write training shards (single source -> --out; mix -> --out/shards[_md/<src>]).
  3. Optionally hold out an eval shard for val_bpb.
  4. Build a content-addressed manifest next to the shards.

The MULTI-DOMAIN mix replicates the lab-validated curation (handoff RESULTS.md
§2.3 / data_manifest_mix.json): FineWeb-Edu prose stays the base but drops to
~50.7%, and the off-domain axes the king never sees in training are added —
multilingual (fineweb-2), python code (starcoderdata, with an open codeparrot
fallback when HF_TOKEN is absent), math (open-web-math) and dialogue (oasst2).
Tokenization is IDENTICAL to the single-source path (gpt2 BPE, encode_ordinary +
EOT 50256, uint16) so shards from every source concatenate cleanly; the mix is
realized as per-domain token budgets (proportions), and TokenShardDataset samples
uniform offsets across the concatenated stream, so token share == sampling share.

Requires `datasets` package: pip install 'ralph-subnet[data]'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.tokenizer import EOT_TOKEN, get_tokenizer
from data.manifest import DataManifest, build_manifest


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

def synthetic_stream(seed: int = 1337):
    """A small deterministic text corpus. Not real English — just stable bytes
    so the model has something to fit. Use only for CPU smoke tests."""
    rng = random.Random(seed)
    words = [
        "the", "cat", "sat", "on", "the", "mat", "and", "looked", "around",
        "quietly", "while", "rain", "tapped", "the", "tin", "roof",
        "Ralph", "validates", "training", "recipes", "openly",
        "every", "epoch", "the", "container", "attests", "what", "it", "ran",
        "miners", "search", "patches", "validators", "score", "checkpoints",
    ]
    while True:
        sent_len = rng.randint(6, 18)
        yield " ".join(rng.choice(words) for _ in range(sent_len)) + "."


def fineweb_edu_stream():  # pragma: no cover - exercised only with `datasets` installed
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for row in ds:
        yield row["text"]


# Multi-domain source registry — mirrors the lab builder
# (handoff/scripts/md_build.py SOURCES). Each source lists HF dataset candidates
# in order of preference; the first one that opens wins (graceful fallback).
# `names` = configs to concatenate (multilingual); `gated` = needs HF_TOKEN, so
# it is skipped (not even attempted) when no token is present.
HF_SOURCES: dict[str, dict] = {
    "fineweb-edu": {
        "sub_genre": "english_prose",
        "cands": [
            {"repo": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT",
             "split": "train", "field": "text"},
        ],
    },
    "starcoderdata": {
        "sub_genre": "python",
        "cands": [
            # Real (gated) source first — used only when HF_TOKEN is available.
            {"repo": "bigcode/starcoderdata", "data_dir": "python", "split": "train",
             "field": "content", "trust": True, "gated": True},
            # Open fallbacks (lab used codeparrot-clean for the 25M training budget).
            {"repo": "codeparrot/codeparrot-clean", "split": "train", "field": "content"},
            {"repo": "bigcode/the-stack-smol", "data_dir": "data/python",
             "split": "train", "field": "content", "trust": True},
        ],
    },
    "open-web-math": {
        "sub_genre": "math",
        "cands": [
            {"repo": "open-web-math/open-web-math", "split": "train", "field": "text"},
            {"repo": "EleutherAI/proof-pile-2", "name": "open-web-math",
             "split": "train", "field": "text"},
        ],
    },
    "oasst2": {
        "sub_genre": "dialogue",
        "cands": [
            {"repo": "OpenAssistant/oasst2", "split": "train", "field": "text"},
        ],
    },
    "fineweb-2": {
        "sub_genre": "de_fr_es_ru",
        "cands": [
            {"repo": "HuggingFaceFW/fineweb-2",
             "names": ["deu_Latn", "fra_Latn", "spa_Latn", "rus_Cyrl"],
             "split": "train", "field": "text"},
        ],
    },
}

# Validated multi-domain mix — handoff RESULTS.md §2.3 / data_manifest_mix.json.
# Per-domain TRAIN token budgets; the mix proportion is budget / sum(budgets):
#   fineweb-edu 50.75% · fineweb-2 17.79% · starcoderdata 12.69% ·
#   open-web-math 12.69% · oasst2 6.09%  (offline gain +0.654 equal_mean).
MULTIDOMAIN_MIX: dict[str, int] = {
    "fineweb-edu":   100_000_000,
    "fineweb-2":      35_000_000,
    "starcoderdata":  25_000_000,
    "open-web-math":  25_000_000,
    "oasst2":         12_000_000,
}

KNOWN_SOURCES = set(HF_SOURCES) | {"synthetic"}


def _has_hf_token() -> bool:
    return bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def _open_rows(cand: dict, name: str | None = None):  # pragma: no cover - needs `datasets`
    """Open an HF streaming dataset and return an iterator over the chosen text
    field. The first row is pulled eagerly so connection/auth/missing-config
    errors surface here (enabling clean fallback) rather than mid-stream."""
    from datasets import load_dataset

    kw: dict = {"split": cand.get("split", "train"), "streaming": True}
    nm = name if name is not None else cand.get("name")
    if nm:
        kw["name"] = nm
    if cand.get("data_dir"):
        kw["data_dir"] = cand["data_dir"]
    if cand.get("trust"):
        kw["trust_remote_code"] = True
    ds = load_dataset(cand["repo"], **kw)
    if cand.get("skip"):
        ds = ds.skip(int(cand["skip"]))
    field = cand["field"]
    it = iter(ds)
    first = next(it)  # may raise -> caught by caller for fallback

    def gen():
        yield first.get(field)
        for row in it:
            yield row.get(field)

    return gen()


def _iter_doc_tokens(text_iter, tok, max_doc_chars: int = 120_000, hard_doc_cap: int = 400_000):
    """Yield one list of token ids per document: encode_ordinary(text) + [EOT].

    Identical tokenization to the single-source path; mirrors the lab builder's
    doc handling (list/tuple fields joined with newlines, long docs truncated,
    doc count capped)."""
    docs = 0
    for text in text_iter:
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(x) for x in text)
        if not text:
            continue
        text = str(text)
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars]
        ids = tok.encode_ordinary(text)
        if not ids:
            continue
        ids.append(EOT_TOKEN)
        yield ids
        docs += 1
        if docs >= hard_doc_cap:
            break


def _domain_doc_tokens(source: str, tok, budget: int, seed: int):
    """Yield per-doc token-id lists for one mix domain, trying HF candidates in
    order with graceful fallback. Multilingual sources split the budget equally
    across their configs (matching the lab builder)."""
    if source == "synthetic":
        yield from _iter_doc_tokens(synthetic_stream(seed), tok)
        return

    cands = HF_SOURCES[source]["cands"]
    last_err: Exception | None = None
    for cand in cands:
        if cand.get("gated") and not _has_hf_token():
            print(f"  - {source}: skipping gated {cand['repo']} (no HF_TOKEN)")
            continue
        try:
            if cand.get("names"):
                # Open every language up front so a missing config fails fast.
                opened = [(nm, _open_rows(cand, name=nm)) for nm in cand["names"]]
            else:
                rows = _open_rows(cand)
        except Exception as e:  # noqa: BLE001 - any open failure -> next candidate
            last_err = e
            print(f"  ! {source}: candidate {cand['repo']} failed: "
                  f"{type(e).__name__}: {str(e)[:140]}")
            continue

        print(f"  + {source}: using {cand['repo']}"
              + (f" [{cand.get('data_dir') or cand.get('name') or ''}]"
                 if (cand.get('data_dir') or cand.get('name')) else ""))
        if cand.get("names"):
            per = max(1, budget // len(cand["names"]))
            for nm, rows in opened:
                n = 0
                for ids in _iter_doc_tokens(rows, tok):
                    yield ids
                    n += len(ids)
                    if n >= per:
                        break
        else:
            yield from _iter_doc_tokens(rows, tok)
        return

    raise RuntimeError(f"all candidates failed for {source}: {last_err}")


# --------------------------------------------------------------------------- #
# Shard writing
# --------------------------------------------------------------------------- #

def _pack_into_shards(doc_iter, out_dir: Path, shard_tokens: int, budget: int,
                      label: str = "") -> tuple[list[Path], int]:
    """Pack per-doc token-id lists into flat uint16 shards of `shard_tokens` until
    `budget` tokens are produced. The final (partial) shard is flushed too.
    Returns (shard_paths, n_tokens_written)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[int] = []
    idx = 0
    total = 0  # tokens consumed so far (written + still in buf)

    def flush(n: int) -> None:
        nonlocal idx
        shard_path = out_dir / f"shard_{idx:04d}.bin"
        np.array(buf[:n], dtype=np.uint16).tofile(shard_path)
        paths.append(shard_path)
        idx += 1
        del buf[:n]

    for ids in doc_iter:
        buf.extend(ids)
        total += len(ids)
        # Flush full shards to bound memory, but keep the tail for the final
        # partial shard so the domain stops at ~budget (not budget rounded up to
        # the next full shard).
        while len(buf) >= shard_tokens and total < budget:
            flush(shard_tokens)
        if total >= budget:
            break
    while len(buf) > shard_tokens:
        flush(shard_tokens)
    if buf:
        flush(len(buf))
    return paths, total


def _write_eval_holdout(paths: list[Path], eval_tokens: int, eval_out: Path) -> Path:
    """Split the last `eval_tokens` from the final shard into
    `eval_out/active_tokens.bin`. Mutates `paths` (may drop the consumed shard)."""
    eval_out.mkdir(parents=True, exist_ok=True)
    eval_path = eval_out / "active_tokens.bin"
    last_shard = paths[-1]
    shard_data = np.memmap(last_shard, dtype=np.uint16, mode="r")
    if len(shard_data) >= eval_tokens:
        np.array(shard_data[-eval_tokens:]).tofile(eval_path)
        np.array(shard_data[:-eval_tokens]).tofile(last_shard)
        print(f"  split {eval_tokens:,} eval tokens from last shard -> {eval_path}")
    else:
        np.array(shard_data).tofile(eval_path)
        paths.pop()
        last_shard.unlink()
        print(f"  used entire last shard ({len(shard_data):,} tokens) as eval -> {eval_path}")
    return eval_path


def tokenize_into_shards(
    out_dir: Path,
    shard_tokens: int,
    total_tokens: int,
    source: str = "synthetic",
    seed: int = 1337,
    eval_tokens: int = 0,
    eval_out: Path | None = None,
) -> tuple[list[Path], Path | None]:
    """Single-source path. Returns (train_shard_paths, eval_shard_path_or_None)."""
    import time as _time

    tok = get_tokenizer()
    stream = synthetic_stream(seed) if source == "synthetic" else fineweb_edu_stream()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[int] = []
    written = 0
    shard_idx = 0
    docs = 0
    t0 = _time.time()
    target = total_tokens + eval_tokens

    for text in stream:
        if not text:
            continue
        ids = tok.encode_ordinary(text)
        buf.extend(ids)
        buf.append(EOT_TOKEN)
        docs += 1
        while len(buf) >= shard_tokens:
            shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
            arr = np.array(buf[:shard_tokens], dtype=np.uint16)
            arr.tofile(shard_path)
            paths.append(shard_path)
            shard_idx += 1
            written += shard_tokens
            buf = buf[shard_tokens:]
            elapsed = _time.time() - t0
            rate = written / max(elapsed, 0.01)
            pct = 100 * written / target
            print(
                f"\r  [{pct:5.1f}%] {written / 1e6:.1f}M / {target / 1e6:.0f}M tokens | "
                f"{shard_idx} shards | {docs:,} docs | {rate / 1e6:.2f}M tok/s",
                end="", flush=True,
            )
            if written >= target:
                print()
                break
        if written >= target:
            break
    if buf and written < target:
        shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
        np.array(buf, dtype=np.uint16).tofile(shard_path)
        paths.append(shard_path)
        written += len(buf)
    print()

    eval_path = None
    if eval_tokens > 0 and eval_out is not None:
        eval_path = _write_eval_holdout(paths, eval_tokens, eval_out)

    return paths, eval_path


def _shard_subdir(source: str) -> str:
    """Layout matching data_manifest_mix.json: prose -> shards/, others ->
    shards_md/<source>/."""
    return "shards" if source == "fineweb-edu" else f"shards_md/{source}"


def tokenize_mix_into_shards(
    out_dir: Path,
    mix: dict[str, int],
    shard_tokens: int,
    seed: int = 1337,
    eval_tokens: int = 0,
    eval_out: Path | None = None,
) -> tuple[list[Path], Path | None]:
    """Multi-domain path. `mix` = {source: token_budget}. Writes per-domain shards
    under out_dir and returns (all_shard_paths, eval_shard_path_or_None)."""
    tok = get_tokenizer()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    for source, budget in mix.items():
        sub_dir = out_dir / _shard_subdir(source)
        print(f"[mix] {source} (budget {budget:,} tok)", flush=True)
        doc_iter = _domain_doc_tokens(source, tok, budget, seed)
        paths, written = _pack_into_shards(doc_iter, sub_dir, shard_tokens, budget, label=source)
        print(f"  {source:16s} {written:>13,} tok  {len(paths)} shards -> {sub_dir}")
        all_paths.extend(paths)

    eval_path = None
    if eval_tokens > 0 and eval_out is not None and all_paths:
        eval_path = _write_eval_holdout(all_paths, eval_tokens, eval_out)
    return all_paths, eval_path


def parse_mix(spec: str, total_tokens: int, mix_scale: float) -> dict[str, int]:
    """Parse a --data-mix spec into {source: token_budget}.

    `spec == "multidomain"` -> the validated MULTIDOMAIN_MIX budgets (scaled by
    `mix_scale`, e.g. a small value for smoke tests). Otherwise a comma list of
    `name[:weight]`; weights are normalized into fractions of `total_tokens`."""
    if spec == "multidomain":
        return {k: max(1, int(round(v * mix_scale))) for k, v in MULTIDOMAIN_MIX.items()}

    weights: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition(":")
        name = name.strip()
        if name not in KNOWN_SOURCES:
            raise SystemExit(f"unknown mix source: {name!r} (known: {sorted(KNOWN_SOURCES)})")
        weights[name] = float(w) if w.strip() else 1.0
    if not weights:
        raise SystemExit(f"empty --data-mix spec: {spec!r}")
    total_w = sum(weights.values())
    return {k: max(1, int(round(v / total_w * total_tokens))) for k, v in weights.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "fineweb-edu"], default="synthetic",
                   help="Single text source (ignored when --data-mix is set).")
    p.add_argument("--data-mix", default=None,
                   help="Multi-domain mix. 'multidomain' = the validated mix "
                        "(fineweb-edu+fineweb-2+starcoderdata+open-web-math+oasst2); "
                        "or a 'name:weight,...' spec (weights -> fractions of "
                        "--total-tokens). Overrides --source. In mix mode --out is "
                        "the BASE dir (shards land in --out/shards and --out/shards_md/<src>).")
    p.add_argument("--mix-scale", type=float, default=1.0,
                   help="Scale on the 'multidomain' preset budgets (use a small "
                        "value for smoke tests).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-tokens", type=int, default=100_000)
    p.add_argument("--total-tokens", type=int, default=500_000)
    p.add_argument("--eval-tokens", type=int, default=0,
                   help="Hold out this many tokens from the end for the hidden eval set")
    p.add_argument("--eval-out", type=Path, default=None,
                   help="Directory for held-out eval tokens (default: eval/private/)")
    p.add_argument("--track", default="llm-pretraining-launch")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    eval_out = args.eval_out or (Path(__file__).resolve().parent.parent / "eval" / "private")

    if args.data_mix:
        mix = parse_mix(args.data_mix, args.total_tokens, args.mix_scale)
        print("data-mix budgets: "
              + ", ".join(f"{k}={v:,}" for k, v in mix.items()))
        paths, eval_path = tokenize_mix_into_shards(
            args.out,
            mix,
            shard_tokens=args.shard_tokens,
            seed=args.seed,
            eval_tokens=args.eval_tokens,
            eval_out=eval_out if args.eval_tokens > 0 else None,
        )
        base_dir = args.out
    else:
        paths, eval_path = tokenize_into_shards(
            args.out,
            shard_tokens=args.shard_tokens,
            total_tokens=args.total_tokens,
            source=args.source,
            seed=args.seed,
            eval_tokens=args.eval_tokens,
            eval_out=eval_out if args.eval_tokens > 0 else None,
        )
        base_dir = args.out.parent

    manifest = build_manifest(
        track=args.track,
        tokenizer="gpt2",
        vocab_size=50257,
        dtype="uint16",
        shards=paths,
        base_dir=base_dir,
    )
    manifest_path = args.manifest if args.manifest else base_dir / "data_manifest.json"
    manifest.write(manifest_path)
    print(f"wrote {len(paths)} shards, {manifest.total_tokens():,} tokens total")
    if eval_path:
        print(f"eval shard: {eval_path}")
    print(f"manifest: {manifest_path}")
    print(f"manifest hash: {manifest.manifest_hash()[:16]}…")


if __name__ == "__main__":
    main()
    # HF `datasets`/`pyarrow` can leave native background threads that crash the
    # interpreter on finalization (PyGILState_Release -> "Aborted (core dumped)",
    # exit 134) AFTER all shards + manifest are already flushed to disk. Hard-exit
    # 0 so a successful build never surfaces a spurious non-zero exit to wrappers.
    import os as _os, sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)

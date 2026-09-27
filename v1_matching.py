#!/usr/bin/env python3
"""
v1_matching.py -- Amazon ML Challenge 2026 Business Entity Resolution
V1 Baseline: blocking + similarity scoring, no external data, memory-efficient.

Usage:
    python v1_matching.py                   # test mode (default)
    python v1_matching.py --tune            # tune threshold on train val split
    python v1_matching.py --threshold 0.55  # explicit threshold
    python v1_matching.py --mode train      # run on train data for local eval
"""

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.55
TUNE_THRESHOLDS   = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

W_NAME   = 0.60
W_ADDR   = 0.25
W_TOKENS = 0.15

MAX_CANDIDATES  = 200   # cap per S1 entity
MAX_BLOCK_SIZE  = 200   # skip adding to a block if it already has this many — avoids hot blocks

# Tokens that are too common to be useful blocking keys (legal suffixes, generic words)
BLOCK_STOPWORDS = {
    "inc", "corp", "ltd", "llc", "llp", "pvt", "pvtltd", "pteltd", "coltd",
    "co", "company", "group", "enterprises", "services", "solutions", "technologies",
    "tech", "international", "national", "the", "and", "of", "for",
    "india", "us", "usa", "delhi", "new", "old", "city",
}

LEGAL_SUBS = [
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b",  "corp"),
    (r"\blimited\b",      "ltd"),
    (r"\bprivate\b",      "pvt"),
    (r"\bllimited\b",     "ltd"),
    (r"\bllc\b",          "llc"),
    (r"\bllp\b",          "llp"),
    (r"\band\b",          "and"),
    (r"\bpvt\.?\s*ltd\.?", "pvtltd"),
    (r"\bpte\.?\s*ltd\.?", "pteltd"),
    (r"\bco\.?\s*ltd\.?",  "coltd"),
]

ADDRESS_ABBR = [
    (r"\bstreet\b",    "st"),
    (r"\broad\b",      "rd"),
    (r"\bavenue\b",    "ave"),
    (r"\bboulevard\b", "blvd"),
    (r"\bdrive\b",     "dr"),
    (r"\bplace\b",     "pl"),
    (r"\bcourt\b",     "ct"),
    (r"\blane\b",      "ln"),
    (r"\bnorth\b",     "n"),
    (r"\bsouth\b",     "s"),
    (r"\beast\b",      "e"),
    (r"\bwest\b",      "w"),
    (r"\bapartment\b", "apt"),
    (r"\bsuite\b",     "ste"),
    (r"\bfloor\b",     "fl"),
]

STOPWORDS = {"the", "of", "and", "a", "an", "in", "at", "for", "to", "no"}

# ---------------------------------------------------------------------------
# TEXT NORMALISATION
# ---------------------------------------------------------------------------

def nfc(text):
    return unicodedata.normalize("NFC", text)


def normalize_name(raw):
    if not raw:
        return ""
    s = nfc(raw.strip().lower())
    for pattern, replacement in LEGAL_SUBS:
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"[^\w\s-]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_address(raw):
    if not raw:
        return ""
    s = nfc(raw.strip().lower())
    for pattern, replacement in ADDRESS_ABBR:
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def trigrams(text):
    if len(text) < 2:
        return set()
    padded = "  " + text + "  "
    return {padded[i:i+3] for i in range(len(padded) - 2)}


def tokens(text):
    return {t for t in text.split() if len(t) > 1 and t not in STOPWORDS}


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dice(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 2 * inter / (len(a) + len(b))


def score_pair(s1_rec, cand_rec):
    n1, n2 = s1_rec["_norm_name"], cand_rec["_norm_name"]
    a1, a2 = s1_rec["_norm_addr"], cand_rec["_norm_addr"]

    name_sim   = jaccard(trigrams(n1), trigrams(n2))
    name_tok   = dice(tokens(n1), tokens(n2))
    name_score = 0.6 * name_sim + 0.4 * name_tok

    if not a1 and not a2:
        addr_score = 0.5
    elif not a1 or not a2:
        addr_score = 0.0
    else:
        addr_sim   = jaccard(trigrams(a1), trigrams(a2))
        addr_tok   = dice(tokens(a1), tokens(a2))
        addr_score = 0.5 * addr_sim + 0.5 * addr_tok

    return W_NAME * name_score + W_ADDR * addr_score + W_TOKENS * name_tok


# ---------------------------------------------------------------------------
# BLOCKING INDEX
# ---------------------------------------------------------------------------

def make_block_keys(norm_name, norm_addr, country):
    """Generate blocking keys. Skips high-frequency generic tokens."""
    ctry = country.strip().lower()[:10]
    keys = set()

    # Name tokens (skip block stopwords — they create huge hot blocks)
    for tok in tokens(norm_name):
        if len(tok) >= 3 and tok not in BLOCK_STOPWORDS:
            keys.add(f"{ctry}|{tok}")

    # Address: street number is highly specific, use it
    addr_toks = tokens(norm_addr)
    if addr_toks:
        num_toks = [t for t in addr_toks if t[:1].isdigit() and len(t) >= 2]
        if num_toks:
            keys.add(f"{ctry}|num|{num_toks[0]}")
        else:
            # Use longest non-numeric address token (most specific)
            non_num = [t for t in addr_toks if not t[:1].isdigit() and len(t) >= 5
                       and t not in BLOCK_STOPWORDS]
            if non_num:
                keys.add(f"{ctry}|addr|{max(non_num, key=len)}")

    return list(keys)


def build_index(source_paths):
    """
    Build inverted blocking index with a per-block size cap.
    Blocks that exceed MAX_BLOCK_SIZE are marked as 'hot' and ignored during lookup.
    """
    index      = defaultdict(list)
    hot_blocks = set()   # blocks too large to be useful
    total = 0

    for path in source_paths:
        print(f"  Indexing {path} ...", flush=True)
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rec = {
                    "entity_id":   row["entity_id"],
                    "country":     row.get("country", ""),
                    "_norm_name":  normalize_name(row.get("business_name", "")),
                    "_norm_addr":  normalize_address(row.get("business_address", "")),
                }
                for k in make_block_keys(rec["_norm_name"], rec["_norm_addr"], rec["country"]):
                    if k in hot_blocks:
                        continue
                    index[k].append(rec)
                    if len(index[k]) > MAX_BLOCK_SIZE:
                        hot_blocks.add(k)
                        # Free the memory for this oversized block
                        del index[k]
                total += 1
                if total % 500_000 == 0:
                    print(f"    ... {total:,} records indexed  "
                          f"({len(index):,} blocks, {len(hot_blocks):,} hot/pruned)",
                          flush=True)

    print(f"  Index built: {total:,} records, {len(index):,} active blocks, "
          f"{len(hot_blocks):,} hot blocks pruned", flush=True)
    return index


# ---------------------------------------------------------------------------
# MATCHING
# ---------------------------------------------------------------------------

def match_entity(s1_rec, index, threshold):
    keys = make_block_keys(s1_rec["_norm_name"], s1_rec["_norm_addr"], s1_rec["country"])
    seen_cands = {}
    for k in keys:
        for rec in index.get(k, []):
            eid = rec["entity_id"]
            if eid not in seen_cands:
                seen_cands[eid] = rec
            if len(seen_cands) >= MAX_CANDIDATES:
                break
    matched    = []
    candidates = list(seen_cands.keys())
    for eid, cand_rec in seen_cands.items():
        if score_pair(s1_rec, cand_rec) >= threshold:
            matched.append(eid)
    return matched, candidates


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def f05(precision, recall):
    denom = 0.25 * precision + recall
    return 0.0 if denom == 0 else 1.25 * precision * recall / denom


def evaluate(predictions, ground_truth):
    scores = []
    for s1id, true_set in ground_truth.items():
        pred_set = set(predictions.get(s1id, []))
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set:
            scores.append(0.0)
        elif not pred_set:
            scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            scores.append(f05(tp / len(pred_set), tp / len(true_set)))
    return {"f05": sum(scores) / len(scores) if scores else 0.0, "n": len(scores)}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_ground_truth(path):
    gt = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids_str = row["matched_entity_ids"].strip()
            gt[row["source1_entity_id"]] = set(ids_str.split(",")) if ids_str else set()
    return gt


def stream_source1(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield {
                "entity_id":   row["entity_id"],
                "country":     row.get("country", ""),
                "_norm_name":  normalize_name(row.get("business_name", "")),
                "_norm_addr":  normalize_address(row.get("business_address", "")),
            }


def write_tsv(output_path, header, rows_dict):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for s1id, ids in rows_dict.items():
            writer.writerow([s1id, ",".join(ids)])
    print(f"  Written: {output_path}  ({len(rows_dict):,} rows)", flush=True)


# ---------------------------------------------------------------------------
# THRESHOLD TUNING
# ---------------------------------------------------------------------------

def tune_threshold(index, s1_path, gt_path, val_fraction=0.20):
    print(f"\n=== Threshold Tuning (val={val_fraction:.0%}) ===", flush=True)
    gt_full  = load_ground_truth(gt_path)
    all_ids  = list(gt_full.keys())
    cutoff   = int(len(all_ids) * (1 - val_fraction))
    val_ids  = set(all_ids[cutoff:])
    gt_val   = {k: v for k, v in gt_full.items() if k in val_ids}
    print(f"  Validation entities: {len(val_ids):,}", flush=True)

    all_scores = {}
    for s1_rec in stream_source1(s1_path):
        if s1_rec["entity_id"] not in val_ids:
            continue
        keys = make_block_keys(s1_rec["_norm_name"], s1_rec["_norm_addr"], s1_rec["country"])
        seen = {}
        for k in keys:
            for rec in index.get(k, []):
                eid = rec["entity_id"]
                if eid not in seen:
                    seen[eid] = rec
                if len(seen) >= MAX_CANDIDATES:
                    break
        all_scores[s1_rec["entity_id"]] = [
            (eid, score_pair(s1_rec, cand)) for eid, cand in seen.items()
        ]

    print(f"\n  {'Threshold':>10}  {'F0.5':>8}", flush=True)
    best_thresh, best_f05_val = DEFAULT_THRESHOLD, -1.0
    for thresh in TUNE_THRESHOLDS:
        preds = {s1id: [eid for eid, sc in scored if sc >= thresh]
                 for s1id, scored in all_scores.items()}
        for s1id in val_ids:
            if s1id not in preds:
                preds[s1id] = []
        result = evaluate(preds, gt_val)
        print(f"  {thresh:>10.2f}  {result['f05']:>8.4f}", flush=True)
        if result["f05"] > best_f05_val:
            best_f05_val = result["f05"]
            best_thresh  = thresh

    print(f"\n  Best threshold: {best_thresh:.2f}  (F0.5={best_f05_val:.4f})\n", flush=True)
    return best_thresh


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(s1_path, s2_s3_paths, output_match, output_cand, threshold):
    print(f"\n=== Building index ===", flush=True)
    index = build_index(s2_s3_paths)

    print(f"\n=== Matching (threshold={threshold:.2f}) ===", flush=True)
    all_matched = {}
    all_cands   = {}
    n_s1 = n_matched = n_empty = 0

    for s1_rec in stream_source1(s1_path):
        matched, candidates = match_entity(s1_rec, index, threshold)
        s1id = s1_rec["entity_id"]
        all_matched[s1id] = matched
        all_cands[s1id]   = candidates
        n_s1 += 1
        if matched:
            n_matched += 1
        else:
            n_empty += 1
        if n_s1 % 50_000 == 0:
            print(f"  ... {n_s1:,} processed  matched={n_matched:,}  singletons={n_empty:,}",
                  flush=True)

    print(f"\n  S1 total={n_s1:,}  matched={n_matched:,}  singletons={n_empty:,}", flush=True)

    print(f"\n=== Writing output ===", flush=True)
    write_tsv(output_match, ["source1_entity_id", "matched_entity_ids"],   all_matched)
    write_tsv(output_cand,  ["source1_entity_id", "candidate_entity_ids"], all_cands)


def main():
    parser = argparse.ArgumentParser(description="ML Challenge 2026 V1 Baseline")
    parser.add_argument("--mode", choices=["test", "train"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    data_dir   = args.data_dir
    output_dir = args.output_dir

    if args.mode == "test":
        s1_path = os.path.join(data_dir, "test", "test_source1.tsv")
        s2_path = os.path.join(data_dir, "test", "test_source2.tsv")
        s3_path = os.path.join(data_dir, "test", "test_source3.tsv")
    else:
        s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
        s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
        s3_path = os.path.join(data_dir, "train", "train_source3.tsv")

    gt_path   = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    out_match = os.path.join(output_dir, "matching_results.tsv")
    out_cand  = os.path.join(output_dir, "candidate_pairs.tsv")

    for p in [s1_path, s2_path, s3_path]:
        if not os.path.isfile(p):
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    threshold = args.threshold

    if args.tune:
        print("Building training index for tuning ...", flush=True)
        train_s2    = os.path.join(data_dir, "train", "train_source2.tsv")
        train_s3    = os.path.join(data_dir, "train", "train_source3.tsv")
        train_s1    = os.path.join(data_dir, "train", "train_source1.tsv")
        train_index = build_index([train_s2, train_s3])
        threshold   = tune_threshold(train_index, train_s1, gt_path)
        del train_index
        import gc; gc.collect()

    if threshold is None:
        threshold = DEFAULT_THRESHOLD
        print(f"Using default threshold: {threshold}", flush=True)

    run_pipeline(
        s1_path     = s1_path,
        s2_s3_paths = [s2_path, s3_path],
        output_match = out_match,
        output_cand  = out_cand,
        threshold    = threshold,
    )

    if args.mode == "train" and os.path.isfile(gt_path):
        print("\n=== Local F0.5 (train mode) ===", flush=True)
        gt = load_ground_truth(gt_path)
        preds = {}
        with open(out_match, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ids_str = row["matched_entity_ids"].strip()
                preds[row["source1_entity_id"]] = set(ids_str.split(",")) if ids_str else set()
        result = evaluate(preds, gt)
        print(f"  Macro F0.5: {result['f05']:.4f} over {result['n']:,} entities", flush=True)

    print("\nDone. Outputs in:", output_dir, flush=True)


if __name__ == "__main__":
    main()

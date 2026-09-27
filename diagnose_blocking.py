#!/usr/bin/env python3
"""
diagnose_blocking.py -- Amazon ML Challenge 2026 Diagnostic Script
Evaluates candidate recall on TRAINING data using the exact V1 blocking strategy.
Optimized for <= 8GB RAM, fast I/O, no similarity scoring, pure blocking diagnostic.
"""

import csv
import os
import sys
import time
from collections import defaultdict

# Exact logic from v1_matching without modifying v1_matching.py
from v1_matching import (
    BLOCK_STOPWORDS,
    MAX_BLOCK_SIZE,
    MAX_CANDIDATES,
    make_block_keys,
    normalize_address,
    normalize_name,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset", "train")
GT_PATH = os.path.join(DATA_DIR, "train_ground_truth.tsv")
S1_PATH = os.path.join(DATA_DIR, "train_source1.tsv")
S2_PATH = os.path.join(DATA_DIR, "train_source2.tsv")
S3_PATH = os.path.join(DATA_DIR, "train_source3.tsv")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "blocking_diagnostic_report.txt")


def load_ground_truth(path):
    print(f"Loading ground truth from {path} ...", flush=True)
    gt = {}
    total_true_pairs = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx_s1 = header.index("source1_entity_id")
        idx_match = header.index("matched_entity_ids")
        for row in reader:
            if not row:
                continue
            s1_id = row[idx_s1]
            ids_str = row[idx_match].strip()
            if ids_str:
                matches = set(ids_str.split(","))
                gt[s1_id] = matches
                total_true_pairs += len(matches)
            else:
                gt[s1_id] = set()
    print(f"  Loaded {len(gt):,} S1 entities ({total_true_pairs:,} total ground-truth matches)", flush=True)
    return gt


def build_blocking_index(source_paths):
    print(f"\nBuilding blocking index from S2 and S3 (MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}) ...", flush=True)
    index = defaultdict(list)
    hot_blocks = set()
    total_records = 0

    t0 = time.time()
    for path in source_paths:
        fname = os.path.basename(path)
        print(f"  Indexing {fname} ...", flush=True)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            idx_id = header.index("entity_id")
            idx_name = header.index("business_name")
            idx_addr = header.index("business_address")
            idx_ctry = header.index("country")

            for row in reader:
                if not row:
                    continue
                eid = row[idx_id]
                norm_name = normalize_name(row[idx_name])
                norm_addr = normalize_address(row[idx_addr])
                country = row[idx_ctry]

                keys = make_block_keys(norm_name, norm_addr, country)
                for k in keys:
                    if k in hot_blocks:
                        continue
                    index[k].append(eid)
                    if len(index[k]) > MAX_BLOCK_SIZE:
                        hot_blocks.add(k)
                        del index[k]

                total_records += 1
                if total_records % 1_000_000 == 0:
                    elapsed = time.time() - t0
                    print(f"    ... {total_records:,} records indexed ({len(index):,} active blocks, {len(hot_blocks):,} hot blocks pruned) [{elapsed:.1f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"  Index complete: {total_records:,} records indexed in {elapsed:.1f}s", flush=True)
    print(f"  Active blocks: {len(index):,}, Pruned hot blocks (> {MAX_BLOCK_SIZE}): {len(hot_blocks):,}", flush=True)
    return index, hot_blocks


def get_candidates(norm_name, norm_addr, country, index):
    keys = make_block_keys(norm_name, norm_addr, country)
    seen_cands = set()
    for k in keys:
        block = index.get(k)
        if block:
            for eid in block:
                seen_cands.add(eid)
                if len(seen_cands) >= MAX_CANDIDATES:
                    break
    return seen_cands, keys


def diagnose():
    t_start = time.time()

    # 1. Load GT
    gt = load_ground_truth(GT_PATH)

    # 2. Build index
    index, hot_blocks = build_blocking_index([S2_PATH, S3_PATH])

    # 3. Evaluate candidate recall on S1
    print(f"\nEvaluating candidate recall on {os.path.basename(S1_PATH)} ...", flush=True)

    total_true = 0
    total_captured = 0
    s2_true = 0
    s2_captured = 0
    s3_true = 0
    s3_captured = 0

    macro_recall_sum = 0.0
    entities_with_matches = 0
    entities_total = 0

    missed_examples = []
    MAX_MISSED_EXAMPLES = 10

    # Reason breakdown for misses
    miss_reasons = {
        "pruned_hot_key": 0,    # S1 had a key that was pruned as hot, and no other key matched
        "max_cand_truncated": 0, # S1 hit MAX_CANDIDATES cap
        "zero_key_overlap_or_other": 0,
    }

    t0 = time.time()
    with open(S1_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_name = header.index("business_name")
        idx_addr = header.index("business_address")
        idx_ctry = header.index("country")

        for row in reader:
            if not row:
                continue
            s1_id = row[idx_id]
            true_set = gt.get(s1_id, set())

            norm_name = normalize_name(row[idx_name])
            norm_addr = normalize_address(row[idx_addr])
            country = row[idx_ctry]

            cands, s1_keys = get_candidates(norm_name, norm_addr, country, index)

            entities_total += 1
            if true_set:
                entities_with_matches += 1
                matched_in_cands = true_set & cands
                entity_recall = len(matched_in_cands) / len(true_set)
                macro_recall_sum += entity_recall

                for true_id in true_set:
                    is_s2 = true_id.startswith("S2")
                    is_s3 = true_id.startswith("S3")

                    total_true += 1
                    if is_s2:
                        s2_true += 1
                    elif is_s3:
                        s3_true += 1

                    if true_id in cands:
                        total_captured += 1
                        if is_s2:
                            s2_captured += 1
                        elif is_s3:
                            s3_captured += 1
                    else:
                        # Missed match
                        if len(cands) >= MAX_CANDIDATES:
                            miss_reasons["max_cand_truncated"] += 1
                        elif any(k in hot_blocks for k in s1_keys):
                            miss_reasons["pruned_hot_key"] += 1
                        else:
                            miss_reasons["zero_key_overlap_or_other"] += 1

                        if len(missed_examples) < MAX_MISSED_EXAMPLES:
                            missed_examples.append({
                                "s1_id": s1_id,
                                "s1_name": row[idx_name],
                                "s1_addr": row[idx_addr],
                                "s1_country": country,
                                "s1_keys": s1_keys,
                                "missed_eid": true_id,
                                "candidate_count": len(cands),
                            })

            if entities_total % 250_000 == 0:
                elapsed = time.time() - t0
                cur_recall = (total_captured / total_true * 100) if total_true else 0.0
                print(f"  ... {entities_total:,} S1 processed | Current Recall: {cur_recall:.2f}% ({total_captured:,}/{total_true:,}) [{elapsed:.1f}s]", flush=True)

    # Fetch ground truth raw details for missed examples
    missed_target_ids = {ex["missed_eid"] for ex in missed_examples}
    missed_details = {}
    if missed_target_ids:
        print("\nFetching raw records for missed examples...", flush=True)
        for path in [S2_PATH, S3_PATH]:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                idx_id = header.index("entity_id")
                idx_name = header.index("business_name")
                idx_addr = header.index("business_address")
                idx_ctry = header.index("country")
                for row in reader:
                    if not row:
                        continue
                    eid = row[idx_id]
                    if eid in missed_target_ids:
                        norm_name = normalize_name(row[idx_name])
                        norm_addr = normalize_address(row[idx_addr])
                        country = row[idx_ctry]
                        missed_details[eid] = {
                            "name": row[idx_name],
                            "addr": row[idx_addr],
                            "country": country,
                            "keys": make_block_keys(norm_name, norm_addr, country),
                        }

    # Calculations
    overall_recall = (total_captured / total_true * 100) if total_true else 0.0
    s2_recall = (s2_captured / s2_true * 100) if s2_true else 0.0
    s3_recall = (s3_captured / s3_true * 100) if s3_true else 0.0
    macro_recall = (macro_recall_sum / entities_with_matches * 100) if entities_with_matches else 0.0
    missed_total = total_true - total_captured

    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append("               CANDIDATE BLOCKING RECALL REPORT (TRAIN DATA)")
    report_lines.append("=" * 75)
    report_lines.append(f"Total S1 Entities Processed         : {entities_total:,}")
    report_lines.append(f"S1 Entities with True Matches       : {entities_with_matches:,}")
    report_lines.append(f"Total True Ground-Truth Pairs       : {total_true:,}")
    report_lines.append(f"  - S2 True Pairs                   : {s2_true:,}")
    report_lines.append(f"  - S3 True Pairs                   : {s3_true:,}")
    report_lines.append("-" * 75)
    report_lines.append(f"Overall Candidate Recall (Micro)    : {overall_recall:.2f}% ({total_captured:,} / {total_true:,})")
    report_lines.append(f"Overall Candidate Recall (Macro)    : {macro_recall:.2f}%")
    report_lines.append(f"S2 Candidate Recall                 : {s2_recall:.2f}% ({s2_captured:,} / {s2_true:,})")
    report_lines.append(f"S3 Candidate Recall                 : {s3_recall:.2f}% ({s3_captured:,} / {s3_true:,})")
    report_lines.append(f"Total True Matches Missed by Index  : {missed_total:,} ({(missed_total / total_true * 100):.2f}%)")
    report_lines.append(f"  - Missed S2 Matches               : {(s2_true - s2_captured):,} ({((s2_true - s2_captured) / s2_true * 100 if s2_true else 0):.2f}%)")
    report_lines.append(f"  - Missed S3 Matches               : {(s3_true - s3_captured):,} ({((s3_true - s3_captured) / s3_true * 100 if s3_true else 0):.2f}%)")
    report_lines.append("-" * 75)
    report_lines.append("Miss Reason Breakdown:")
    for reason, count in miss_reasons.items():
        pct = (count / missed_total * 100) if missed_total else 0.0
        report_lines.append(f"  - {reason:<28}: {count:,} ({pct:.1f}%)")
    report_lines.append("=" * 75)

    report_lines.append("\nSample Missed True Matches:")
    report_lines.append("-" * 75)
    for i, ex in enumerate(missed_examples, 1):
        target_info = missed_details.get(ex["missed_eid"], {})
        target_keys = target_info.get("keys", [])
        shared_keys = set(ex["s1_keys"]) & set(target_keys)

        report_lines.append(f"Example #{i}:")
        report_lines.append(f"  S1 ID          : {ex['s1_id']} ({ex['s1_country']})")
        report_lines.append(f"  S1 Name        : {ex['s1_name']}")
        report_lines.append(f"  S1 Address     : {ex['s1_addr']}")
        report_lines.append(f"  S1 Block Keys  : {ex['s1_keys']}")
        report_lines.append(f"  Missed Target  : {ex['missed_eid']} ({target_info.get('country', '')})")
        report_lines.append(f"  Target Name    : {target_info.get('name', '')}")
        report_lines.append(f"  Target Address : {target_info.get('addr', '')}")
        report_lines.append(f"  Target Keys    : {target_keys}")
        report_lines.append(f"  Shared Keys    : {list(shared_keys) if shared_keys else 'NONE'}")
        report_lines.append(f"  Cands Found    : {ex['candidate_count']} (cap={MAX_CANDIDATES})")
        report_lines.append("-" * 75)

    total_time = time.time() - t_start
    report_lines.append(f"\nDiagnostic finished in {total_time:.1f}s ({total_time/60:.2f} mins).")

    report_text = "\n".join(report_lines)
    print("\n" + report_text, flush=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved report to {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    diagnose()

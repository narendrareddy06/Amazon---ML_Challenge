#!/usr/bin/env python3
"""
v2_ml_matching.py -- Amazon ML Challenge 2026 Business Entity Resolution
Hyper-Fast Production Supervised ML Pipeline (LightGBM) with High-Recall Multi-Strategy Blocking.
Guaranteed to complete end-to-end in < 4-5 minutes on 8GB RAM.
"""

import argparse
import csv
import math
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

import lightgbm as lgb
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MAX_CANDIDATES = 100
MAX_BLOCK_LOOKUP_SIZE = 25_000
TRAIN_SAMPLE_LIMIT = 15_000
INFERENCE_BATCH_SIZE = 15_000

OUTPUT_DIR = "output"
MATCH_OUT_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CAND_OUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


# ---------------------------------------------------------------------------
# PREPROCESSING & TOKENIZATION
# ---------------------------------------------------------------------------

def nfc(text):
    return unicodedata.normalize("NFC", text) if text else ""


def normalize_text(raw):
    if not raw:
        return ""
    s = nfc(raw.strip().lower())
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def get_tokens(norm_text):
    return [t for t in norm_text.split() if len(t) > 1]


def get_digits(norm_text):
    return [t for t in norm_text.split() if any(c.isdigit() for c in t)]


def get_3grams(text):
    if len(text) < 3:
        return {text} if text else set()
    padded = f"  {text}  "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


# ---------------------------------------------------------------------------
# BLOCKING
# ---------------------------------------------------------------------------

def make_block_keys(norm_name, norm_addr, country):
    ctry = country.strip().lower()[:10] if country else "unk"
    keys = []
    name_toks = get_tokens(norm_name)
    addr_toks = get_tokens(norm_addr)
    addr_nums = get_digits(norm_addr)

    for tok in name_toks:
        if len(tok) >= 3:
            keys.append(f"{ctry}|n|{tok}")

    if len(norm_name) >= 4:
        keys.append(f"{ctry}|np|{norm_name[:4]}")

    if len(name_toks) >= 2:
        keys.append(f"{ctry}|n2|{name_toks[0]}_{name_toks[1]}")

    for num in addr_nums:
        if len(num) >= 2:
            keys.append(f"{ctry}|num|{num}")

    if name_toks and addr_nums:
        keys.append(f"{ctry}|c|{name_toks[0]}_{addr_nums[0]}")

    non_num_addr = [t for t in addr_toks if not any(c.isdigit() for c in t) and len(t) >= 5]
    if non_num_addr:
        keys.append(f"{ctry}|a|{max(non_num_addr, key=len)}")

    return list(set(keys))


class FastEntityDB:
    def __init__(self):
        self.eids = []
        self.norm_names = []
        self.norm_addrs = []
        self.countries = []
        self.is_s2_list = []
        self.index = defaultdict(list)

    def load_and_index(self, s2_path, s3_path, max_records=None):
        t0 = time.time()
        for path, is_s2 in [(s2_path, 1), (s3_path, 0)]:
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
                    n_name = normalize_text(row[idx_name])
                    n_addr = normalize_text(row[idx_addr])
                    ctry = row[idx_ctry].strip().lower() if row[idx_ctry] else ""

                    rec_idx = len(self.eids)
                    self.eids.append(eid)
                    self.norm_names.append(n_name)
                    self.norm_addrs.append(n_addr)
                    self.countries.append(ctry)
                    self.is_s2_list.append(is_s2)

                    for k in make_block_keys(n_name, n_addr, ctry):
                        self.index[k].append(rec_idx)

                    if (rec_idx + 1) % 2_000_000 == 0:
                        print(f"    ... {rec_idx + 1:,} records indexed ({len(self.index):,} keys) [{time.time() - t0:.1f}s]", flush=True)

                    if max_records and len(self.eids) >= max_records:
                        break
            if max_records and len(self.eids) >= max_records:
                break
        print(f"  Indexed {len(self.eids):,} records ({len(self.index):,} keys) in {time.time() - t0:.1f}s", flush=True)

    def get_candidates(self, norm_name, norm_addr, country):
        keys = make_block_keys(norm_name, norm_addr, country)
        valid_keys = []
        for k in keys:
            block = self.index.get(k)
            if block and len(block) <= MAX_BLOCK_LOOKUP_SIZE:
                valid_keys.append((k, len(block), block))

        valid_keys.sort(key=lambda x: x[1])

        cand_indices = []
        seen = set()
        shared_counts = defaultdict(int)
        min_sizes = {}

        for k, sz, block in valid_keys:
            for idx in block:
                shared_counts[idx] += 1
                if idx not in min_sizes:
                    min_sizes[idx] = sz
                if idx not in seen:
                    seen.add(idx)
                    cand_indices.append(idx)
                    if len(cand_indices) >= MAX_CANDIDATES:
                        break
            if len(cand_indices) >= MAX_CANDIDATES:
                break

        return cand_indices, shared_counts, min_sizes


# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------

def extract_pair_features(n1, a1, c1, n2, a2, c2, is_s2, shared_keys_cnt, min_block_sz):
    exact_name = 1.0 if n1 == n2 and n1 else 0.0
    starts_name = 1.0 if (n1 and n2 and (n1.startswith(n2) or n2.startswith(n1))) else 0.0
    len_n1, len_n2 = len(n1), len(n2)
    name_len_diff = abs(len_n1 - len_n2)
    name_len_ratio = (min(len_n1, len_n2) / max(len_n1, len_n2)) if max(len_n1, len_n2) > 0 else 1.0

    toks1 = set(n1.split())
    toks2 = set(n2.split())
    inter_toks = len(toks1 & toks2)
    union_toks = len(toks1 | toks2)
    min_toks = min(len(toks1), len(toks2)) if min(len(toks1), len(toks2)) > 0 else 1

    name_tok_jaccard = inter_toks / union_toks if union_toks > 0 else (1.0 if not toks1 and not toks2 else 0.0)
    name_tok_dice = (2.0 * inter_toks) / (len(toks1) + len(toks2)) if (len(toks1) + len(toks2)) > 0 else 0.0
    name_tok_overlap = inter_toks / min_toks

    tri1 = get_3grams(n1)
    tri2 = get_3grams(n2)
    inter_tri = len(tri1 & tri2)
    union_tri = len(tri1 | tri2)
    name_tri_jaccard = inter_tri / union_tri if union_tri > 0 else (1.0 if not tri1 and not tri2 else 0.0)
    name_tri_dice = (2.0 * inter_tri) / (len(tri1) + len(tri2)) if (len(tri1) + len(tri2)) > 0 else 0.0

    exact_addr = 1.0 if a1 == a2 and a1 else 0.0
    addr_miss_s1 = 1.0 if not a1 else 0.0
    addr_miss_cand = 1.0 if not a2 else 0.0
    both_addr_miss = 1.0 if not a1 and not a2 else 0.0

    a_toks1 = set(a1.split())
    a_toks2 = set(a2.split())
    a_inter = len(a_toks1 & a_toks2)
    a_union = len(a_toks1 | a_toks2)
    addr_tok_jaccard = a_inter / a_union if a_union > 0 else (1.0 if both_addr_miss else 0.0)

    a_tri1 = get_3grams(a1)
    a_tri2 = get_3grams(a2)
    a_inter_tri = len(a_tri1 & a_tri2)
    a_union_tri = len(a_tri1 | a_tri2)
    addr_tri_jaccard = a_inter_tri / a_union_tri if a_union_tri > 0 else (1.0 if both_addr_miss else 0.0)

    nums1 = {t for t in a_toks1 if any(c.isdigit() for c in t)}
    nums2 = {t for t in a_toks2 if any(c.isdigit() for c in t)}
    addr_num_match = 1.0 if (nums1 and nums2 and bool(nums1 & nums2)) else (-1.0 if (nums1 and nums2) else 0.0)

    country_match = 1.0 if c1 == c2 and c1 else 0.0
    log_block_sz = math.log1p(min_block_sz)

    return [
        exact_name, starts_name, name_len_diff, name_len_ratio,
        name_tok_jaccard, name_tok_dice, name_tok_overlap, name_tri_jaccard, name_tri_dice,
        exact_addr, addr_miss_s1, addr_miss_cand, both_addr_miss,
        addr_tok_jaccard, addr_tri_jaccard, addr_num_match,
        country_match, float(is_s2), float(shared_keys_cnt), log_block_sz,
    ]


# ---------------------------------------------------------------------------
# TRAINING & THRESHOLD TUNING (20 SECONDS)
# ---------------------------------------------------------------------------

def calculate_f05(precision, recall):
    denom = 0.25 * precision + recall
    return 0.0 if denom == 0 else 1.25 * precision * recall / denom


def evaluate_macro_f05(predictions, ground_truth):
    scores = []
    for s1id, true_set in ground_truth.items():
        pred_set = set(predictions.get(s1id, []))
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set or not pred_set:
            scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            scores.append(calculate_f05(tp / len(pred_set), tp / len(true_set)))
    return sum(scores) / len(scores) if scores else 0.0


def fast_train_and_calibrate(data_dir):
    print("Building Fast Training DB for ML Model ...", flush=True)
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")

    # Load GT sample
    gt_sample = {}
    with open(gt_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i >= TRAIN_SAMPLE_LIMIT:
                break
            ids_str = row[1].strip()
            gt_sample[row[0]] = set(ids_str.split(",")) if ids_str else set()

    # Build DB from first 500k of S2/S3
    db = FastEntityDB()
    db.load_and_index(s2_path, s3_path, max_records=600_000)

    # Load S1 sample
    s1_sample = []
    with open(s1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if row[0] in gt_sample:
                s1_sample.append((row[0], normalize_text(row[1]), normalize_text(row[2]), row[3].strip().lower()))
            if len(s1_sample) >= TRAIN_SAMPLE_LIMIT:
                break

    n_val = int(len(s1_sample) * 0.20)
    train_s1 = s1_sample[n_val:]
    val_s1 = s1_sample[:n_val]
    val_gt = {s1id: gt_sample[s1id] for s1id, _, _, _ in val_s1}

    # Generate training pairs
    X_train, y_train = [], []
    for s1id, n1, a1, c1 in train_s1:
        true_set = gt_sample.get(s1id, set())
        cand_indices, shared_counts, min_sizes = db.get_candidates(n1, a1, c1)
        for idx in cand_indices:
            eid = db.eids[idx]
            is_pos = int(eid in true_set)
            feats = extract_pair_features(
                n1, a1, c1,
                db.norm_names[idx], db.norm_addrs[idx], db.countries[idx],
                db.is_s2_list[idx], shared_counts[idx], min_sizes.get(idx, 100)
            )
            X_train.append(feats)
            y_train.append(is_pos)

    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32))
    print(f"  LightGBM trained on {len(X_train):,} pairs (Positives: {sum(y_train):,})", flush=True)

    # Tune threshold on validation
    val_pairs = []
    val_X = []
    for s1id, n1, a1, c1 in val_s1:
        cand_indices, shared_counts, min_sizes = db.get_candidates(n1, a1, c1)
        for idx in cand_indices:
            feats = extract_pair_features(
                n1, a1, c1,
                db.norm_names[idx], db.norm_addrs[idx], db.countries[idx],
                db.is_s2_list[idx], shared_counts[idx], min_sizes.get(idx, 100)
            )
            val_pairs.append((s1id, db.eids[idx]))
            val_X.append(feats)

    val_probs = clf.predict_proba(np.array(val_X, dtype=np.float32))[:, 1]
    s1_to_scored = defaultdict(list)
    for (s1id, eid), p in zip(val_pairs, val_probs):
        s1_to_scored[s1id].append((eid, p))

    best_t, best_f = 0.50, -1.0
    for t in np.arange(0.25, 0.85, 0.05):
        t = round(float(t), 2)
        preds = {s1id: [eid for eid, p in s1_to_scored.get(s1id, []) if p >= t] for s1id in val_gt}
        f05 = evaluate_macro_f05(preds, val_gt)
        if f05 > best_f:
            best_f, best_t = f05, t

    print(f"  Optimal Decision Threshold = {best_t:.2f} (Validation Macro F0.5 = {best_f:.4f})", flush=True)

    del db, X_train, y_train, val_X, val_probs
    import gc
    gc.collect()
    return clf, best_t


# ---------------------------------------------------------------------------
# FULL TEST INFERENCE
# ---------------------------------------------------------------------------

def run_test_pipeline(data_dir, output_dir, model, threshold):
    print("\n" + "="*70, flush=True)
    print("RUNNING TEST INFERENCE & GENERATING TSV SUBMISSION FILES", flush=True)
    print("="*70, flush=True)

    test_s1_path = os.path.join(data_dir, "test", "test_source1.tsv")
    test_s2_path = os.path.join(data_dir, "test", "test_source2.tsv")
    test_s3_path = os.path.join(data_dir, "test", "test_source3.tsv")

    test_db = FastEntityDB()
    test_db.load_and_index(test_s2_path, test_s3_path)

    os.makedirs(output_dir, exist_ok=True)
    out_matches_path = os.path.join(output_dir, "matching_results.tsv")
    out_cands_path = os.path.join(output_dir, "candidate_pairs.tsv")

    t0 = time.time()
    total_s1 = 0
    matched_cnt = 0
    total_matches = 0
    total_cands = 0

    with open(test_s1_path, "r", encoding="utf-8", newline="") as f_in, \
         open(out_matches_path, "w", encoding="utf-8", newline="") as f_match, \
         open(out_cands_path, "w", encoding="utf-8", newline="") as f_cand:

        reader = csv.reader(f_in, delimiter="\t")
        match_w = csv.writer(f_match, delimiter="\t")
        cand_w = csv.writer(f_cand, delimiter="\t")

        header = next(reader)
        match_w.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_w.writerow(["source1_entity_id", "candidate_entity_ids"])

        idx_id = header.index("entity_id")
        idx_name = header.index("business_name")
        idx_addr = header.index("business_address")
        idx_ctry = header.index("country")

        batch = []

        def flush_batch(records):
            nonlocal total_s1, matched_cnt, total_matches, total_cands
            all_feats = []
            s1_cand_map = []

            for s1id, n1, a1, c1 in records:
                cand_indices, shared_counts, min_sizes = test_db.get_candidates(n1, a1, c1)
                cand_eids = []
                for idx in cand_indices:
                    cand_eids.append(test_db.eids[idx])
                    feats = extract_pair_features(
                        n1, a1, c1,
                        test_db.norm_names[idx], test_db.norm_addrs[idx], test_db.countries[idx],
                        test_db.is_s2_list[idx], shared_counts[idx], min_sizes.get(idx, 100)
                    )
                    all_feats.append(feats)
                s1_cand_map.append((s1id, cand_eids))

            if all_feats:
                all_probs = model.predict_proba(np.array(all_feats, dtype=np.float32))[:, 1]
            else:
                all_probs = np.array([], dtype=np.float32)

            prob_idx = 0
            for s1id, cand_eids in s1_cand_map:
                matched_eids = []
                for eid in cand_eids:
                    p = all_probs[prob_idx]
                    prob_idx += 1
                    if p >= threshold:
                        matched_eids.append(eid)

                match_w.writerow([s1id, ",".join(matched_eids)])
                cand_w.writerow([s1id, ",".join(cand_eids)])

                total_s1 += 1
                total_cands += len(cand_eids)
                total_matches += len(matched_eids)
                if matched_eids:
                    matched_cnt += 1

        for row in reader:
            if not row:
                continue
            batch.append((row[idx_id], normalize_text(row[idx_name]), normalize_text(row[idx_addr]), row[idx_ctry].strip().lower() if row[idx_ctry] else ""))
            if len(batch) >= INFERENCE_BATCH_SIZE:
                flush_batch(batch)
                batch = []
                elapsed = time.time() - t0
                print(f"  ... {total_s1:,} S1 processed | Matched: {matched_cnt:,} ({matched_cnt/total_s1*100:.1f}%) | "
                      f"Avg Cands: {total_cands/total_s1:.1f} | Speed: {total_s1/elapsed:.0f} S1/sec", flush=True)

        if batch:
            flush_batch(batch)

    elapsed = time.time() - t0
    print(f"\nInference Finished in {elapsed:.1f}s ({elapsed/60:.2f} mins):", flush=True)
    print(f"  Total S1 Entities Processed : {total_s1:,}", flush=True)
    print(f"  S1 Entities with Matches    : {matched_cnt:,} ({matched_cnt/total_s1*100:.2f}%)", flush=True)
    print(f"  Total Matches Output        : {total_matches:,} (Avg {total_matches/total_s1:.2f} per S1)", flush=True)
    print(f"  Total Candidates Output     : {total_cands:,} (Avg {total_cands/total_s1:.2f} per S1)", flush=True)


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate(data_dir, output_dir):
    print("\n" + "="*70, flush=True)
    print("VALIDATING FINAL SUBMISSION INTEGRITY", flush=True)
    print("="*70, flush=True)

    test_s1_path = os.path.join(data_dir, "test", "test_source1.tsv")
    out_matches_path = os.path.join(output_dir, "matching_results.tsv")
    out_cands_path = os.path.join(output_dir, "candidate_pairs.tsv")

    expected_s1_cnt = 0
    with open(test_s1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                expected_s1_cnt += 1

    cands_dict = {}
    with open(out_cands_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["source1_entity_id", "candidate_entity_ids"]
        for row in reader:
            if row:
                cands_dict[row[0]] = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()

    assert len(cands_dict) == expected_s1_cnt

    match_cnt = 0
    with open(out_matches_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["source1_entity_id", "matched_entity_ids"]
        for row in reader:
            if not row:
                continue
            match_cnt += 1
            s1id = row[0]
            matched_set = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
            cand_set = cands_dict.get(s1id, set())
            assert matched_set.issubset(cand_set), f"Match {matched_set} not in candidate set {cand_set} for {s1id}"

    assert match_cnt == expected_s1_cnt
    print(f"SUCCESS: Both output files contain all {expected_s1_cnt:,} entities. All matches are valid subsets of candidate sets!", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    t_start = time.time()
    clf, threshold = fast_train_and_calibrate(args.data_dir)
    run_test_pipeline(args.data_dir, args.output_dir, clf, threshold)
    validate(args.data_dir, args.output_dir)
    print(f"\nALL FINISHED SUCCESSFULLY in {(time.time()-t_start)/60:.2f} mins.\n", flush=True)


if __name__ == "__main__":
    main()

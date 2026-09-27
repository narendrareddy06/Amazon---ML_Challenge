#!/usr/bin/env python3
"""
benchmark_blocking_v2.py -- Measure candidate recall with frequency-ordered multi-key blocking
"""
import csv, os, sys, time, unicodedata, re
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset", "train")
GT_PATH = os.path.join(DATA_DIR, "train_ground_truth.tsv")
S1_PATH = os.path.join(DATA_DIR, "train_source1.tsv")
S2_PATH = os.path.join(DATA_DIR, "train_source2.tsv")
S3_PATH = os.path.join(DATA_DIR, "train_source3.tsv")

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

def make_block_keys(norm_name, norm_addr, country):
    ctry = country.strip().lower()[:10] if country else "unk"
    keys = []
    
    name_toks = get_tokens(norm_name)
    addr_toks = get_tokens(norm_addr)
    addr_nums = get_digits(norm_addr)
    
    # 1. Country + Name tokens (length >= 3)
    for tok in name_toks:
        if len(tok) >= 3:
            keys.append(f"{ctry}|n|{tok}")
            
    # 2. Country + Name prefix (first 4 chars of name)
    if len(norm_name) >= 4:
        keys.append(f"{ctry}|np|{norm_name[:4]}")
        
    # 3. Country + First two name tokens (if multi-word)
    if len(name_toks) >= 2:
        keys.append(f"{ctry}|n2|{name_toks[0]}_{name_toks[1]}")
        
    # 4. Country + Address numbers
    for num in addr_nums:
        if len(num) >= 2:
            keys.append(f"{ctry}|num|{num}")
            
    # 5. Compound: Country + First name token + Address number
    if name_toks and addr_nums:
        keys.append(f"{ctry}|c|{name_toks[0]}_{addr_nums[0]}")
        
    # 6. Country + Longest address token (length >= 5)
    non_num_addr = [t for t in addr_toks if not any(c.isdigit() for c in t) and len(t) >= 5]
    if non_num_addr:
        longest = max(non_num_addr, key=len)
        keys.append(f"{ctry}|a|{longest}")
        
    return list(set(keys))

def load_gt_sample(n_sample=50000):
    gt = {}
    total_pairs = 0
    with open(GT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i >= n_sample:
                break
            s1_id = row[0]
            ids_str = row[1].strip()
            matches = set(ids_str.split(",")) if ids_str else set()
            gt[s1_id] = matches
            total_pairs += len(matches)
    return gt, total_pairs

def run_test():
    gt, total_gt_pairs = load_gt_sample(50000)
    print(f"Loaded ground truth for {len(gt):,} S1 sample ({total_gt_pairs:,} true pairs)")
    
    # Build index from S2 and S3 (first 1M each for quick speed check)
    index = defaultdict(list)
    total_indexed = 0
    t0 = time.time()
    for path in [S2_PATH, S3_PATH]:
        print(f"Indexing {os.path.basename(path)} ...")
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                eid = row[0]
                norm_name = normalize_text(row[1])
                norm_addr = normalize_text(row[2])
                ctry = row[3]
                for k in make_block_keys(norm_name, norm_addr, ctry):
                    index[k].append(eid)
                total_indexed += 1
                if total_indexed % 2000000 == 0:
                    print(f"  ... {total_indexed:,} records indexed ({len(index):,} keys) [{time.time()-t0:.1f}s]")
    
    print(f"Total indexed: {total_indexed:,} in {time.time()-t0:.1f}s. Unique keys: {len(index):,}")
    
    # Evaluate recall on GT sample
    MAX_CANDIDATES = 150
    captured = 0
    total_true = 0
    total_cands_generated = 0
    
    with open(S1_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            s1_id = row[0]
            if s1_id not in gt:
                continue
            true_matches = gt[s1_id]
            if not true_matches:
                continue
                
            norm_name = normalize_text(row[1])
            norm_addr = normalize_text(row[2])
            ctry = row[3]
            keys = make_block_keys(norm_name, norm_addr, ctry)
            
            # Sort keys by specificity (block size ascending)
            # Filter out keys with > 50,000 elements to avoid uninformative mega-blocks
            valid_keys = [(k, len(index.get(k, []))) for k in keys if k in index and len(index[k]) <= 50000]
            valid_keys.sort(key=lambda x: x[1])
            
            cands = set()
            for k, sz in valid_keys:
                for eid in index[k]:
                    cands.add(eid)
                    if len(cands) >= MAX_CANDIDATES:
                        break
                if len(cands) >= MAX_CANDIDATES:
                    break
                    
            total_true += len(true_matches)
            captured += len(true_matches & cands)
            total_cands_generated += len(cands)
            
    recall = (captured / total_true * 100) if total_true else 0
    avg_cands = total_cands_generated / len(gt)
    print("="*60)
    print(f"Candidate Recall on Sample: {recall:.2f}% ({captured:,} / {total_true:,})")
    print(f"Average Candidates per S1 : {avg_cands:.1f} (cap={MAX_CANDIDATES})")
    print("="*60)

if __name__ == "__main__":
    run_test()

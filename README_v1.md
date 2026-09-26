# ML Challenge 2026 — Business Entity Resolution: V1 Baseline

## Quick Start

```bash
# Step 1 — generate matches on the test set (writes output/matching_results.tsv)
python v1_matching.py

# Step 2 — validate the output format
python utils/validate_submission.py

# Optional — tune the similarity threshold on the training validation split
python v1_matching.py --tune

# Optional — run on training data and compute local F0.5
python v1_matching.py --mode train
```

No packages beyond the Python standard library are required (Python >= 3.8).

---

## Approach

### 1. Normalisation

Every business name and address is Unicode-NFC normalised (non-English scripts
preserved), lower-cased, legal-suffix collapsed (Inc, Corp, Limited → inc, corp,
ltd), and address abbreviations expanded/contracted (Street ↔ st, Avenue ↔ ave).

### 2. Blocking (Candidate Generation)

An inverted index maps **blocking keys → list of S2/S3 records**.
Two key families per record:

| Key type | Example |
|---|---|
| `country\|name_token` | `us\|barber` |
| `country\|num\|<street_num>` | `india\|num\|570` |
| `country\|addr\|<lead_addr_tok>` | `france\|addr\|gulmohar` |

For each S1 entity we retrieve all S2/S3 records sharing at least one key.
A hard cap of 500 candidates per S1 keeps memory bounded.

This avoids an all-vs-all comparison across 2M × 10M records.

### 3. Scoring

Each candidate pair receives a composite score ∈ [0, 1]:

```
score = 0.60 × name_score + 0.25 × addr_score + 0.15 × name_token_overlap
```

Where:
- `name_score = 0.6 × Jaccard(name trigrams) + 0.4 × Dice(name tokens)`
- `addr_score = 0.5 × Jaccard(addr trigrams) + 0.5 × Dice(addr tokens)`
- Empty-address pairs get a neutral `addr_score = 0.5` to avoid penalising
  records where one side simply has no address.

### 4. Thresholding

Default threshold: **0.55** (conservative, F0.5 favours precision).

Use `--tune` to automatically sweep thresholds [0.40 … 0.70] on a 20%
held-out validation split of the training data and pick the best F0.5.

### 5. Output

`output/matching_results.tsv` — two columns, tab-separated:
- `source1_entity_id`
- `matched_entity_ids` (comma-separated S2/S3 IDs, empty string if no match)

Every S1 entity appears exactly once (singletons get an empty `matched_entity_ids`).

`output/candidate_pairs.tsv` — same format, listing all blocking candidates
before the threshold is applied.

---

## File Layout

```
student_resource/
├── v1_matching.py          # main script
├── requirements.txt        # dependencies (stdlib only)
├── README.md               # this file
├── dataset/
│   ├── train/              # training TSVs + ground truth
│   └── test/               # test TSVs
├── output/                 # created automatically
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
└── utils/
    └── validate_submission.py
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Stdlib only | No install step, works offline, reproducible |
| Character trigrams | Robust to typos and transliteration variants |
| Name tokens as blocking key | High recall, manageable index size |
| Country prefix on all keys | Prevents cross-country false matches |
| Conservative threshold (0.55) | F0.5 penalises false positives 2× over misses |
| Neutral addr score when empty | Many legitimate records have no address |
| Streaming S1 / index for S2+S3 | RAM stays under 8 GB for the full dataset |

---

## CLI Reference

```
python v1_matching.py [options]

  --mode {test,train}    test (default) or train
  --threshold FLOAT      override similarity threshold
  --tune                 auto-tune threshold on train val split
  --data-dir PATH        root of dataset/ (default: dataset)
  --output-dir PATH      output folder (default: output)
```

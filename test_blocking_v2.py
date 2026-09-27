#!/usr/bin/env python3
"""
test_blocking_v2.py -- Validate multi-strategy frequency-ordered blocking on a sample of train data
"""
import csv, os, sys, time, unicodedata, re
from collections import defaultdict, Counter

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
        
    # 4. Country + Address numbers (street number / pin / zip)
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

print("Testing blocking key generation:")
k1 = make_block_keys("Prime Money LLC", "17560 Ellis Road, Tahlequah, OK", "US")
print("Prime Money:", k1)

import json
import os

cache_path = "data/extracted_triples_cache.json"
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    
    total = len(cache)
    empty_count = sum(1 for triples in cache.values() if not triples)
    
    print(f"Total cache entries: {total}")
    print(f"Empty cache entries (triples = []): {empty_count} ({empty_count/total*100:.1f}%)")
    
    # Print some of the keys that are empty
    empty_keys = [k for k, v in cache.items() if not v]
    print(f"Sample empty keys: {empty_keys[:5]}")
else:
    print("Cache file not found.")

import json
import os

cache_path = "data/extracted_triples_cache.json"
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    
    total = len(cache)
    # Remove empty list entries
    clean_cache = {k: v for k, v in cache.items() if v}
    clean_total = len(clean_cache)
    removed = total - clean_total
    
    print(f"Total entries: {total}")
    print(f"Removed empty entries: {removed}")
    print(f"Remaining entries: {clean_total}")
    
    # Save back
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(clean_cache, f, ensure_ascii=False, indent=2)
    print("Cache cleaned and saved successfully.")
else:
    print("Cache file not found.")

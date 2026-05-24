import json
import os

cache_path = "data/extracted_triples_cache.json"
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Total cache size: {len(cache)} entries")
    
    found_count = 0
    for key, triples in cache.items():
        triples_str = json.dumps(triples).lower()
        if "derrickson" in triples_str:
            print(f"Found in key {key}:")
            print(json.dumps(triples, indent=2))
            found_count += 1
            if found_count > 5:
                break
    if found_count == 0:
        print("No matching triples found for 'derrickson'.")
else:
    print("Cache file not found.")

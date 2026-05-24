import hashlib
import json
import os

p1 = 'David Weissman is a screenwriter and director.  His film credits include "The Family Man" (2000), "Evolution" (2001), and ""When in Rome"" (2010).'
p2 = "The Family Man is a 2000 American romantic comedy-drama film directed by Brett Ratner, written by David Diamond and David Weissman, and starring Nicolas Cage and T\u00e9a Leoni.  Cage's production company, Saturn Films, helped produce the film.  The film centers on a man who sees what could have been had he made a different decision 13 years prior."

cache_path = "data/extracted_triples_cache.json"
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    
    h1 = hashlib.md5(p1.encode("utf-8")).hexdigest()
    h2 = hashlib.md5(p2.encode("utf-8")).hexdigest()
    
    print(f"David Weissman paragraph MD5: {h1}")
    if h1 in cache:
        print("David Weissman Triples:")
        print(json.dumps(cache[h1], indent=2))
    else:
        print("David Weissman not found in cache.")

    print(f"\nThe Family Man paragraph MD5: {h2}")
    if h2 in cache:
        print("The Family Man Triples:")
        print(json.dumps(cache[h2], indent=2))
    else:
        print("The Family Man not found in cache.")
else:
    print("Cache file not found.")

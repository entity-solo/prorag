import hashlib
import json
import os

text = 'Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.  He lives in Los Angeles, California.  He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange."'

text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
print(f"MD5 Hash: {text_hash}")

cache_path = "data/extracted_triples_cache.json"
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if text_hash in cache:
        print("Cached Triples:")
        print(json.dumps(cache[text_hash], indent=2))
    else:
        print("Not found in cache.")
else:
    print("Cache file not found.")

import hashlib
import json
import os

p1 = "End of Days is a 1999 American fantasy action horror thriller film directed by Peter Hyams and starring Arnold Schwarzenegger, Gabriel Byrne, Robin Tunney, Kevin Pollak, Rod Steiger, CCH Pounder, and Udo Kier.  The film follows former New York Police Department detective Jericho Cane (Schwarzenegger) after he saves a banker (Byrne) from an assassin, finds himself embroiled in a religious conflict, and must protect an innocent young woman (Tunney) who is chosen by evil forces to conceive the Antichrist with Satan."
p2 = '"Oh My God" is a song by Guns N\' Roses released in 1999 on the soundtrack to the film "End of Days".  The song was sent out to radio stations in November 1999 as a promo for the soundtrack and the band.  Despite being the band\'s first recorded release in almost five years, it was never issued as a stand-alone single for public retail.'

cache_path = "data/extracted_triples_cache.json"
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    
    h1 = hashlib.md5(p1.encode("utf-8")).hexdigest()
    h2 = hashlib.md5(p2.encode("utf-8")).hexdigest()
    
    print(f"End of Days paragraph MD5: {h1}")
    if h1 in cache:
        print("End of Days Triples:")
        print(json.dumps(cache[h1], indent=2))
    else:
        print("End of Days not found in cache.")

    print(f"\nOh My God paragraph MD5: {h2}")
    if h2 in cache:
        print("Oh My God Triples:")
        print(json.dumps(cache[h2], indent=2))
    else:
        print("Oh My God not found in cache.")
else:
    print("Cache file not found.")

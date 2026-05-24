import json
import os
import hashlib
import sys

def main():
    dataset_path = "data/hotpot_dev_distractor_v1.json"
    cache_path = "data/extracted_triples_cache.json"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return
    if not os.path.exists(cache_path):
        print(f"Error: Cache {cache_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Let's show the graph for the first N questions
    limit = 12
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    print(f"Showing graphs for the first {limit} questions based on cached triples:\n")

    for idx in range(limit):
        item = data[idx]
        question = item["question"]
        gold_answer = item["answer"]
        context_data = item["context"]

        print("=" * 80)
        print(f"QUESTION {idx+1}: {question}")
        print(f"Ground Truth Answer: {gold_answer}")
        print("-" * 80)

        # Collect triples from all paragraphs
        all_triples = []
        for title, sentences in context_data:
            paragraph = " ".join(sentences)
            text_hash = hashlib.md5(paragraph.encode("utf-8")).hexdigest()
            
            triples = cache.get(text_hash, [])
            for t in triples:
                # Add source paragraph title for reference
                t_ref = t.copy()
                t_ref["source"] = title
                all_triples.append(t_ref)

        if not all_triples:
            print("  [No triples found in graph - Graph is EMPTY]")
        else:
            print(f"  Knowledge Graph ({len(all_triples)} facts):")
            # Group by subject for clean visualization
            by_subject = {}
            for t in all_triples:
                subj = t["subject"]
                if subj not in by_subject:
                    by_subject[subj] = []
                by_subject[subj].append(t)

            for subj, facts in sorted(by_subject.items()):
                print(f"  * Entity: {subj.upper()}")
                for t in facts:
                    negstr = "NOT " if t.get("negated") else ""
                    condstr = f" [{t['condition']}]" if t.get("condition") else ""
                    source = t.get("source", "unknown")
                    print(f"    - {t['relation'].upper()} -> {t['object']} {negstr}{condstr} (Source: {source})")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

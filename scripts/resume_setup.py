import sys
import os
import json

def resume_benchmark_run(results_path):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} does not exist.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    details = full_data.get("details", [])
    completed_ids = {item["id"] for item in details if item["prorag"]["answer"] != "ERROR" and item["prorag"]["answer"] != "SKIPPED"}
    print(f"Loaded {len(completed_ids)} completed items from {results_path}.")

    # Load dataset
    dataset_path = "data/hotpot_dev_distractor_v1.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    limit = 25
    eval_set = raw_dataset[:limit]

    # Create a backup file of the input run
    backup_path = results_path + ".bak"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    print(f"Backup created at: {backup_path}")

    # Build the resume script
    # We will modify run_benchmark.py to accept a --resume flag or just do it programmatically
    # But wait, we can also write a clean script `run_benchmark_resumed.py` that inherits or runs the same logic,
    # but skips completed questions and loads existing results from the JSON file!
    # Let's write `run_benchmark_resumed.py` as a wrapper!
    print("Writing run_benchmark_resumed.py...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resume_setup.py <results_json_file>")
    else:
        resume_benchmark_run(sys.argv[1])

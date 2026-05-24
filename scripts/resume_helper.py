import sys
import os
import json

def resume_benchmark(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # Let's see how many completed results we have
    details = full_data.get("details", [])
    completed_ids = {item["id"] for item in details if item["prorag"]["answer"] != "ERROR"}
    print(f"Found {len(completed_ids)} successfully evaluated items in {input_path}")
    print(f"Keys: {completed_ids}")

    # Now let's read the target dataset
    dataset_path = "data/hotpot_dev_distractor_v1.json"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    # Reconstruct details with correct ordering, keeping results for completed items
    # and setting others to run or skip.
    # We want exactly N = 25 questions total
    limit = 25
    eval_set = raw_dataset[:limit]

    # Let's print out what we found
    for idx, item in enumerate(eval_set):
        q_id = item.get("_id", f"q_{idx}")
        is_done = q_id in completed_ids
        print(f"Index {idx+1}: ID={q_id} - {'DONE' if is_done else 'TODO'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resume_helper.py <results_json_file>")
    else:
        resume_benchmark(sys.argv[1], "results/resumed_state.json")

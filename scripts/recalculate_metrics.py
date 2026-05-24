import json
import re
import sys
import os

def normalize_answer(s: str) -> str:
    """Lowercases, removes punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(truth))


def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
        
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
        
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


def clean_answer(ans: str) -> str:
    # Strip <think>...</think> block including the tags
    ans_clean = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL)
    # Strip markdown formatting like **Answer:**
    ans_clean = re.sub(r'(?i)\*\*answer:\*\*', '', ans_clean)
    ans_clean = re.sub(r'(?i)answer:', '', ans_clean)
    # Also strip some conversational prefixes if we want to be more lenient, 
    # but let's stick to simple clean for now.
    return ans_clean.strip()


def main():
    if len(sys.argv) < 2:
        # Find latest benchmark file in results/
        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found.")
            return
        files = [f for f in os.listdir(results_dir) if f.startswith("benchmark_") and f.endswith(".json")]
        if not files:
            print("No benchmark files found.")
            return
        files.sort()
        file_path = os.path.join(results_dir, files[-1])
    else:
        file_path = sys.argv[1]

    print(f"Recalculating metrics for: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("details", [])
    limit = len(details)
    if limit == 0:
        print("No detail entries found in benchmark file.")
        return

    summary = {
        "naive": {"f1": 0.0, "em": 0.0, "latency": 0.0, "calls": 0.0, "tokens": 0.0},
        "prorag": {"f1": 0.0, "em": 0.0, "latency": 0.0, "calls": 0.0, "tokens": 0.0}
    }

    print("\n" + "="*80)
    print(f" RECALCULATING RESULTS (N={limit}) ".center(80, "="))
    print("="*80)

    for idx, r in enumerate(details):
        question = r["question"]
        gold_answer = r["truth"]

        # Recalculate Naive (usually SKIPPED)
        naive_ans = r["naive"]["answer"]
        if naive_ans != "SKIPPED":
            naive_ans_clean = clean_answer(naive_ans)
            r["naive"]["f1"] = compute_f1(naive_ans_clean, gold_answer)
            r["naive"]["em"] = compute_exact_match(naive_ans_clean, gold_answer)
        else:
            r["naive"]["f1"] = 0.0
            r["naive"]["em"] = 0

        # Recalculate ProRAG
        prorag_ans = r["prorag"]["answer"]
        if prorag_ans not in ("ERROR", "SKIPPED"):
            prorag_ans_clean = clean_answer(prorag_ans)
            r["prorag"]["f1"] = compute_f1(prorag_ans_clean, gold_answer)
            r["prorag"]["em"] = compute_exact_match(prorag_ans_clean, gold_answer)
        else:
            r["prorag"]["f1"] = 0.0
            r["prorag"]["em"] = 0
            prorag_ans_clean = prorag_ans

        print(f"\n[{idx+1}/{limit}] Question: {question}")
        print(f"Ground Truth: {gold_answer}")
        print(f"Raw ProRAG Answer: {prorag_ans.replace(chr(10), ' | ')[:120]}...")
        print(f"Cleaned ProRAG Answer: {prorag_ans_clean}")
        print(f"Metrics (Cleaned) -> F1: {r['prorag']['f1']:.4f} | EM: {r['prorag']['em']}")

    # Re-aggregate
    for r in details:
        for mode in ["naive", "prorag"]:
            summary[mode]["f1"] += r[mode]["f1"] / limit
            summary[mode]["em"] += r[mode]["em"] / limit
            summary[mode]["latency"] += r[mode]["latency"] / limit
            summary[mode]["calls"] += r[mode]["calls"] / limit
            summary[mode]["tokens"] += r[mode]["tokens"] / limit

    # Save recalculated version (with suffix _recalculated)
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    out_path = os.path.join(dir_name, f"{name}_recalculated{ext}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f" RECALCULATED BENCHMARK SUMMARY (N={limit})".center(60))
    print("="*60)
    print(f"{'Metric':<25} | {'Naive RAG':<12} | {'ProRAG (Cleaned)':<12}")
    print("-"*60)
    print(f"{'Accuracy (F1)':<25} | {summary['naive']['f1']:.4f}       | {summary['prorag']['f1']:.4f}")
    print(f"{'Exact Match (EM)':<25} | {summary['naive']['em']:.4f}       | {summary['prorag']['em']:.4f}")
    print(f"{'Latency (Avg)':<25} | {summary['naive']['latency']:.2f}s       | {summary['prorag']['latency']:.2f}s")
    print(f"{'LLM Calls / Query':<25} | {summary['naive']['calls']:.1f}         | {summary['prorag']['calls']:.1f}")
    print(f"{'Estimated Tokens / Query':<25} | {summary['naive']['tokens']:.0f}         | {summary['prorag']['tokens']:.0f}")
    print("="*60)
    print(f"Recalculated results saved to: {out_path}\n")

if __name__ == "__main__":
    main()

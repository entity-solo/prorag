import json

with open("data/hotpot_dev_distractor_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

q12 = data[11] # 0-indexed, so 11 is Question 12
print(f"Question: {q12['question']}")
print(f"Answer: {q12['answer']}")
print("\nAll Paragraphs:")
for title, sentences in q12["context"]:
    print(f"--- {title} ---")
    print(" ".join(sentences))

import json

with open("data/hotpot_dev_distractor_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

q1 = data[0]
print(f"Question: {q1['question']}")
print(f"Answer: {q1['answer']}")
print("\nAll Paragraph Titles:")
for title, sentences in q1["context"]:
    print(f"- {title}")

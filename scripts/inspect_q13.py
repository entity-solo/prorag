import json

with open("data/hotpot_dev_distractor_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

q13 = data[12] # Question 13
print(f"Question: {q13['question']}")
print(f"Answer: {q13['answer']}")
print("\nAll Paragraphs:")
for title, sentences in q13["context"]:
    print(f"--- {title} ---")
    text = " ".join(sentences)
    print(text)

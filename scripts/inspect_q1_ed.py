import json

with open("data/hotpot_dev_distractor_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

q1 = data[0]
for title, sentences in q1["context"]:
    if title == "Ed Wood":
        print(f"--- {title} ---")
        print(" ".join(sentences))

import json
import os
import hashlib
import re

def main():
    dataset_path = "data/hotpot_dev_distractor_v1.json"
    cache_path = "data/extracted_triples_cache.json"
    output_path = r"C:\Users\hanng\.gemini\antigravity\brain\c7c02d44-7ee7-4dee-9885-86e8a6eb8250\question_graphs.md"

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

    limit = 25
    md_content = []
    md_content.append("# Đồ thị Thực thể (Knowledge Graphs) của 25 Câu hỏi\n")
    md_content.append("Tài liệu này hiển thị chi tiết đồ thị tri thức (Knowledge Graph) được ProRAG xây dựng cho từng câu hỏi dựa trên các mối quan hệ (triples) đã được trích xuất và lưu trong bộ nhớ đệm (cache).\n")

    for idx in range(limit):
        item = data[idx]
        question = item["question"]
        gold_answer = item["answer"]
        context_data = item["context"]

        md_content.append(f"## Câu hỏi {idx+1}: {question}")
        md_content.append(f"**Đáp án chuẩn (Ground Truth):** `{gold_answer}`\n")
        
        md_content.append("### Văn bản gốc (Original Paragraphs):")
        for title, sentences in context_data:
            paragraph = " ".join(sentences)
            md_content.append(f"- **[{title}]**: {paragraph}")
        md_content.append("")

        # Collect triples
        all_triples = []
        for title, sentences in context_data:
            paragraph = " ".join(sentences)
            text_hash = hashlib.md5(paragraph.encode("utf-8")).hexdigest()
            
            triples = cache.get(text_hash, [])
            for t in triples:
                t_ref = t.copy()
                t_ref["source"] = title
                all_triples.append(t_ref)

        if not all_triples:
            md_content.append("> [!WARNING]\n> Đồ thị rỗng (Không trích xuất được mối quan hệ nào trong cache cũ).\n")
        else:
            md_content.append(f"### Đồ thị Tri thức ({len(all_triples)} quan hệ):")
            md_content.append("| Thực thể gốc (Subject) | Mối quan hệ (Relation) | Thực thể đích (Object) | Bối cảnh (Condition) | Nguồn (Source) |")
            md_content.append("| :--- | :--- | :--- | :--- | :--- |")
            
            # Sort triples for clean presentation
            all_triples.sort(key=lambda x: (x["subject"].lower(), x["relation"].lower()))
            for t in all_triples:
                negstr = "NOT " if t.get("negated") else ""
                condstr = f"`{t['condition']}`" if t.get("condition") else "-"
                md_content.append(f"| **{t['subject']}** | {negstr}{t['relation'].upper()} | {t['object']} | {condstr} | *{t['source']}* |")
            md_content.append("")

            # Generate Mermaid Graph
            question_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
            gold_words = set(re.findall(r"\b\w{3,}\b", gold_answer.lower()))

            def get_relevance(t):
                score = 0
                subj_lower = t["subject"].lower()
                obj_lower = t["object"].lower()
                rel_lower = t["relation"].lower()
                for w in question_words:
                    if w in subj_lower:
                        score += 2
                    if w in obj_lower:
                        score += 2
                    if w in rel_lower:
                        score += 1
                for w in gold_words:
                    if w in subj_lower:
                        score += 3
                    if w in obj_lower:
                        score += 3
                return score

            mermaid_triples = sorted(all_triples, key=lambda x: (-get_relevance(x), x["subject"].lower()))[:15]

            md_content.append("### Sơ đồ Trực quan (Mermaid Graph):")
            md_content.append("```mermaid")
            md_content.append("graph TD")

            added_edges = set()
            node_ids = {}
            node_counter = 0

            def get_node_id(name):
                nonlocal node_counter
                norm = name.lower().strip()
                if norm not in node_ids:
                    node_ids[norm] = f"n{node_counter}"
                    node_counter += 1
                return node_ids[norm]

            for t in mermaid_triples:
                subj = t["subject"].strip()
                obj = t["object"].strip()
                rel = t["relation"].strip().upper()
                if t.get("negated"):
                    rel = "NOT " + rel

                subj_clean = subj.replace('"', '\\"').replace('(', '[').replace(')', ']')
                obj_clean = obj.replace('"', '\\"').replace('(', '[').replace(')', ']')
                rel_clean = rel.replace('"', '\\"').replace('(', '[').replace(')', ']')

                sid = get_node_id(subj)
                oid = get_node_id(obj)

                edge_key = (sid, oid, rel_clean)
                if edge_key not in added_edges:
                    added_edges.add(edge_key)
                    md_content.append(f'  {sid}["{subj_clean}"] -->|"{rel_clean}"| {oid}["{obj_clean}"]')

            md_content.append("```\n")
        
        md_content.append("---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"Markdown file generated successfully at: {output_path}")

if __name__ == "__main__":
    main()

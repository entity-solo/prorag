#!/usr/bin/env python3
"""
ProRAG CLI.
"""

import argparse
import json
import os
import sys


def _load_rag(graph_path: str):
    from prorag import ProRAG

    rag = ProRAG()
    if graph_path and os.path.exists(graph_path):
        rag.load(graph_path)
        print(f"[prorag] Loaded graph from {graph_path}")
    return rag


def cmd_ingest(args):
    rag = _load_rag(args.graph)
    print(f"[prorag] Ingesting {args.file} ...")
    count = rag.ingest_file(args.file, source=args.source or args.file)
    print(f"[prorag] Extracted {count} triples. Stats: {rag.stats()}")
    if args.graph:
        rag.save(args.graph)
        print(f"[prorag] Saved to {args.graph}")


def cmd_ask(args):
    rag = _load_rag(args.graph)
    result = rag.ask(args.question)
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
    print(f"Triples used: {result['triples_used']}")
    if result["has_contradictions"]:
        print("[warning] Contradicting information found in graph.")


def cmd_stats(args):
    rag = _load_rag(args.graph)
    print(json.dumps(rag.stats(), indent=2))


def cmd_interactive(args):
    rag = _load_rag(args.graph)
    print("[prorag] Interactive mode. Type 'quit' to exit, 'stats' for graph info.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "stats":
            print(json.dumps(rag.stats(), indent=2))
            continue
        result = rag.ask(question)
        print(f"\n{result['answer']}\n")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print()


def main():
    parser = argparse.ArgumentParser(prog="prorag", description="ProRAG entity-graph RAG")
    parser.add_argument("--graph", default="graph.json", help="Path to graph file")
    sub = parser.add_subparsers(dest="cmd")

    ingest_parser = sub.add_parser("ingest", help="Ingest a text file into the graph")
    ingest_parser.add_argument("file", help="Path to text file")
    ingest_parser.add_argument("--source", help="Source label for provenance")

    ask_parser = sub.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question string")

    sub.add_parser("stats", help="Show graph statistics")
    sub.add_parser("interactive", help="Interactive Q&A session")

    args = parser.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "interactive":
        cmd_interactive(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

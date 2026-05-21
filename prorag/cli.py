#!/usr/bin/env python3
"""
ProRAG CLI

Usage:
    prorag ingest <file> [--graph graph.json] [--source SOURCE]
    prorag ask <question> [--graph graph.json]
    prorag stats [--graph graph.json]
    prorag interactive [--graph graph.json]
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
    n = rag.ingest_file(args.file, source=args.source or args.file)
    print(f"[prorag] Extracted {n} triples. Stats: {rag.stats()}")
    if args.graph:
        rag.save(args.graph)
        print(f"[prorag] Saved to {args.graph}")


def cmd_ask(args):
    rag = _load_rag(args.graph)
    result = rag.ask(args.question)
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
    print(f"Domains: {', '.join(result['domains'])} | Triples used: {result['triples_used']}")
    if result["has_contradictions"]:
        print("⚠️  Contradicting information found in graph.")


def cmd_stats(args):
    rag = _load_rag(args.graph)
    print(json.dumps(rag.stats(), indent=2))


def cmd_interactive(args):
    rag = _load_rag(args.graph)
    print("[prorag] Interactive mode. Type 'quit' to exit, 'stats' for graph info.\n")
    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower() == "stats":
            print(json.dumps(rag.stats(), indent=2))
            continue
        result = rag.ask(q)
        print(f"\n{result['answer']}\n")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print()


def main():
    parser = argparse.ArgumentParser(prog="prorag", description="ProRAG — Proactive Knowledge Graph RAG")
    parser.add_argument("--graph", default="graph.json", help="Path to graph file (default: graph.json)")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest a text file into the graph")
    p_ingest.add_argument("file", help="Path to text file")
    p_ingest.add_argument("--source", help="Source label for provenance")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", help="Question string")

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

import subprocess
import time
import requests
import sys

def main():
    import os
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable is required to run this test.")
        sys.exit(1)
        
    print("[test] Starting ProRAG daemon server on port 11888...")
    # Clean up any existing test_graph.json
    graph_file = "test_daemon_graph.json"
    if os.path.exists(graph_file):
        os.remove(graph_file)

    # Launch daemon server as a background process using python -m
    # We pass --graph to use a temporary file
    proc = subprocess.Popen(
        [sys.executable, "-m", "prorag.cli", "--graph", graph_file, "serve", "--port", "11888", "--model", "llama-3.1-8b-instant"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Give the server 3 seconds to spin up
        print("[test] Waiting for server to start...")
        time.sleep(3.0)

        # Check if the process died early
        ret = proc.poll()
        if ret is not None:
            stdout, stderr = proc.communicate()
            print(f"[test] Error: Server failed to start with return code {ret}")
            print(f"Stdout:\n{stdout}")
            print(f"Stderr:\n{stderr}")
            sys.exit(1)

        base_url = "http://127.0.0.1:11888"

        # 1. Health check
        print("[test] 1. Checking health endpoint...")
        resp = requests.get(f"{base_url}/")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        print(f"Health response: {data}")
        assert data["status"] == "ok"
        assert data["nodes"] == 0

        # 2. Clear graph (just to test endpoint)
        print("[test] 2. Testing /v1/clear endpoint...")
        resp = requests.post(f"{base_url}/v1/clear")
        assert resp.status_code == 200
        print(f"Clear response: {resp.json()}")

        # 3. Ingest text
        print("[test] 3. Testing /v1/ingest/text endpoint...")
        payload = {
            "text": "The Space Race was a 20th-century competition between two Cold War adversaries, the Soviet Union and the United States, to achieve superior spaceflight capability.",
            "source": "Wikipedia",
            "domains": ["History", "Space"]
        }
        resp = requests.post(f"{base_url}/v1/ingest/text", json=payload)
        if resp.status_code != 200:
            print(f"[test] Ingest failed: status={resp.status_code}, response={resp.text}")
        assert resp.status_code == 200
        ingest_data = resp.json()
        print(f"Ingest response: {ingest_data}")
        assert ingest_data["status"] == "success"
        assert ingest_data["triples_added"] > 0

        # 4. Get stats
        print("[test] 4. Testing /v1/stats endpoint...")
        resp = requests.get(f"{base_url}/v1/stats")
        assert resp.status_code == 200
        stats_data = resp.json()
        print(f"Stats response: {stats_data}")
        assert stats_data["nodes"] > 0
        domains_lower = [d.lower() for d in stats_data["domains"]]
        assert "history" in domains_lower or "space" in domains_lower

        # 5. Query graph (ask)
        print("[test] 5. Testing /v1/ask endpoint...")
        query_payload = {
            "question": "Which two adversaries competed in the Space Race?"
        }
        resp = requests.post(f"{base_url}/v1/ask", json=query_payload)
        assert resp.status_code == 200
        ask_data = resp.json()
        print(f"Answer: {ask_data['answer']}")
        print(f"Sources used: {ask_data['sources']}")
        assert len(ask_data["sources"]) > 0
        assert "Soviet Union" in ask_data["answer"] or "United States" in ask_data["answer"]

        print("\n[test] All API endpoints verified successfully!")

    finally:
        print("[test] Terminating daemon server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        # Clean up the test graph file
        if os.path.exists(graph_file):
            os.remove(graph_file)

if __name__ == "__main__":
    main()

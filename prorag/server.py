import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prorag import ProRAG

app = FastAPI(
    title="ProRAG Daemon (Open-Source Edition)",
    description="FastAPI server for local-first single-user entity-graph RAG",
    version="0.2.0"
)

# Enable CORS for local app calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
MODEL_NAME = os.environ.get("PRORAG_MODEL_NAME", "llama-3.3-70b-versatile")
GRAPH_PATH = os.environ.get("PRORAG_GRAPH_PATH", "graph.json")

# Initialize single global ProRAG instance
rag_instance = ProRAG(model=MODEL_NAME)
if os.path.exists(GRAPH_PATH):
    try:
        rag_instance.load(GRAPH_PATH)
        print(f"[prorag-server] Loaded local graph from {GRAPH_PATH}")
    except Exception as e:
        print(f"[prorag-server] Failed to load local graph from {GRAPH_PATH}: {e}")
else:
    print(f"[prorag-server] Initialized new local graph. Path: {GRAPH_PATH}")


def save_graph() -> None:
    """Save the global ProRAG graph instance to disk."""
    try:
        rag_instance.save(GRAPH_PATH)
        print(f"[prorag-server] Saved graph changes to {GRAPH_PATH}")
    except Exception as e:
        print(f"[prorag-server] Failed to auto-save graph to {GRAPH_PATH}: {e}")


# Pydantic Schemas
class IngestTextRequest(BaseModel):
    text: str
    source: Optional[str] = None


class AskRequest(BaseModel):
    question: str


@app.get("/")
def read_root():
    """Health check showing system status."""
    stats = rag_instance.stats()
    return {
        "status": "ok",
        "app": "ProRAG Daemon (Open-Source Local-First Edition)",
        "model": MODEL_NAME,
        "graph_file": GRAPH_PATH,
        "nodes": stats.get("nodes", 0),
        "edges": stats.get("edges", 0)
    }


@app.post("/v1/ingest/text")
def ingest_text_endpoint(req: IngestTextRequest):
    """Ingest raw text directly into the local graph."""
    try:
        n = rag_instance.ingest(req.text, source=req.source or "")
        save_graph()
        return {"status": "success", "triples_added": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest/file")
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    source: Optional[str] = Form(None)
):
    """Upload and ingest a text file into the local graph."""
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        file_source = source or file.filename or "uploaded_file"
        n = rag_instance.ingest(text, source=file_source)
        save_graph()
        return {"status": "success", "triples_added": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ask")
def ask_endpoint(req: AskRequest):
    """Query the local knowledge graph with a question."""
    try:
        res = rag_instance.ask(req.question)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stats")
def stats_endpoint():
    """Retrieve statistics of the local graph."""
    return rag_instance.stats()


@app.post("/v1/clear")
def clear_endpoint():
    """Clear all data in the local graph."""
    from prorag.graph import ProRAGGraph
    rag_instance.graph = ProRAGGraph()
    save_graph()
    return {"status": "success", "message": "Local graph cleared and saved"}

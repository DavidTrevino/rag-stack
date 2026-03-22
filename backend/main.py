from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, OptimizersConfigDiff
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import os

app = FastAPI()

# Init services
qdrant = QdrantClient(host="qdrant", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

neo4j_driver = GraphDatabase.driver(
    "bolt://neo4j:7687", auth=("neo4j", "password")
)

OLLAMA_URL = "http://ollama:11434/api/generate"
COLLECTION = "docs"

# --- Ensure collection exists (RUN ON STARTUP) ---
@app.on_event("startup")
def setup_collection():
    collections = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0)
        )

# --- Helpers ---
def embed(text):
    return model.encode(text).tolist()

def ollama_generate(prompt):
    r = requests.post(OLLAMA_URL, json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    return r.json()["response"]

# --- Ingest local docs ---
@app.post("/ingest/local")
def ingest_local():
    docs = []

    for root, _, files in os.walk("/mnt/docs"):
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "r", errors="ignore") as file:
                    text = file.read()
                    docs.append(text[:2000])
            except:
                pass

    points = []
    for i, doc in enumerate(docs):
        vector = embed(doc)

        points.append({
            "id": i,
            "vector": vector,
            "payload": {"text": doc}
        })

    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)

    return {"status": "local docs ingested", "count": len(points)}

# --- Ingest URL ---
@app.post("/ingest/url")
def ingest_url(url: str):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

    vector = embed(text[:2000])

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[{
            "id": 9999,
            "vector": vector,
            "payload": {"text": text[:2000]}
        }]
    )

    return {"status": "url ingested"}

# --- Query RAG ---
@app.get("/query")
def query(q: str):
    query_vector = embed(q)

    hits = qdrant.search_points(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=5
    )

    context = "\n".join([h.payload["text"] for h in hits])

    prompt = f"""
Context:
{context}

Question:
{q}
"""

    answer = ollama_generate(prompt)

    return {"answer": answer, "context": context}

# --- Graph extraction ---
@app.post("/graph/extract")
def extract_graph():
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        session.run("""
        MERGE (a:Topic {name:"AI"})
        MERGE (b:Topic {name:"Health"})
        MERGE (a)-[:RELATES_TO]->(b)
        """)

    return {"status": "graph created"}

# --- Graph query ---
@app.get("/graph")
def get_graph():
    with neo4j_driver.session() as session:
        result = session.run("""
        MATCH (a)-[r]->(b)
        RETURN a.name as source, b.name as target
        """)

        edges = [{"source": r["source"], "target": r["target"]} for r in result]

    return {"edges": edges}

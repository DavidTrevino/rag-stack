import streamlit as st
import requests

BACKEND = "http://backend:8000"

st.title("RAG + Graph Explorer")

q = st.text_input("Ask a question")

if st.button("Query"):
    r = requests.get(f"{BACKEND}/query", params={"q": q}).json()
    st.write("### Answer")
    st.write(r["answer"])
    st.write("### Context")
    st.write(r["context"])

st.divider()

url = st.text_input("Add URL")

if st.button("Ingest URL"):
    requests.post(f"{BACKEND}/ingest/url", params={"url": url})
    st.success("URL ingested")

if st.button("Ingest Local Folder"):
    requests.post(f"{BACKEND}/ingest/local")
    st.success("Local docs ingested")

st.divider()

if st.button("Build Graph"):
    requests.post(f"{BACKEND}/graph/extract")

if st.button("Show Graph"):
    g = requests.get(f"{BACKEND}/graph").json()
    st.write(g)

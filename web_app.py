#!/usr/bin/env python3
"""Web UI for the ASV Bible RAG Agent."""

import atexit
import json
import os
import re
import signal
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen

from flask import Flask, jsonify, render_template, request, Response, session, stream_with_context
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM

FAISS_INDEX_DIR = "faiss_index"
ASV_FILE = "asv.txt"
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
KNOWLEDGE_FILE = "knowledge.json"

app = Flask(__name__)
app.secret_key = os.urandom(24)

LEARNED_TOPICS_FILE = "learned_topics.json"
QA_CACHE_FILE = "qa_cache.json"
MAX_HISTORY = 6

verse_db: dict[str, str] = {}
chapter_db: dict[str, list[dict]] = {}
learned_topics: dict[str, list[str]] = {}
qa_cache: dict[str, str] = {}


def load_learned_topics():
    if Path(LEARNED_TOPICS_FILE).exists():
        with open(LEARNED_TOPICS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_learned_topics():
    with open(LEARNED_TOPICS_FILE, "w") as f:
        json.dump(learned_topics, f, indent=2)


def load_qa_cache():
    if Path(QA_CACHE_FILE).exists():
        with open(QA_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_qa_cache():
    with open(QA_CACHE_FILE, "w") as f:
        json.dump(qa_cache, f, indent=2)


def load_verse_db(filepath: str):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            ref, text = line.split("\t", 1)
            parts = ref.rsplit(":", 1)
            if len(parts) != 2:
                continue
            verse_db[ref] = text
            chapter_key = parts[0]
            verse_num = parts[1]
            chapter_db.setdefault(chapter_key, []).append({
                "ref": ref,
                "verse": verse_num,
                "text": text,
            })


def load_verses(filepath: str) -> list[Document]:
    chapters: dict[str, list[str]] = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            ref, text = line.split("\t", 1)
            parts = ref.rsplit(":", 1)
            if len(parts) != 2:
                continue
            chapter_key = parts[0]
            chapters.setdefault(chapter_key, []).append(f"{ref}  {text}")
    docs = []
    for chapter_key, verses in chapters.items():
        content = "\n".join(verses)
        docs.append(Document(page_content=content, metadata={"chapter": chapter_key}))
    return docs


def build_index(docs: list[Document], embeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n"],
    )
    chunks = splitter.split_documents(docs)
    total = len(chunks)
    print(f"  Embedding {total} chunks (one-time operation)...")

    batch_size = 50
    vectorstore = None
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        done = min(i + batch_size, total)
        pct = done * 100 // total
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {done}/{total} ({pct}%)", end="", flush=True)

    print()
    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore


def get_vectorstore(embeddings) -> FAISS:
    if Path(FAISS_INDEX_DIR).exists():
        return FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    docs = load_verses(ASV_FILE)
    return build_index(docs, embeddings)


ollama_process = None
ollama_started_by_us = False


def stop_ollama():
    if ollama_started_by_us and ollama_process and ollama_process.poll() is None:
        print("\nStopping Ollama...")
        ollama_process.terminate()
        try:
            ollama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ollama_process.kill()
        print("Ollama stopped.")


def ensure_ollama():
    global ollama_process, ollama_started_by_us
    try:
        urlopen("http://localhost:11434/api/tags", timeout=2)
        print("Ollama already running (external).")
        return
    except Exception:
        pass

    print("Starting Ollama (4 threads)...")
    env = os.environ.copy()
    env["OLLAMA_NUM_THREADS"] = "4"
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    ollama_started_by_us = True
    atexit.register(stop_ollama)

    for _ in range(15):
        time.sleep(1)
        try:
            urlopen("http://localhost:11434/api/tags", timeout=2)
            print("Ollama started (will stop when app exits).")
            return
        except Exception:
            pass
    print("WARNING: Ollama may not have started. Check /tmp/ollama.log")


knowledge: dict = {}


def load_knowledge():
    if Path(KNOWLEDGE_FILE).exists():
        with open(KNOWLEDGE_FILE, "r") as f:
            return json.load(f)
    return {}


def resolve_chapters(question: str) -> list[str]:
    """Resolve a question to relevant chapters using knowledge config + learned topics."""
    q_lower = question.lower()
    chapters = []

    synonyms = knowledge.get("synonyms", {})
    for alias, canonical in synonyms.items():
        if alias.lower() in q_lower:
            q_lower = q_lower + " " + canonical.lower()

    topics = knowledge.get("topics", {})
    for topic, chs in topics.items():
        if topic in q_lower:
            chapters.extend(chs)

    for topic, chs in learned_topics.items():
        if topic in q_lower:
            chapters.extend(chs)

    book_groups = knowledge.get("book_groups", {})
    for group_name, books in book_groups.items():
        if group_name in q_lower:
            for book in books:
                book_chapters = [k for k in chapter_db if k.startswith(book + " ")]
                chapters.extend(book_chapters[:3])

    return chapters


def keyword_search(question: str) -> list[Document]:
    hint_chapters = resolve_chapters(question)

    if not hint_chapters:
        return []

    docs = []
    seen = set()
    for ch in hint_chapters:
        if ch in seen:
            continue
        seen.add(ch)
        verses = chapter_db.get(ch)
        if not verses:
            continue
        content = "\n".join(f"{v['ref']}  {v['text']}" for v in verses)
        docs.append(Document(page_content=content, metadata={"chapter": ch}))

    return docs


def hybrid_search(question: str) -> list[Document]:
    vector_docs = retriever.invoke(question)
    keyword_docs = keyword_search(question)

    seen_chapters = set()
    merged = []
    for doc in keyword_docs + vector_docs:
        ch = doc.metadata.get("chapter", "")
        if ch not in seen_chapters:
            seen_chapters.add(ch)
            merged.append(doc)

    return merged[:8]


ensure_ollama()

print("Loading knowledge config...")
knowledge.update(load_knowledge())
topics_count = len(knowledge.get("topics", {}))
groups_count = len(knowledge.get("book_groups", {}))
synonyms_count = len(knowledge.get("synonyms", {}))
print(f"  {topics_count} topics, {groups_count} book groups, {synonyms_count} synonyms")

print("Loading learned data...")
learned_topics.update(load_learned_topics())
qa_cache.update(load_qa_cache())
print(f"  {len(learned_topics)} learned topics, {len(qa_cache)} cached answers")

print("Loading verse database...")
load_verse_db(ASV_FILE)
print(f"  {len(verse_db)} verses loaded")

print("Loading models...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = get_vectorstore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
llm = OllamaLLM(model=MODEL_NAME, temperature=0.1)
print("Ready!")

SYSTEM_PROMPT = """Answer using ONLY these ASV scripture passages. Cite book, chapter, and verse. Be concise.

{context}
{history}
Question: {question}

Answer:"""

chat_histories: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    return chat_histories.get(session_id, [])


def add_to_history(session_id: str, role: str, text: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    chat_histories[session_id].append({"role": role, "text": text})
    if len(chat_histories[session_id]) > MAX_HISTORY * 2:
        chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY * 2:]


def format_history(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return ""
    lines = ["\nRecent conversation:"]
    for entry in history[-MAX_HISTORY * 2:]:
        prefix = "User" if entry["role"] == "user" else "Assistant"
        text = entry["text"][:200]
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines) + "\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default")
    if not question:
        return json.dumps({"error": "Empty question"}), 400

    add_to_history(session_id, "user", question)

    cache_key = question.lower().strip()
    if cache_key in qa_cache:
        def cached_generate():
            yield f"data: {json.dumps({'log': '> Cache hit!'})}\n\n"
            yield f"data: {json.dumps({'token': qa_cache[cache_key]})}\n\n"
            yield "data: [DONE]\n\n"
        add_to_history(session_id, "assistant", qa_cache[cache_key])
        return Response(
            stream_with_context(cached_generate()),
            mimetype="text/event-stream",
        )

    def generate():
        yield f"data: {json.dumps({'log': '> Searching index...'})}\n\n"
        docs = hybrid_search(question)
        refs = []
        for doc in docs:
            ch = doc.metadata.get("chapter", "?")
            if ch not in refs:
                refs.append(ch)
        ref_str = ", ".join(refs)
        yield f"data: {json.dumps({'log': '> Retrieved ' + str(len(docs)) + ' chunks: ' + ref_str})}\n\n"

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        history = format_history(session_id)
        prompt = SYSTEM_PROMPT.format(context=context, history=history, question=question)
        ctx_len = len(prompt)
        has_history = " (with history)" if history else ""
        yield f"data: {json.dumps({'log': '> Prompt: ' + str(ctx_len) + ' chars -> ' + MODEL_NAME + has_history})}\n\n"
        yield f"data: {json.dumps({'log': '> Generating response...'})}\n\n"

        token_count = 0
        full_answer = ""
        for chunk in llm.stream(prompt):
            token_count += 1
            full_answer += chunk
            yield f"data: {json.dumps({'token': chunk})}\n\n"

        add_to_history(session_id, "assistant", full_answer)

        yield f"data: {json.dumps({'log': '> Done (' + str(token_count) + ' tokens generated)'})}\n\n"
        yield f"data: {json.dumps({'chapters': refs})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    chapters = data.get("chapters", [])
    vote = data.get("vote", "")

    if vote == "up" and question and chapters:
        q_lower = question.lower().strip()
        if q_lower not in learned_topics:
            learned_topics[q_lower] = chapters
            save_learned_topics()

        if q_lower not in qa_cache and answer:
            qa_cache[q_lower] = answer
            save_qa_cache()

        return jsonify({"status": "saved", "learned_topic": q_lower, "chapters": chapters})

    return jsonify({"status": "ok"})


@app.route("/status", methods=["GET"])
def status():
    try:
        resp = urlopen("http://localhost:11434/api/ps", timeout=2)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return jsonify({"ollama": "running", "models_loaded": models})
    except Exception:
        return jsonify({"ollama": "disconnected", "models_loaded": []})


@app.route("/verse", methods=["GET"])
def get_verse():
    ref = request.args.get("ref", "").strip()
    if not ref:
        return jsonify({"error": "Missing ref"}), 400

    parts = ref.rsplit(":", 1)
    if len(parts) != 2:
        return jsonify({"error": "Invalid reference"}), 400

    chapter_key = parts[0]
    verses = chapter_db.get(chapter_key)
    if not verses:
        return jsonify({"error": f"Chapter not found: {chapter_key}"}), 404

    target_verse = parts[1]
    target_idx = None
    for i, v in enumerate(verses):
        if v["verse"] == target_verse:
            target_idx = i
            break

    if target_idx is None:
        return jsonify({"error": f"Verse not found: {ref}"}), 404

    return jsonify({
        "chapter": chapter_key,
        "target_ref": ref,
        "target_verse": target_verse,
        "verses": verses,
    })


if __name__ == "__main__":
    def handle_signal(sig, frame):
        stop_ollama()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    threading.Timer(1.5, webbrowser.open, args=["http://localhost:7777"]).start()
    try:
        app.run(host="127.0.0.1", port=7777, debug=False)
    finally:
        stop_ollama()

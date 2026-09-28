#!/usr/bin/env python3
"""Local RAG agent for querying the ASV Bible text file."""

import os
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, OllamaLLM

FAISS_INDEX_DIR = "faiss_index"
ASV_FILE = "asv.txt"
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"


def load_verses(filepath: str) -> list[Document]:
    """Load the ASV text file, grouping verses into chapter-level chunks."""
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


def build_index(docs: list[Document], embeddings: Embeddings) -> FAISS:
    """Split documents and build a FAISS vector index with progress."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n"],
    )
    chunks = splitter.split_documents(docs)
    total = len(chunks)
    print(f"  Created {total} chunks from {len(docs)} chapters")
    print(f"  Embedding chunks (this is a one-time operation)...")

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
    """Load existing FAISS index or build a new one."""
    if Path(FAISS_INDEX_DIR).exists():
        print("Loading existing index...")
        return FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    print("Building index from asv.txt (first run, may take a minute)...")
    docs = load_verses(ASV_FILE)
    print(f"  Loaded {len(docs)} chapters")
    return build_index(docs, embeddings)


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def main():
    print("=" * 60)
    print("  ASV Bible RAG Agent (100% Local)")
    print("=" * 60)
    print()

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = get_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    print(f"Connecting to Ollama ({MODEL_NAME})...")
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.1)

    prompt = ChatPromptTemplate.from_template(
        """You are a knowledgeable Bible study assistant. Answer the question
using ONLY the provided scripture passages from the American Standard Version (ASV).
Always cite the specific book, chapter, and verse references in your answer.
If the passages don't contain enough information to answer, say so honestly.

Scripture passages:
{context}

Question: {question}

Answer:"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print()
    print("Ready! Ask any question about the Bible. Type 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nSearching scriptures...\n")
        try:
            answer = chain.invoke(question)
            print(f"Agent: {answer}")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()

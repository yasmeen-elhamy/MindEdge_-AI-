"""
rag.py — Production-safe RAG Module
"""

import os
import glob
import chromadb

from huggingface_hub import snapshot_download, InferenceClient
from sentence_transformers import SentenceTransformer

from config import OUTPUT_DIR, RAG_DIR, CHAT_LOG_DIR, MODEL_DIR
from llm import generate_response
from config import export_to_folder



def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text[:5000]



print("\n Loading embedding model...")

model_path = os.path.join(MODEL_DIR, "config.json")

if not os.path.exists(model_path):
    print("Downloading embedding model...")
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=MODEL_DIR
    )

_EMBEDDING_MODEL = SentenceTransformer(MODEL_DIR)

print(" Embedding model ready")



def build_or_load_index(folder: str = OUTPUT_DIR, persist_dir: str = RAG_DIR):

    print(" Loading RAG index...")

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("edu_docs")

   
    if collection.count() > 0:
        print(f" Using cached index ({collection.count()} docs)")
        return collection

    docs = []

    for filepath in glob.glob(os.path.join(folder, "*.md")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = clean_text(f.read())
                docs.append({"id": filepath, "text": text})
        except:
            continue

    if not docs:
        print(" No documents found")
        return collection

    print(f" Encoding {len(docs)} docs...")

    embeddings = [
        _EMBEDDING_MODEL.encode(d["text"]).tolist()
        for d in docs
    ]

    collection.upsert(
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["id"]} for d in docs],
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
    )

    print(f" Indexed {len(docs)} documents")

    return collection



def retrieve_passages(query: str, collection, top_k: int = 5):

    if not collection or collection.count() == 0:
        return []

    try:
        q_emb = _EMBEDDING_MODEL.encode(query).tolist()

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, collection.count())
        )

        flat = []
        for group in results.get("documents", []):
            flat.extend(group if isinstance(group, list) else [group])

        return flat

    except Exception as e:
        print(f" Retrieval error: {e}")
        return []



def get_study_context(collection, text: str, top_k: int = 5):

    if not collection or collection.count() == 0:
        return text[:2000]

    try:
        passages = retrieve_passages(text, collection, top_k)

        if not passages:
            return text[:2000]

        return "\n\n".join(passages)

    except:
        return text[:2000]

from openai import OpenAI
import re

def auto_scan_text(text_content: str, api_key: str, model_id: str):

    print(" Scanning definitions & rules...")

    system_prompt = """
You are a strict extraction engine.

ONLY output lines in EXACT format:

[DEFINITION: Name] explanation
[RULE: Name] formula

NO extra text.
"""

    user_prompt = f"""
Extract definitions and rules.

TEXT:
{text_content[:6000]}
"""

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.2
        )

        result = response.choices[0].message.content

        if not result:
            print(" Empty LLM response")
            return

        print(" LLM output sample:\n", result[:300])

        extracted_count = 0

        pattern = r"\[(DEFINITION|RULE):\s*(.*?)\]\s*(.*)"

        for line in result.split("\n"):
            line = line.strip()

            match = re.match(pattern, line, re.IGNORECASE)

            if not match:
                continue

            type_, name, body = match.groups()

            name = name.strip() or "Unknown"
            body = body.strip()

            if not body:
                continue

            if type_.upper() == "DEFINITION":
                export_to_folder("definitions", name, body)
                print(f" Definition saved: {name}")
                extracted_count += 1

            elif type_.upper() == "RULE":
                export_to_folder("rules", name, body)
                print(f" Rule saved: {name}")
                extracted_count += 1

        if extracted_count < 3:
            print(" Weak extraction → running fallback")

            sentences = re.split(r'[.\n]', text_content)

            for s in sentences:
                s = s.strip()

                if len(s) < 40:
                    continue

                lower = s.lower()

                if any(x in lower for x in [" is ", " refers to ", " defined as "]):
                    export_to_folder("definitions", "Auto", s)

                if any(x in s for x in ["=", "→", "∝", "|"]):
                    export_to_folder("rules", "Auto", s)

        print(f" Total extracted: {extracted_count}")

    except Exception as e:
        print(f" Auto scan failed: {e}")


def save_chat_log(question: str, answer: str, log_filename="persistent_memory.md"):

    path = os.path.join(CHAT_LOG_DIR, log_filename)

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\nA: {answer}\n---\n")


def load_previous_summary(log_filename="persistent_memory.md"):

    path = os.path.join(CHAT_LOG_DIR, log_filename)

    if not os.path.exists(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()[-3000:]

        return generate_response(
            "Summarize briefly:\n\n" + content,
            max_tokens=200
        )

    except:
        return ""


def summarize_chat_history(chat_history: list):

    if not chat_history:
        return ""

    text = "\n".join(
        [f"User: {e['question']}\nAI: {e['answer']}" for e in chat_history]
    )

    return generate_response(
        "Summarize:\n\n" + text,
        max_tokens=300
    )

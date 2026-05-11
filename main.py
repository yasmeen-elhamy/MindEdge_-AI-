"""
EduScan – Intelligent Study Scanner & Tutor
"""

import os
import sys
import glob
import zipfile
import platform
import subprocess
import nltk
from datetime import datetime
from pathlib import Path

from config import (
    setup_output_dirs,
    load_token,
    pick_files,
    PROJECT_ROOT,
    OUTPUT_DIR,
    CHAT_LOG_DIR
)

from llm import summarize_text, test_connection
from ocr import setup_tesseract, run_ocr_pipeline
from rag import build_or_load_index, auto_scan_text
from chat import run_chat_loop
from image import extract_and_analyze_graphs
from study_plan import generate_study_plan
from topic_extractor import extract_topics_from_text   

setup_output_dirs()
HF_TOKEN = load_token()
FILE_PATHS = pick_files()

if not FILE_PATHS:
    print(" No files selected.")
    sys.exit(1)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

print(f"\n Configuration complete.")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"FILES      : {len(FILE_PATHS)}")

if not test_connection():
    print(" LLM connection failed.")

setup_tesseract()


all_corrected_texts, last_raw_text = run_ocr_pipeline(FILE_PATHS)

if not all_corrected_texts:
    print(" No text extracted.")
    sys.exit(1)


print("\n GRAPH ANALYSIS")

graph_results = []

for path in FILE_PATHS:
    try:
        graph_results.extend(
            extract_and_analyze_graphs(path, hf_token=HF_TOKEN)
        )
    except Exception:
        continue

graphs_md_path = os.path.join(OUTPUT_DIR, "graphs.md")

with open(graphs_md_path, "w", encoding="utf-8") as f:
    f.write("# Graph Analysis\n\n")
    for r in graph_results:
        f.write(f"## {os.path.basename(r['image_path'])}\n\n")
        f.write(f"{r['analysis']}\n\n---\n\n")

print(f" {len(graph_results)} graphs analyzed")

print("\n SUMMARIZATION")

detected_topics = []

for path, corrected in all_corrected_texts.items():
    print(f"→ {os.path.basename(path)}")

    summary = summarize_text(corrected)
    if summary.startswith("Error"):
        summary = "No summary"

    stem = Path(path).stem.replace(" ", "_")

    with open(os.path.join(OUTPUT_DIR, f"{stem}_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)

    try:
        topics = extract_topics_from_text(corrected[:3000])
        detected_topics.extend(topics)
    except:
        detected_topics.append(stem)

detected_topics = list(set(detected_topics))

print(f" Summaries done")


print("\n BUILDING RAG")

collection = build_or_load_index(folder=OUTPUT_DIR)

print(f" {collection.count()} docs indexed")


print("\n STUDY PLAN")

try:
    sp_days = int(input("Days [7]: ") or 7)
    sp_hours = int(input("Hours/day [2]: ") or 2)
    sp_level = input("Level (Beginner/Intermediate/Advanced) [Intermediate]: ") or "Intermediate"

    if sp_level not in ["Beginner", "Intermediate", "Advanced"]:
        sp_level = "Intermediate"

    sp_subject = input(f"Subject [{detected_topics[0] if detected_topics else 'Study'}]: ") \
        or (detected_topics[0] if detected_topics else "Study")

except:
    sp_days, sp_hours, sp_level = 7, 2, "Intermediate"
    sp_subject = detected_topics[0] if detected_topics else "Study"

study_plan_dict = generate_study_plan(
    topics=detected_topics,
    days=sp_days,
    hours_per_day=sp_hours,
    subject=sp_subject,
    level=sp_level,
    collection=collection  
)

sp_path = os.path.join(OUTPUT_DIR, "study_plan.md")

with open(sp_path, "w", encoding="utf-8") as f:
    f.write(f"# Study Plan — {sp_subject}\n\n")
    for d, content in study_plan_dict.items():
        f.write(f"## {d}\n\n{content}\n\n---\n\n")

print(" Study plan generated")


if last_raw_text:
    from llm import _QW_MODEL_ID
    auto_scan_text(last_raw_text, HF_TOKEN, _QW_MODEL_ID)


chat_history = run_chat_loop(collection)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if chat_history:
    path = os.path.join(OUTPUT_DIR, f"chat_{timestamp}.md")

    with open(path, "w", encoding="utf-8") as f:
        for e in chat_history:
            f.write(f"Q: {e['question']}\nA: {e['answer']}\n---\n")

    print(f" Chat saved")


print("\n EXPORT")

export_path = str(PROJECT_ROOT / f"EduScan_{timestamp}.zip")

with zipfile.ZipFile(export_path, "w") as z:
    for folder in [OUTPUT_DIR, CHAT_LOG_DIR]:
        for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
            if os.path.isfile(f):
                z.write(f, os.path.relpath(f, folder))

print(f" Exported → {export_path}")

print("\n DONE")

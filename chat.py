"""
chat.py — Clean & Safe Interactive Chat Loop (TTS + Study Plan)
"""

import os
import re

from config import CHAT_LOG_DIR, export_to_folder
from llm import dev_ask_llm
from study_plan import generate_study_plan
from topic_extractor import extract_topics_from_text
from rag import (
    retrieve_passages,
    save_chat_log,
    load_previous_summary,
    summarize_chat_history,
)
from tts_utils import safe_generate_tts


def run_chat_loop(collection) -> list:
    log_filename = "persistent_memory.md"
    chat_history = []
    all_exchanges = []

    try:
        print("\n🎧 Voice mode:")
        print("1) Always ON")
        print("2) Ask each time")
        print("3) OFF")
        mode = input("Choose: ").strip()

        use_tts_global = mode == "1"
        ask_each_time = mode == "2"

    except Exception:
        use_tts_global = False
        ask_each_time = False

    current_summary = load_previous_summary(log_filename)

    print("\n" + "═" * 60)
    print("  AI CHAT (RAG + STUDY PLAN + TTS)")
    print("Type 'exit' to quit")
    print("═" * 60)

    while True:
        try:
            query = input("\n Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Session Ended]")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            break

        
        if "study plan" in query.lower():

            print(" Generating study plan...")

            passages = retrieve_passages(query, collection, top_k=5)
            context = "\n".join(passages)

            topics = extract_topics_from_text(context[:3000])

            if not topics:
                print(" Could not extract topics")
                continue

            try:
                plan = generate_study_plan(
                    topics=topics,
                    days=7,
                    hours_per_day=2,
                    level="Intermediate"
                )
            except Exception as e:
                print(f" Study plan failed: {e}")
                continue

            print("\n Study Plan:\n")

            plan_text = ""
            for day, content in plan.items():
                print(day)
                print(content)
                print("-" * 40)

                plan_text += f"{day}\n{content}\n\n"

            if use_tts_global or (ask_each_time and input(" Play audio? (y/n): ") == "y"):
                clean = plan_text.replace("\n", " ").strip()
                filename = safe_generate_tts(clean[:1000])
                if filename:
                    print(f" audio_cache/{filename}")

            continue

        
        passages = retrieve_passages(query, collection, top_k=5)
        context = "\n\n".join(passages) if passages else "No context found."

        # memory
        memory_path = os.path.join(CHAT_LOG_DIR, log_filename)
        if os.path.exists(memory_path):
            with open(memory_path, "r", encoding="utf-8") as f:
                current_summary = f.read()[-3000:]

        system_prompt = "You are a helpful AI study assistant."

        user_prompt = (
            f"[MEMORY]:\n{current_summary}\n\n"
            f"[CONTEXT]:\n{context}\n\n"
            f"[QUESTION]: {query}\n\n"
            "Answer clearly. Use [RULE: Name] for formulas."
        )

        answer = dev_ask_llm(system_prompt, user_prompt)

        print("\n" + "─" * 60)
        print(answer)
        print("─" * 60)

      
        use_tts = use_tts_global

        if ask_each_time:
            use_tts = input("🔊 Play audio? (y/n): ").strip().lower() == "y"

        if use_tts and not answer.startswith("Error"):
            clean_answer = answer.replace("\n", " ").strip()
            filename = safe_generate_tts(clean_answer[:1000])
            if filename:
                print(f" audio_cache/{filename}")

        
        try:
            rules = re.findall(r"\[RULE:(.*?)\](.*)", answer)
            for name, content in rules:
                export_to_folder("rules", name.strip(), content.strip())
        except:
            pass

        try:
            defs = re.findall(r"\[DEFINITION:(.*?)\](.*)", answer)
            for name, content in defs:
                export_to_folder("definitions", name.strip(), content.strip())
        except:
            pass

        
        save_chat_log(query, answer, log_filename=log_filename)

        chat_history.append({"question": query, "answer": answer})
        all_exchanges.append({"question": query, "answer": answer})

        if len(chat_history) >= 5:
            current_summary = summarize_chat_history(chat_history)
            chat_history = []

    return all_exchanges

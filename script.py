# from playwright.sync_api import sync_playwright
# import json
#
# LOGIN_URL = "https://elearn.bits-pilani.ac.in/"
# FAQ_URL = "https://elearn.bits-pilani.ac.in/studentsupport/"
#
# USERNAME = "202217b2052@wilp.bits-pilani.ac.in"
# PASSWORD = "Grlpksbits@2424"
#
# faq_data = []
#
# with sync_playwright() as p:
#     browser = p.chromium.launch(headless=False)
#     page = browser.new_page()
#
#     # 1️⃣ Login
#     page.goto(LOGIN_URL)
#     page.fill("#username", USERNAME)
#     page.fill("#password", PASSWORD)
#     page.click("#submitbtn")
#     page.wait_for_load_state("networkidle")
#
#     # 2️⃣ Go to FAQ page
#     page.goto(FAQ_URL)
#     page.wait_for_timeout(3000)
#
#     # 3️⃣ Get all accordion items
#     accordion_items = page.locator(".accordion-item")
#     count = accordion_items.count()
#
#     print(f"Found {count} FAQ items")
#
#     for i in range(count):
#         item = accordion_items.nth(i)
#
#         # Question
#         question_btn = item.locator("button.accordion-button")
#         question = question_btn.inner_text().strip()
#
#         # Click to expand
#         question_btn.click()
#         page.wait_for_timeout(300)
#
#         # Answer (inside collapse div)
#         answer_div = item.locator(".accordion-collapse")
#         answer = answer_div.inner_text().strip()
#
#         if question and answer:
#             faq_data.append({
#                 "category": item.get_attribute("data-section"),
#                 "question": question,
#                 "answer": answer,
#                 "source": "Official Portal"
#             })
#
#     # 4️⃣ Save JSON
#     with open("faq_dataset.json", "w", encoding="utf-8") as f:
#         json.dump(faq_data, f, indent=2, ensure_ascii=False)
#
#     browser.close()
#
# print(f"✅ Extracted {len(faq_data)} FAQs successfully")

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "faq_dataset.json")
DATA_PATH = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_PATH, "faq_dataset_v1.json")


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra spaces and unwanted characters
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)   # remove extra spaces/newlines
    text = text.replace("\u00a0", " ") # non-breaking space
    return text.strip()


def assign_difficulty(question: str) -> str:
    """
    Simple heuristic to assign difficulty level
    (can be improved later – mention this in report)
    """
    length = len(question.split())

    if length <= 6:
        return "Easy"
    elif length <= 12:
        return "Medium"
    else:
        return "Hard"


def prepare_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_data = []

    for idx, item in enumerate(raw_data, start=1):
        question = clean_text(item.get("question", ""))
        answer = clean_text(item.get("answer", ""))

        if not question or not answer:
            continue  # skip invalid entries

        record = {
            "id": f"FAQ_{idx:04d}",
            "category": item.get("category", "GENERAL"),
            "question": question,
            "answer": answer,
            "difficulty": assign_difficulty(question),
            "source": item.get("source", "Official Portal")
        }

        processed_data.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print("✅ Dataset preparation completed")
    print(f"Total records saved: {len(processed_data)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_dataset()

from playwright.sync_api import sync_playwright
import json
import time

URL = "https://elearn.bits-pilani.ac.in/"
USERNAME = "202217b2052@wilp.bits-pilani.ac.in"
PASSWORD = "Grlpksbits@2424"

output = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # 1️⃣ Open login page
    page.goto(URL)

    # 2️⃣ Login (update selectors)
    page.fill("input[name='username']", USERNAME)
    page.fill("input[name='password']", PASSWORD)
    page.click("button[type='submit']")

    page.wait_for_load_state("networkidle")

    # 3️⃣ Navigate to FAQ page
    page.goto("https://elearn.bits-pilani.ac.in/studentsupport/")
    page.wait_for_timeout(3000)

    # 4️⃣ Click all expandable questions
    questions = page.locator(".faq-question")  # update selector
    count = questions.count()

    for i in range(count):
        questions.nth(i).click()
        page.wait_for_timeout(300)

    # 5️⃣ Extract Q&A
    faq_blocks = page.locator(".faq-item")  # update selector

    for i in range(faq_blocks.count()):
        q = faq_blocks.nth(i).locator(".faq-question").inner_text()
        a = faq_blocks.nth(i).locator(".faq-answer").inner_text()

        output.append({
            "category": "INSTRUCTION",
            "question": q.strip(),
            "answer": a.strip(),
            "source": "Official Portal"
        })

    # 6️⃣ Save JSON
    with open("faq_dataset.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    browser.close()

print("✅ FAQ data extracted successfully")

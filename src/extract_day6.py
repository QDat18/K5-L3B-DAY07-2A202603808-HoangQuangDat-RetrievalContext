"""Script to extract Day 6 Strategy B chunks from PostgreSQL and construct Day 7 benchmark."""
import json
from pathlib import Path
import psycopg

DAY7_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DAY7_DIR / "data"
DAY6_DIR = DAY7_DIR.parent / "K5-L3B-DAY06-HoangQuangDat-2A202603808-DataIngestion"
DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"


def extract_chunks():
    print(f"Connecting to {DB_URL}...")
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, document_id, source, page, section, text, metadata 
                FROM chunks 
                WHERE strategy = 'B' 
                ORDER BY page, chunk_id;
            """)
            rows = cur.fetchall()
    
    chunks = []
    for r in rows:
        meta = r[6] if isinstance(r[6], dict) else {}
        meta["section"] = r[4]
        chunks.append({
            "id": r[0],
            "source_id": str(r[1]),
            "source": r[2],
            "page": int(r[3]),
            "text": r[5],
            "metadata": meta
        })
    print(f"Extracted {len(chunks)} chunks for Strategy B.")
    return chunks


def build_questions(chunks):
    # Load Day 6 20 questions
    day6_q_file = DAY6_DIR / "benchmark" / "questions.json"
    with open(day6_q_file, "r", encoding="utf-8") as f:
        raw_qs = json.load(f)

    questions = []
    cases = []

    # Verify and format 20 answerable questions
    for q in raw_qs:
        ev = q.get("evidence_quote")
        src = str(q.get("gold_source_id"))
        pages = q.get("gold_page")
        if isinstance(pages, list):
            # For q17 which has [45, 46], evidence is on page 45
            page = 45 if 45 in pages else pages[0]
        else:
            page = int(pages)
        
        # Verify hit in chunks
        matched = any(c["source_id"] == src and c["page"] == page and ev in c["text"] for c in chunks)
        if not matched:
            print(f"ERROR: No chunk matched for {q['id']} on page {page}!")
        
        gold_item = {
            "source_id": src,
            "page": page,
            "evidence": ev
        }
        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [gold_item]
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        questions.append(item)

        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })

    # Add 4 unanswerable questions (2 dev, 2 test)
    unanswerable = [
        {
            "id": "q_unans_dev_01",
            "split": "dev",
            "question": "Theo văn bản, mức giá chi trả bảo hiểm y tế cụ thể cho bút tiêm insulin glargine U300 tại bệnh viện tuyến huyện là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Quyết định 5481/QĐ-BYT là hướng dẫn chuyên môn chẩn đoán và điều trị, không quy định biểu giá viện phí hay mức thanh toán BHYT cụ thể cho từng loại biệt dược."
        },
        {
            "id": "q_unans_dev_02",
            "split": "dev",
            "question": "Liều khởi đầu và cách chỉnh liều của thuốc đồng vận thụ thể GLP-1/GIP kép Tirzepatide (Mounjaro) trong điều trị đái tháo đường típ 2 là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Thuốc Tirzepatide được FDA phê chuẩn vào năm 2022, không có trong văn bản hướng dẫn Quyết định 5481/QĐ-BYT ban hành năm 2020."
        },
        {
            "id": "q_unans_test_01",
            "split": "test",
            "question": "Quy trình tầm soát và tiêu chuẩn chẩn đoán đái tháo đường thai kỳ ở tuần thứ 24-28 bằng nghiệm pháp dung nạp glucose theo khuyến cáo mới nhất của ADA 2024 như thế nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Quyết định 5481/QĐ-BYT ban hành năm 2020 dành cho người lớn mắc đái tháo đường típ 2 thông thường, không bao gồm hướng dẫn chuyên biệt thai kỳ theo ADA 2024."
        },
        {
            "id": "q_unans_test_02",
            "split": "test",
            "question": "Liều dùng và chỉ định truyền tĩnh mạch của kháng thể đơn dòng Teplizumab để phòng ngừa tiến triển bệnh đái tháo đường được quy định thế nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Teplizumab là thuốc điều trị đái tháo đường típ 1 được phê duyệt năm 2022, hoàn toàn không có trong hướng dẫn đái tháo đường típ 2 năm 2020."
        }
    ]

    for u in unanswerable:
        cases.append({
            "id": u["id"],
            "split": u["split"],
            "question": u["question"],
            "answerable": False,
            "expected_behavior": u["expected_behavior"],
            "absence_reason": u["absence_reason"]
        })
        questions.append({
            "id": u["id"],
            "split": u["split"],
            "question": u["question"],
            "answerable": False,
            "expected_behavior": u["expected_behavior"],
            "absence_reason": u["absence_reason"]
        })

    return questions, cases


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunks = extract_chunks()
    questions, cases = build_questions(chunks)

    # 1. chunks.json
    chunks_path = DATA_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {chunks_path}")

    # 2. questions_answerable.json (for baseline.py)
    answerable_questions = [q for q in questions if q.get("answerable") is not False]
    ans_q_path = DATA_DIR / "questions_answerable.json"
    with open(ans_q_path, "w", encoding="utf-8") as f:
        json.dump(answerable_questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(answerable_questions)} answerable questions to {ans_q_path}")

    # 3. questions.json (all 24 questions)
    all_q_path = DATA_DIR / "questions.json"
    with open(all_q_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(questions)} total questions to {all_q_path}")

    # 4. cases.json (used by agent_metrics.py)
    cases_path = DATA_DIR / "cases.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} cases to {cases_path}")

    dev_cases = [c for c in cases if c["split"] == "dev"]
    test_cases = [c for c in cases if c["split"] == "test"]
    print(f"\nSummary Benchmark:")
    print(f"  - Dev:  {len(dev_cases)} total ({len([c for c in dev_cases if c['answerable']])} có đáp án + {len([c for c in dev_cases if not c['answerable']])} thiếu evidence)")
    print(f"  - Test: {len(test_cases)} total ({len([c for c in test_cases if c['answerable']])} có đáp án + {len([c for c in test_cases if not c['answerable']])} thiếu evidence)")
    print(f"  - Total: {len(cases)} cases")


if __name__ == "__main__":
    main()

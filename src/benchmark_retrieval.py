import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "starter"))
from baseline import pack_context
from retrievers import DenseRetriever, LexicalRetriever, RerankRetriever

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "starter" / "results"


def is_hit(chunk: dict, gold_list: list[dict]) -> bool:
    """Kiểm tra chunk có đúng nguồn, đúng trang và chứa trích dẫn evidence không."""
    return any(
        chunk["source_id"] == g["source_id"]
        and chunk["page"] == g["page"]
        and g["evidence"] in chunk["text"]
        for g in gold_list
    )


def pack_context_old(results, budget=1500):
    """Cách đóng gói context ban đầu: nếu cả chunk không vừa budget thì bỏ luôn."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def evaluate_retrieval(name: str, search_fn, questions: list[dict], pack_fn=pack_context, budget: int = 1500):
    hit1 = 0
    hit3 = 0
    context_hits = 0
    latencies = []

    for q in questions:
        gold = q["gold"]

        start = time.perf_counter()
        results = search_fn(q)
        latencies.append((time.perf_counter() - start) * 1000)

        # Đánh giá Hit@1
        if results and is_hit(results[0], gold):
            hit1 += 1

        # Đánh giá Evidence Hit@3
        if any(is_hit(r, gold) for r in results[:3]):
            hit3 += 1

        # Đánh giá Context Evidence Rate
        packed_text, packed_ids = pack_fn(results[:3], budget=budget)
        context_chunks = [r for r in results[:3] if r["id"] in packed_ids]
        if any(is_hit(c, gold) for c in context_chunks):
            context_hits += 1

    N = len(questions)
    return {
        "name": name,
        "N": N,
        "hit1": f"{hit1}/{N} ({hit1/N*100:.1f}%)",
        "hit3": f"{hit3}/{N} ({hit3/N*100:.1f}%)",
        "context_rate": f"{context_hits}/{N} ({context_hits/N*100:.1f}%)",
        "avg_latency_ms": f"{sum(latencies)/len(latencies):.1f} ms/câu, 0 USD"
    }


def main():
    with open(DATA_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(DATA_DIR / "questions_answerable.json", "r", encoding="utf-8") as f:
        all_qs = json.load(f)

    dev_qs = [q for q in all_qs if q["split"] == "dev"]
    test_qs = [q for q in all_qs if q["split"] == "test"]

    print("=== KHỞI TẠO CÁC RETRIEVER ===")
    dense = DenseRetriever()
    lexical = LexicalRetriever(chunks)
    rerank = RerankRetriever(base_retriever=dense, candidate_k=10)

    # Đọc lại queries từ trace agent_dev (lần đầu, do model viết lại)
    v1_queries = {}
    agent_dev_trace = RESULTS_DIR / "agent_dev" / "traces.json"
    if agent_dev_trace.exists():
        with open(agent_dev_trace, "r", encoding="utf-8") as f:
            for t in json.load(f):
                evs = t.get("events", [])
                s_ev = [e for e in evs if e.get("event") == "search"]
                if s_ev:
                    v1_queries[t["query_id"]] = s_ev[0]["query"]

    print(f"\n=== ĐO ĐẠC THỰC NGHIỆM DEV ({len(dev_qs)} CÂU) ===")
    
    # 1. Dev Dense (budget 800)
    res_dense_dev = evaluate_retrieval(
        "Phương pháp 2: Dense (dev)",
        lambda q: dense.search(q["question"], filters=q.get("filter"), top_k=3),
        dev_qs,
        pack_fn=pack_context,
        budget=800
    )

    # 2. Dev Lexical (budget 800)
    res_lexical_dev = evaluate_retrieval(
        "Phương pháp 3: Lexical BM25 (dev)",
        lambda q: lexical.search(q["question"], filters=q.get("filter"), top_k=3),
        dev_qs,
        pack_fn=pack_context,
        budget=800
    )

    # 3. Rerank, lần đầu: query do model viết lại, budget 1500, pack cũ
    res_rerank_v1_dev = evaluate_retrieval(
        "Rerank, lần đầu (query viết lại)",
        lambda q: rerank.search(v1_queries.get(q["id"], q["question"]), filters=q.get("filter"), top_k=3),
        dev_qs,
        pack_fn=pack_context_old,
        budget=1500
    )

    # 4. Cải thiện 1: query nguyên câu hỏi, budget 1500, pack cũ
    res_rerank_c1_dev = evaluate_retrieval(
        "Cải thiện 1 (query gốc, pack cũ)",
        lambda q: rerank.search(q["question"], filters=q.get("filter"), top_k=3),
        dev_qs,
        pack_fn=pack_context_old,
        budget=1500
    )

    # 5. Cải thiện 2: query nguyên câu hỏi, budget 1500, pack mới (cắt chunk)
    res_rerank_c2_dev = evaluate_retrieval(
        "Cải thiện 2 (query gốc, pack mới cắt chunk)",
        lambda q: rerank.search(q["question"], filters=q.get("filter"), top_k=3),
        dev_qs,
        pack_fn=pack_context,
        budget=1500
    )

    print(f"\n=== ĐO ĐẠC THỰC NGHIỆM TEST ({len(test_qs)} CÂU) ===")

    # 6. Cấu hình đã chọn (Rerank test, budget 1500)
    res_rerank_test = evaluate_retrieval(
        "Cấu hình đã chọn: Rerank (test)",
        lambda q: rerank.search(q["question"], filters=q.get("filter"), top_k=3),
        test_qs,
        pack_fn=pack_context,
        budget=1500
    )

    # 7. Dense đối chiếu test (budget 1500)
    res_dense_test = evaluate_retrieval(
        "Dense đối chiếu (test)",
        lambda q: dense.search(q["question"], filters=q.get("filter"), top_k=3),
        test_qs,
        pack_fn=pack_context,
        budget=1500
    )

    # 8. Lexical đối chiếu test (budget 1500)
    res_lexical_test = evaluate_retrieval(
        "Lexical đối chiếu (test)",
        lambda q: lexical.search(q["question"], filters=q.get("filter"), top_k=3),
        test_qs,
        pack_fn=pack_context,
        budget=1500
    )

    all_results = [
        res_dense_dev,
        res_lexical_dev,
        res_rerank_v1_dev,
        res_rerank_c1_dev,
        res_rerank_c2_dev,
        res_rerank_test,
        res_dense_test,
        res_lexical_test,
    ]

    print("\n" + "=" * 90)
    print(f"{'Cấu hình':<40} | {'Hit@1':<14} | {'Hit@3':<14} | {'Context Rate':<14} | {'Thời gian / Chi phí'}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['name']:<40} | {r['hit1']:<14} | {r['hit3']:<14} | {r['context_rate']:<14} | {r['avg_latency_ms']}")
    print("=" * 90)


if __name__ == "__main__":
    main()

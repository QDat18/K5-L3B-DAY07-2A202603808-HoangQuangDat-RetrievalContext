import json 
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import sys
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

DAY7_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(DAY7_DIR / "starter"))
from agentic import run_agent
from retrievers import DenseRetriever, LexicalRetriever, RerankRetriever
from grounding import cited_text, quote_in_evidence, ungrounded_numbers
load_dotenv(DAY7_DIR / ".env")

def _log(msg: str) -> None:
    """In log nhưng không bao giờ làm đổi kết quả (console cp1252 có thể không in được tiếng Việt)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


ABSTAIN = {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

def extract_json(text: str) -> dict[str, Any]:
    """Trích xuất JSON chuẩn xác, bỏ qua mọi text/reasoning đi kèm."""
    # 1. Tìm trong markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # 2. Dùng raw_decode từ vị trí dấu { đầu tiên để tránh lỗi Extra data
    start_idx = text.find("{")
    if start_idx != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start_idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 3. Fallback chuẩn
    return json.loads(text.strip())

class NimController: 
    def __init__(self, model_name: str = "nvidia/nemotron-3.5-lightning-30b-a3b", api_key: str | None = None,):
        self.model_name=model_name
        self.api_key= api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found. Please set it in your .env file or pass it as an argument.")
        self.client=OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=self.api_key, timeout=30.0,)

    def _chat(self, prompt: str, max_tokens: int, retries: int = 2, temperature: float = 0.1,
              api_attempts: int = 4) -> str:
        """Gọi model và chỉ trả về `content` cuối cùng (KHÔNG dùng reasoning_content).

        Tắt thinking; nếu content rỗng hoặc bị cắt (finish_reason=length) thì retry với
        max_tokens gấp đôi. Lỗi API tạm thời (429, mạng) được retry với backoff 1s, 2s, 4s.
        Hết lượt retry thì raise để caller abstain.
        """
        for _ in range(retries + 1):
            res = None
            for attempt in range(api_attempts):
                try:
                    res = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    break
                except Exception:
                    if attempt == api_attempts - 1:
                        raise
                    time.sleep(2 ** attempt)
            choice = res.choices[0]
            content = (choice.message.content or "").strip()
            if content and choice.finish_reason != "length":
                return content
            max_tokens *= 2
        raise RuntimeError("Model không trả về content hoàn chỉnh (rỗng hoặc bị cắt do max_tokens)")

    def _verify_support(self, question: str, answer: str, evidence: str) -> bool:
        """LLM kiểm tra từng mệnh đề của answer có được evidence nói rõ ràng không.

        Lỗi verifier (mạng, JSON hỏng) => coi là KHÔNG được hỗ trợ (fail closed).
        """
        prompt = f"""Bạn là người kiểm chứng nghiêm ngặt. Chỉ dựa vào ĐOẠN TRÍCH dưới đây, hãy kiểm tra CÂU TRẢ LỜI.

CÂU HỎI: {question}

ĐOẠN TRÍCH:
\"\"\"
{evidence}
\"\"\"

CÂU TRẢ LỜI: {answer}

Cách đánh giá:
1. Xác định Ý CHÍNH của câu trả lời, tức là thông tin trực tiếp trả lời CÂU HỎI (con số, liều, ngưỡng, tên thuốc, quyết định).
2. supported = true nếu Ý CHÍNH đó được đoạn trích nói rõ ràng, đúng đối tượng/tình huống được hỏi. Thông tin phụ, bổ sung hoặc diễn đạt lại không làm câu trả lời bị loại, miễn là không mâu thuẫn với đoạn trích.
3. supported = false nếu: Ý CHÍNH không có trong đoạn trích; số/liều thuộc về thuốc, nhóm bệnh nhân hay tình huống khác với đối tượng được hỏi; câu trả lời mâu thuẫn với đoạn trích; hoặc câu trả lời trả lời một câu hỏi khác với CÂU HỎI.
Chỉ so với ĐOẠN TRÍCH, không dùng kiến thức bên ngoài.
Trả về đúng một JSON, điền THEO THỨ TỰ các trường sau (không viết gì ngoài JSON):
{{"main_point": "ý chính của câu trả lời, 1 câu ngắn", "quote": "câu NGUYÊN VĂN trong đoạn trích nói rõ ý chính đó, hoặc rỗng nếu không có", "supported": true hoặc false}}
supported phải NHẤT QUÁN với hai trường trước: true khi và chỉ khi quote không rỗng và khớp ý chính, đúng đối tượng của câu hỏi."""
        try:
            data = extract_json(self._chat(prompt, max_tokens=512, temperature=0.0))
            ok = data.get("supported") is True and quote_in_evidence(str(data.get("quote") or ""), evidence)
            if not ok:
                _log(f"[Grounding] Ý chính: {data.get('main_point')!r} | quote: {str(data.get('quote'))[:120]!r} | supported={data.get('supported')}")
            return ok
        except Exception as e:
            _log(f"[Grounding] Verifier lỗi ({e}) -> coi là không hỗ trợ")
            return False

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            # Dùng nguyên câu hỏi tiếng Việt làm query: query do LLM viết lại (tiếng Anh + mã
            # "5481/QĐ-BYT") làm lệch embedding so với kho tài liệu tiếng Việt và không ổn định giữa các lần chạy.
            return {"action": "search", "query": question}
            
        last_turn = history[-1]
        context_text = last_turn.get("context", "")
        context_ids = last_turn.get("context_ids", [])
        prompt = f"""
                You are a professional medical assistant specialized in answering questions based strictly on clinical guidelines for the treatment of Type 2 Diabetes Mellitus (T2DM).

                User's question: "{question}"

                The following excerpts are retrieved from the provided medical guideline documents. Each excerpt has a unique ID in the format [B-5481-...]:
                \"\"\"
                {context_text}
                \"\"\"

                MANDATORY INSTRUCTIONS:

                1. Carefully read and analyze all the provided excerpts.

                2. IF THE EXCERPTS CONTAIN SUFFICIENT EVIDENCE TO ANSWER THE QUESTION:
                - Provide an accurate and comprehensive answer based strictly on the information in the provided documents.
                - Do not add information from external knowledge or sources.
                - Answer ONLY in Vietnamese (tiếng Việt), concisely, and directly answer the question asked; reuse the exact wording and numbers of the excerpts. Never answer in English.
                - START the answer with the direct conclusion (e.g. "Không", "Có", the number/threshold, or the drug name), then add at most one short supporting sentence. Do not list unrelated patient groups.
                - Use only the excerpt(s) that actually answer the question; do not cite or restate unrelated excerpts.
                - You MUST insert the corresponding chunk ID [chunk-id] immediately after each statement or piece of information derived from that excerpt.
                - Include all referenced chunk IDs in the "citations" array.
                - Return the following JSON structure:
                    {{"action": "answer", "answer": "Your answer containing inline citations [chunk-id]...", "citations": ["chunk-id"]}}

                3. IF THE EXCERPTS DO NOT CONTAIN SUFFICIENT EVIDENCE OR DO NOT PROVIDE A RELIABLE BASIS FOR ANSWERING:
                - You MUST refuse to provide an answer.
                - Do NOT make assumptions, infer unsupported conclusions, or fabricate medical information.
                - Return the following JSON structure:
                    {{"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}}

                OUTPUT REQUIREMENTS:
                - Return EXACTLY ONE valid JSON object.
                - Do NOT include any explanations, comments, Markdown formatting, or additional text outside the JSON object.
                - Ensure the JSON is syntactically valid and properly escaped.
                """
        try:
            raw_text = self._chat(prompt, max_tokens=1500)
            data = extract_json(raw_text)
            action = data.get("action")
            if action == "answer":
                answer = data.get("answer", "")
                citations = [str(cid).strip().strip("[]").strip() for cid in (data.get("citations") or [])]
                valid_citations = [cid for cid in citations if cid in context_ids]
                for cid in valid_citations:
                    if f"[{cid}]" not in answer:
                        answer += f" [{cid}]"
                if not valid_citations:
                    return {
                        "action": "answer",
                        "answer": answer.strip() if answer else "Không đủ bằng chứng để trả lời.",
                        "citations": [],
                    }
                evidence = cited_text(context_text, valid_citations)
                bad = ungrounded_numbers(answer, evidence, question)
                if bad:
                    _log(f"[Grounding] Số không có trong chunk được trích: {bad} -> abstain | answer bị loại: {answer[:300]!r}")
                    return dict(ABSTAIN)
                if not self._verify_support(question, answer, evidence):
                    _log(f"[Grounding] Verifier: không được chunk hỗ trợ -> abstain | answer bị loại: {answer[:300]!r}")
                    return dict(ABSTAIN)
                return {
                    "action": "answer",
                    "answer": answer.strip(),
                    "citations": valid_citations,
                }
            elif action == "search" and data.get("query"):
                return data
        except Exception as e:
            print(f"[Controller] Failed to create answer: {e}")
            return {
                "action": "answer",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
            }
        return {
            "action": "answer",
            "answer": "Không đủ bằng chứng để trả lời.",
            "citations": [],
        }

def run_benchmark_agent(
    retriever_name: str,
    retriever,
    split: str = "dev",
    budget: int = 1500,
    top_k: int = 3,
    output_dir: Path | None = None,
):
    """Chạy Agent trên toàn bộ câu hỏi của một split (dev hoặc test)."""
    cases_file = DAY7_DIR / "data" / "cases.json"
    questions_file = DAY7_DIR / "data" / "questions.json"
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = [c for c in json.load(f) if c["split"] == split]
    with open(questions_file, "r", encoding="utf-8") as f:
        q_map = {q["id"]: q for q in json.load(f)}
    controller = NimController()
    traces = []
    print(f"\n>>> ĐANG CHẠY AGENT ({retriever_name}) TRÊN TẬP {split.upper()} ({len(cases)} CÂU)...", flush=True)
    print(f"    (Budget = {budget} chars, Top-K = {top_k})\n", flush=True)
    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        q_obj = q_map[cid]
        question = q_obj["question"]
        filters = q_obj.get("filter")
        print(f"[{idx}/{len(cases)}] {cid} ({'Có đáp án' if case['answerable'] else 'Thiếu bằng chứng'}):", flush=True)
        print(f"    Q: {question}", flush=True)
        print(f"    -> Đang xử lý (suy luận + tìm kiếm + trả lời)...", end="", flush=True)
        try:
            # Gọi vòng lặp Agent ReAct
            agent_res = run_agent(
                question=question,
                controller=controller,
                retriever=retriever,
                filters=filters,
                max_searches=3,
                top_k=top_k,
                budget=budget,
            )
            status = agent_res.get("status")
            answer = agent_res.get("answer")
            cits = agent_res.get("citations", [])
            print(f"\r    Status: {status.upper()} | Citations: {cits}", flush=True)
            print(f"    Ans: {answer[:120]}..." if len(answer) > 120 else f"    Ans: {answer}", flush=True)
            print("-" * 60, flush=True)
            traces.append(
                {
                    "query_id": cid,
                    "split": split,
                    "status": status,
                    "answer": answer,
                    "citations": cits,
                    "events": agent_res.get("events", []),
                }
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            traces.append({"query_id": cid, "split": split, "status": "error", "error": str(e)})
    # Lưu trace ra thư mục output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "traces.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Đã lưu {len(traces)} traces vào: {trace_path}")
    return traces
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chạy Agentic RAG trên benchmark")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--method", choices=["dense", "lexical", "rerank"], default="rerank")
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DAY7_DIR / "starter" / "results" / "agent_dev")
    args = parser.parse_args()
    # Khởi tạo retriever tương ứng
    with open(DAY7_DIR / "data" / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if args.method == "dense":
        retriever = DenseRetriever()
    elif args.method == "lexical":
        retriever = LexicalRetriever(chunks)
    else:
        dense_base = DenseRetriever()
        retriever = RerankRetriever(base_retriever=dense_base, candidate_k=10)
    run_benchmark_agent(
        retriever_name=args.method,
        retriever=retriever,
        split=args.split,
        budget=args.budget,
        top_k=args.top_k,
        output_dir=args.output,
    )

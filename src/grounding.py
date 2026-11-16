"""Kiểm tra grounding: câu trả lời phải dựa trên chính các chunk được trích."""
import re

_BLOCK = re.compile(r"^\[([^\s|\]]+) \|[^\]]*\]\n", re.M)
# ID chunk (B-5481-U013-002) ở bất kỳ dạng nào: trong [..], [a, b] hoặc đứng trơ trọi.
_CITATION = re.compile(r"\[[A-Za-z]+-[^\]]*\]|[A-Za-z]+-\d+-U\d+-\d+")
# Số thứ tự liệt kê: "1)", "2)" hoặc "3. " đầu dòng — không phải thông tin y khoa.
_ENUMERATOR = re.compile(r"(?<![\w.,])\d{1,2}\)|(?m:^\s*\d{1,2}\.\s)")
_NUMBER =re.compile(r"\d+(?:[.,]\d+)?")


def split_context(context_text: str) -> dict[str, str]:
    """Tách context do pack_context tạo ra thành {chunk_id: text}."""
    matches = list(_BLOCK.finditer(context_text))
    out = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context_text)
        out[m.group(1)] = context_text[m.end():end].strip()
    return out


def cited_text(context_text: str, citations: list[str]) -> str:
    chunks = split_context(context_text)
    return "\n".join(chunks[c] for c in citations if c in chunks)


def _norm(num: str) -> str:
    num = num.replace(",", ".")
    return re.sub(r"\.0+$", "", num)  # 7,0 == 7


def ungrounded_numbers(answer: str, evidence: str, question: str = "") -> list[str]:
    """Các số trong answer (bỏ ID trích dẫn) không có trong evidence hoặc câu hỏi."""
    body = _ENUMERATOR.sub(" ", _CITATION.sub(" ", answer))
    have = {_norm(n) for n in _NUMBER.findall(evidence + " " + question)}
    missing = []
    for n in _NUMBER.findall(body):
        n = _norm(n)
        if n not in have and n not in missing:
            missing.append(n)
    return missing


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_evidence(quote: str, evidence: str, min_len: int = 12) -> bool:
    """Trích dẫn verifier đưa ra phải là đoạn nguyên văn của evidence.

    Bỏ qua hoa/thường và khoảng trắng. Quote có thể gồm nhiều đoạn nối bằng "..." hoặc "…":
    MỌI đoạn đều phải có nguyên văn trong evidence, và ít nhất một đoạn dài >= min_len.
    """
    parts = [_squash(p).strip(" .") for p in re.split(r"\.{3,}|…", quote or "")]
    parts = [p for p in parts if p]
    ev = _squash(evidence)
    return any(len(p) >= min_len for p in parts) and all(p in ev for p in parts)

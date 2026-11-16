"""Module cài đặt 3 phương pháp Retrieval: Dense, Lexical (BM25) và Rerank."""
import os
# Bật chế độ offline để tải model trực tiếp từ cache máy, không tốn thời gian gọi mạng HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import math
import re
import unicodedata
from collections import Counter
from typing import Any

import psycopg
from sentence_transformers import CrossEncoder, SentenceTransformer


def normalize_tokens(text: str) -> list[str]:
    """Tách từ đơn giản cho tiếng Việt, chuẩn hóa Unicode NFC và chữ thường."""
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"\w+", text)


# ==============================================================================
# 1. DENSE RETRIEVER (Vector DB pgvector + BAAI/bge-m3)
# ==============================================================================
class DenseRetriever:
    """Truy xuất dày (Dense) sử dụng pgvector và embedding model Day 6."""

    def __init__(
        self,
        db_url: str = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab",
        model_name: str = "BAAI/bge-m3",
        strategy: str = "B",
    ):
        self.db_url = db_url
        self.strategy = strategy
        print(f"[DenseRetriever] Đang tải embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, local_files_only=True)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        query_vec = self.model.encode(
            query, normalize_embeddings=True
        ).tolist()
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec) + "]"

        params = [self.strategy, vec_literal]
        filter_clause = ""
        if filters and "section" in filters and filters["section"]:
            filter_clause = " AND section = %s"
            params.append(filters["section"])

        params.append(top_k)
        sql = f"""
            SELECT chunk_id, document_id, source, page, section, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE strategy = %s {filter_clause}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """

        sql_params = [vec_literal, self.strategy]
        if filter_clause:
            sql_params.append(filters["section"])
        sql_params.extend([vec_literal, top_k])

        results = []
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, sql_params)
                for r in cur.fetchall():
                    meta = r[6] if isinstance(r[6], dict) else {}
                    meta["section"] = r[4]
                    results.append(
                        {
                            "id": r[0],
                            "source_id": str(r[1]),
                            "source": r[2],
                            "page": int(r[3]),
                            "text": r[5],
                            "metadata": meta,
                            "score": float(r[7]),
                            "method": "dense",
                        }
                    )
        return results


# ==============================================================================
# 2. LEXICAL RETRIEVER (BM25 trên 234 chunks)
# ==============================================================================
class LexicalRetriever:
    """Truy xuất từ khóa bằng thuật toán BM25 kinh điển."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # Tiền xử lý token và thống kê độ dài văn bản
        self.doc_tokens = [normalize_tokens(c["text"]) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / (len(self.doc_lens) or 1)
        self.N = len(chunks)

        # Tính Document Frequency (DF)
        df_counter = Counter()
        for tokens in self.doc_tokens:
            df_counter.update(set(tokens))

        # Tính Inverse Document Frequency (IDF) theo chuẩn BM25
        self.idf = {}
        for term, df in df_counter.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_tokens = normalize_tokens(query)
        target_section = filters.get("section") if filters else None

        scores = []
        for i, chunk in enumerate(self.chunks):
            # Áp dụng bộ lọc metadata nếu có
            if target_section and chunk["metadata"].get("section") != target_section:
                continue

            doc_len = self.doc_lens[i]
            tf_map = Counter(self.doc_tokens[i])
            score = 0.0

            for t in q_tokens:
                if t in tf_map:
                    tf = tf_map[t]
                    idf = self.idf.get(t, 0.0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)

            if score > 0:
                scores.append((score, chunk))

        # Sắp xếp điểm BM25 giảm dần
        scores.sort(key=lambda x: -x[0])
        results = []
        for s, c in scores[:top_k]:
            results.append({**c, "score": float(s), "method": "lexical_bm25"})
        return results


# ==============================================================================
# 3. RERANK RETRIEVER (Dense Candidate Search + Cross-Encoder Rerank)
# ==============================================================================
class RerankRetriever:
    """Truy xuất 2 giai đoạn: Lấy candidate từ Dense -> Xếp hạng lại bằng Reranker."""

    def __init__(
        self,
        base_retriever: DenseRetriever,
        reranker_name: str = "BAAI/bge-reranker-base",
        candidate_k: int = 10,
    ):
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        print(f"[RerankRetriever] Đang tải reranker: {reranker_name}...")
        self.reranker = CrossEncoder(reranker_name, local_files_only=True)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        # Giai đoạn 1: Lấy top-N candidate từ Dense
        candidates = self.base_retriever.search(
            query, filters=filters, top_k=self.candidate_k
        )
        if not candidates:
            return []

        # Giai đoạn 2: Tạo cặp (query, chunk_text) và tính điểm Cross-Encoder
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Gán lại điểm rerank
        scored_candidates = []
        for c, r_score in zip(candidates, rerank_scores):
            scored_candidates.append(
                {
                    **c,
                    "score": float(r_score),
                    "dense_score": c.get("score"),
                    "method": "rerank",
                }
            )

        # Sắp xếp lại theo điểm Cross-Encoder giảm dần và cắt lấy top_k
        scored_candidates.sort(key=lambda x: -x["score"])
        return scored_candidates[:top_k]
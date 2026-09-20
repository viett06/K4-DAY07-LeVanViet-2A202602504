from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import GeminiEmbedder, OpenAIEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

# ==============================================================================
# CẤU HÌNH CHIẾN LƯỢC DÀNH CHO TỪNG THÀNH VIÊN TRONG NHÓM:
#
# Chọn 1 trong 4 chiến lược dưới đây:
#   1. "recursive"     -> Mai Quang Dũng (RecursiveChunker: đệ quy ưu tiên \n\n, \n, . )
#   2. "fixed_size"    -> Thành viên phụ trách FixedSizeChunker (cắt cố định + overlap)
#   3. "by_sentences"  -> Thành viên phụ trách SentenceChunker (gom theo ranh giới câu)
#   4. "heading"       -> Thành viên phụ trách HeadingChunker (cắt theo section tiêu đề)
#
# Thành viên nào chạy chỉ cần giữ nguyên hoặc đổi giá trị ở biến DEFAULT_STRATEGY dưới đây,
# hoặc chạy qua terminal: python bench.py --strategy <ten_chien_luoc>
# ==============================================================================
DEFAULT_STRATEGY = "fixed_size"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 50
DEFAULT_SENTENCES_PER_CHUNK = 3

CACHE_FILE = Path("data/ecommerce/.embedding_cache.json")
GEMINI_CACHE_FILE = Path("data/ecommerce/.embedding_cache_gemini.json")


class CachedGeminiEmbedder:
    """Gemini embedder kèm cache đĩa để không gọi lại API khi chạy benchmark lần 2."""

    def __init__(self) -> None:
        self.embedder = GeminiEmbedder()
        self._backend_name = getattr(self.embedder, "_backend_name", "gemini")
        self.cache: dict[str, list[float]] = {}
        self._new_entries = 0
        if GEMINI_CACHE_FILE.exists():
            try:
                self.cache = json.loads(GEMINI_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _hash(self, text: str) -> str:
        cache_key = f"{self._backend_name}\0{text}"
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        delay = 25
        last_error: Exception | None = None
        for _ in range(8):
            try:
                response = self.embedder.client.models.embed_content(
                    model=self.embedder.model_name,
                    contents=texts,
                )
                return [[float(value) for value in item.values] for item in response.embeddings]
            except Exception as error:
                last_error = error
                message = str(error)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    print(f"  Gemini rate limit, đợi {delay}s rồi thử lại...")
                    time.sleep(delay)
                    delay = min(delay + 10, 60)
                    continue
                raise
        raise RuntimeError(f"Gemini embedding failed after retries: {last_error}") from last_error

    def embed_many(self, texts: list[str], batch_size: int = 20) -> None:
        missing: list[tuple[str, str]] = []
        seen: set[str] = set()
        for text in texts:
            digest = self._hash(text)
            if digest in self.cache or digest in seen:
                continue
            seen.add(digest)
            missing.append((digest, text))

        total = len(missing)
        if total:
            print(f"  cần embed {total} chunk mới (đã cache {len(texts) - total})")
        for start in range(0, total, batch_size):
            batch = missing[start : start + batch_size]
            vectors = self._embed_batch([text for _, text in batch])
            for (digest, _), vector in zip(batch, vectors):
                self.cache[digest] = vector
            self._save()
            done = min(start + batch_size, total)
            print(f"  embedded {done}/{total} new chunks")

    def __call__(self, text: str) -> list[float]:
        digest = self._hash(text)
        if digest in self.cache:
            return self.cache[digest]
        vector = self._embed_batch([text])[0]
        self.cache[digest] = vector
        self._save()
        return vector

    def _save(self) -> None:
        try:
            GEMINI_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            GEMINI_CACHE_FILE.write_text(json.dumps(self.cache), encoding="utf-8")
            self._new_entries = 0
        except Exception:
            pass


class CachedOpenAIEmbedder:
    """OpenAI embedder kèm bộ đệm cache trên ổ đĩa để tránh tốn token khi chạy lại."""

    def __init__(self) -> None:
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedder = OpenAIEmbedder(model_name=model_name)
        self.cache: dict[str, list[float]] = {}
        self._new_entries = 0
        if CACHE_FILE.exists():
            try:
                self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _hash(self, text: str) -> str:
        cache_key = f"{self.embedder.model_name}\0{text}"
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

    def embed_many(self, texts: list[str], batch_size: int = 100) -> None:
        """Pre-fill the cache with batched embedding API calls."""
        missing: dict[str, str] = {}
        for value in texts:
            digest = self._hash(value)
            if digest not in self.cache:
                missing[digest] = value

        pending = list(missing.items())
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            response = self.embedder.client.embeddings.create(
                model=self.embedder.model_name,
                input=[value for _, value in batch],
            )
            for (digest, _), item in zip(batch, response.data):
                self.cache[digest] = [float(value) for value in item.embedding]
            self._new_entries += len(batch)
            self._save()

    def __call__(self, text: str) -> list[float]:
        h = self._hash(text)
        if h in self.cache:
            return self.cache[h]
        vec = self.embedder(text)
        self.cache[h] = vec
        self._new_entries += 1
        if self._new_entries >= 25:
            self._save()
        return vec

    def _save(self) -> None:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(self.cache), encoding="utf-8")
            self._new_entries = 0
        except Exception:
            pass


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Tách frontmatter YAML và phần nội dung thân."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()
            meta: dict[str, str] = {}
            for line in fm_text.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    meta[k] = v
            return meta, body
    return {}, content


def load_and_chunk_corpus(corpus_dir: Path, chunker_instance: Any) -> list[Document]:
    """Đọc mọi file .md trong data/ecommerce và chia nhỏ thành Document chunks."""
    all_docs: list[Document] = []
    md_files = sorted(corpus_dir.glob("*.md"))

    for file_path in md_files:
        raw_text = file_path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw_text)

        chunks = chunker_instance.chunk(body)
        doc_id = meta.get("doc_id", file_path.stem)

        for i, chunk_text in enumerate(chunks):
            chunk_meta = {
                **meta,
                "doc_id": doc_id,
                "source": str(file_path),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            all_docs.append(
                Document(
                    id=f"{doc_id}#{i}",
                    content=chunk_text,
                    metadata=chunk_meta,
                )
            )

    return all_docs


def llm_answer_fn(prompt: str) -> str:
    """Tạo câu trả lời dựa trên LLM (sử dụng OpenAI nếu khả dụng hoặc trích xuất chuẩn)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and not api_key.startswith("your-key"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý RAG hỗ trợ chính sách sàn TMĐT Shopee. Hãy trả lời ngắn gọn, chính xác dựa trên ngữ cảnh được cung cấp, có trích dẫn nguồn [1], [2].",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return completion.choices[0].message.content or ""
        except Exception:
            pass

    return f"[RAG Agent Answer] Phản hồi dựa trên ngữ cảnh: {prompt[:250]}..."


# 5 Benchmark Queries thống nhất của nhóm
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Người mua có thể gửi yêu cầu trả hàng hoàn tiền trong thời hạn bao lâu sau khi nhận hàng?",
        "gold_answer": "Trong vòng 15 ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công (riêng thực phẩm tươi sống và đông lạnh là trong vòng 24 giờ).",
        "gold_doc": "shopee-return-refund-buyer",
        "key_fact": "15 (mười lăm) ngày",
        "answer_facts": ["15 ngày", "24 giờ"],
        "filter": None,
        "note": "Truy xuất thời hạn người mua yêu cầu trả hàng",
    },
    {
        "id": 2,
        "query": "Thời hạn xử lý và phản hồi khiếu nại yêu cầu trả hàng hoàn tiền của Người Bán là bao lâu?",
        "gold_answer": "Người Bán cần gửi phản hồi trong vòng 02 ngày lịch kể từ ngày nhận được thông báo của Shopee nếu không đồng ý với quyết định hoàn tiền.",
        "gold_doc": "shopee-return-refund-seller",
        "key_fact": "02 ngày lịch",
        "answer_facts": ["02 ngày lịch"],
        "filter": {"audience": "seller"},
        "note": "Câu A/B Filter: Lọc audience='seller' để phân biệt với thời hạn 15 ngày của Người mua",
    },
    {
        "id": 3,
        "query": "Người bán có hành vi gian lận tạo đơn hàng ảo trên Shopee bị xử phạt bồi thường bao nhiêu tiền cho mỗi đơn hàng vi phạm?",
        "gold_answer": "Người Bán sẽ phải bồi thường cho Shopee một khoản tiền lên đến 10.000.000 VND (Mười triệu đồng) cho từng đơn hàng vi phạm Chính Sách, được cấn trừ vào Số dư Tài Khoản Shopee.",
        "gold_doc": "shopee-antifraud-seller-penalty",
        "key_fact": "10.000.000",
        "answer_facts": ["10.000.000"],
        "filter": {"audience": "seller"},
        "note": "Mức phạt bồi thường vi phạm gian lận đơn ảo",
    },
    {
        "id": 4,
        "query": "Quy định về hạn sử dụng của hàng hóa khi Người Bán giao đi trên Shopee phải còn lại ít nhất bao nhiêu?",
        "gold_answer": "Hàng hóa khi giao đi phải còn ít nhất 30% thời hạn sử dụng và còn ít nhất 30 ngày, tính từ thời điểm hiện tại đến ngày hết hạn.",
        "gold_doc": "shopee-seller-listing-rules",
        "key_fact": "30%",
        "answer_facts": ["30%", "30 ngày"],
        "filter": None,
        "note": "Quy định về hạn sử dụng hàng hóa khi đăng bán",
    },
    {
        "id": 5,
        "query": "Sản phẩm mua trên Shopee được bảo hành miễn phí khi đáp ứng những điều kiện nào?",
        "gold_answer": "Sản phẩm bị lỗi kỹ thuật do nhà sản xuất, còn trong thời hạn bảo hành, có hóa đơn điện tử hoặc mã đơn hàng (ID đơn hàng), phiếu/tem bảo hành của nhà sản xuất còn nguyên vẹn.",
        "gold_doc": "shopee-warranty-buyer",
        "key_fact": "lỗi kỹ thuật",
        "answer_facts": ["lỗi kỹ thuật", "thời hạn bảo hành", "mã đơn hàng"],
        "filter": {"audience": "buyer"},
        "note": "Điều kiện bảo hành sản phẩm Shopee",
    },
]


CHUNKER_REGISTRY = {
    "fixed_size": (
        lambda: FixedSizeChunker(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP),
        f"FixedSizeChunker (size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_OVERLAP})",
    ),
    "by_sentences": (
        lambda: SentenceChunker(max_sentences_per_chunk=DEFAULT_SENTENCES_PER_CHUNK),
        f"SentenceChunker (max_sentences={DEFAULT_SENTENCES_PER_CHUNK})",
    ),
    # "heading": (
    #     lambda: HeadingChunker(max_size=DEFAULT_CHUNK_SIZE),
    #     f"HeadingChunker (max_size={DEFAULT_CHUNK_SIZE}, hierarchical Markdown sections)",
    # ),
    "recursive": (
        lambda: RecursiveChunker(chunk_size=DEFAULT_CHUNK_SIZE),
        f"RecursiveChunker (chunk_size={DEFAULT_CHUNK_SIZE})",
    ),
}

CHUNKER_ALIASES = {
    "fixed": "fixed_size",
    "fixedsize": "fixed_size",
    "sentence": "by_sentences",
    "sentences": "by_sentences",
    "headingchunker": "heading",
    "section": "heading",
}


def get_chunker_by_name(name: str):
    normalized_name = name.strip().lower()
    strategy_name = CHUNKER_ALIASES.get(normalized_name, normalized_name)
    try:
        factory, display_name = CHUNKER_REGISTRY[strategy_name]
    except KeyError as error:
        valid_names = ", ".join(CHUNKER_REGISTRY)
        raise ValueError(f"Unknown chunking strategy '{name}'. Choose one of: {valid_names}") from error
    return factory(), display_name


def main():
    parser = argparse.ArgumentParser(description="Lab 07 Benchmark Runner — K4-L3B TMĐT Shopee")
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        help=f"Chiến lược chunking cần chạy (mặc định: {DEFAULT_STRATEGY})",
    )
    args = parser.parse_args()
    selected_strategy = args.strategy.strip().lower()
    selected_strategy = CHUNKER_ALIASES.get(selected_strategy, selected_strategy)
    if selected_strategy not in CHUNKER_REGISTRY:
        print(
            f"[!] Chiến lược '{args.strategy}' không dùng được. "
            f"Chuyển sang '{DEFAULT_STRATEGY}'."
        )
        selected_strategy = DEFAULT_STRATEGY

    corpus_dir = Path("data/ecommerce")
    chunker, display_name = get_chunker_by_name(selected_strategy)

    print("=================================================================")
    print("        BÁO CÁO BENCHMARK RETRIEVAL — K4-L3B TMĐT SHOPEE        ")
    print(f"        Chiến lược đang chạy: {display_name}")
    print("=================================================================\n")

    # Khởi tạo Embedder: ưu tiên Gemini nếu có key, rồi OpenAI, cuối cùng mock.
    embedder: Any = _mock_embed
    backend_name = "MockEmbedder"
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    try:
        if provider == "gemini" or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            embedder = CachedGeminiEmbedder()
            backend_name = getattr(embedder, "_backend_name", "gemini") + " (có cache disk)"
        elif provider == "openai" or os.getenv("OPENAI_API_KEY"):
            embedder = CachedOpenAIEmbedder()
            backend_name = "OpenAI text-embedding-3-small (có cache disk)"
    except Exception as error:
        print(f"[*] Fallback Embedder ({error})")
        embedder = _mock_embed
        backend_name = "MockEmbedder"
    print(f"[*] Embedding Backend: {backend_name}")

    # Nạp và chunk tài liệu
    docs = load_and_chunk_corpus(corpus_dir, chunker)
    avg_chunk_length = sum(len(doc.content) for doc in docs) / len(docs) if docs else 0.0
    max_chunk_length = max((len(doc.content) for doc in docs), default=0)
    print(f"[*] Tổng số tài liệu: {len(list(corpus_dir.glob('*.md')))} file .md")
    print(f"[*] Tổng số chunks tạo ra: {len(docs)} chunks")
    print(f"[*] Độ dài chunk: trung bình={avg_chunk_length:.1f}, lớn nhất={max_chunk_length} ký tự")

    # Pre-fill in batches: hundreds of chunks become a handful of API calls,
    # while EmbeddingStore can keep its simple one-text callback interface.
    if hasattr(embedder, "embed_many"):
        query_texts = [item["query"] for item in BENCHMARK_QUERIES]
        embedder.embed_many([doc.content for doc in docs] + query_texts)

    # Nạp vào EmbeddingStore
    store = EmbeddingStore(collection_name=f"bench_{selected_strategy}", embedding_fn=embedder)
    store.add_documents(docs)
    if hasattr(embedder, "_save"):
        embedder._save()
    print(f"[*] Đã nạp thành công {store.get_collection_size()} chunks vào EmbeddingStore\n")

    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_answer_fn)

    output_lines: list[str] = []
    output_lines.append(f"=== KẾT QUẢ ĐO LƯỜNG BENCHMARK ({display_name}) ===")
    output_lines.append(f"Tổng số tài liệu: {len(list(corpus_dir.glob('*.md')))} file .md | Tổng số chunks: {len(docs)}")
    output_lines.append(
        f"Độ dài chunk: trung bình={avg_chunk_length:.1f} ký tự | lớn nhất={max_chunk_length} ký tự"
    )
    output_lines.append(f"Mô hình embedding: {backend_name}\n")

    total_points = 0
    hit_at_1_count = 0
    hit_at_3_count = 0
    reciprocal_rank_sum = 0.0

    for b in BENCHMARK_QUERIES:
        qid = b["id"]
        qtext = b["query"]
        gold_ans = b["gold_answer"]
        gold_doc = b["gold_doc"]
        key_fact = b["key_fact"]
        answer_facts = b["answer_facts"]
        meta_filter = b["filter"]

        header = f"--- [Câu hỏi {qid}] {qtext} ---"
        print(header)
        output_lines.append(header)
        output_lines.append(f"Ghi chú: {b['note']}")

        if meta_filter:
            output_lines.append(f"Áp dụng Metadata Filter: {meta_filter}")
            results = store.search_with_filter(qtext, top_k=3, metadata_filter=meta_filter)
        else:
            output_lines.append("Áp dụng Metadata Filter: Không (None)")
            results = store.search(qtext, top_k=3)

        gold_rank = None

        for rank, r in enumerate(results, start=1):
            c_doc_id = r["metadata"].get("doc_id")
            c_score = r["score"]
            c_text = r["content"].replace("\n", " ")
            is_gold = c_doc_id == gold_doc and key_fact.lower() in c_text.lower()

            if is_gold and gold_rank is None:
                gold_rank = rank

            line = f"  Top-{rank} [Score: {c_score:.4f}] ({c_doc_id}): {c_text[:110]}..."
            print(line)
            output_lines.append(line)

        # Agent trả lời
        agent_answer = agent.answer(qtext, top_k=3, metadata_filter=meta_filter)
        print(f"  => Agent Answer: {agent_answer}\n")
        output_lines.append(f"  => Gold Answer : {gold_ans}")
        output_lines.append(f"  => Agent Answer: {agent_answer}")

        normalized_answer = " ".join(agent_answer.lower().split())
        answer_is_correct = all(fact.lower() in normalized_answer for fact in answer_facts)
        answer_has_citation = bool(re.search(r"\[\d+\]", agent_answer))

        if gold_rank == 1:
            hit_at_1_count += 1
        if gold_rank is not None:
            hit_at_3_count += 1
            reciprocal_rank_sum += 1.0 / gold_rank

        if gold_rank == 1 and answer_is_correct and answer_has_citation:
            query_points = 2
            status = "2/2 - Chunk đúng ở Top-1, Agent trả lời đủ key facts và có trích dẫn"
        elif gold_rank is not None:
            query_points = 1
            status = "1/2 - Chunk đúng ở Top-2/3 hoặc Agent trả lời/trích dẫn chưa đầy đủ"
        else:
            query_points = 0
            status = "0/2 - Không có chunk đúng chứa key fact trong Top-3"
        total_points += query_points
        output_lines.append(
            f"  => Gold rank: {gold_rank or 'không có trong Top-3'} | "
            f"Agent facts: {'đủ' if answer_is_correct else 'thiếu'} | "
            f"Citation: {'có' if answer_has_citation else 'không'}"
        )
        output_lines.append(f"  => Đánh giá: {status}\n")

    # Thực hiện kiểm tra A/B cho Câu 2 (So sánh Có lọc vs Không lọc)
    output_lines.append("=================================================================")
    output_lines.append("               A/B TESTING METADATA FILTER (CÂU 2)               ")
    output_lines.append("=================================================================")
    ab_query = "Thời hạn xử lý và phản hồi khiếu nại yêu cầu trả hàng hoàn tiền của Người Bán là bao lâu?"

    res_no_filter = store.search(ab_query, top_k=3)
    res_with_filter = store.search_with_filter(ab_query, top_k=3, metadata_filter={"audience": "seller"})

    output_lines.append(f"Câu hỏi: {ab_query}\n")
    output_lines.append("[LẦN 1 - KHÔNG LỌC METADATA (Unfiltered)]:")
    for i, r in enumerate(res_no_filter, 1):
        output_lines.append(
            f"  Top-{i} [Audience: {r['metadata'].get('audience')} | Score: {r['score']:.4f}]: {r['content'][:110].replace(chr(10), ' ')}..."
        )

    output_lines.append("\n[LẦN 2 - CÓ LỌC METADATA {'audience': 'seller'} (Filtered)]:")
    for i, r in enumerate(res_with_filter, 1):
        output_lines.append(
            f"  Top-{i} [Audience: {r['metadata'].get('audience')} | Score: {r['score']:.4f}]: {r['content'][:110].replace(chr(10), ' ')}..."
        )

    output_lines.append("\n=> KẾT LUẬN A/B TEST:")
    output_lines.append(
        "- Khi không lọc: Kết quả có nguy cơ lẫn lộn với các văn bản của Người mua (thời hạn 15 ngày), làm nhiễu thông tin phản hồi của người bán."
    )
    output_lines.append(
        "- Khi có lọc {'audience': 'seller'}: Loại bỏ 100% tài liệu người mua, tập trung chính xác vào quy định phản hồi trong vòng 02 ngày lịch của Người bán."
    )

    query_count = len(BENCHMARK_QUERIES)
    hit_at_1 = hit_at_1_count / query_count if query_count else 0.0
    hit_at_3 = hit_at_3_count / query_count if query_count else 0.0
    mrr = reciprocal_rank_sum / query_count if query_count else 0.0
    output_lines.append("\n=== TỔNG KẾT ĐÁNH GIÁ ===")
    output_lines.append(f"Điểm Retrieval Quality: {total_points}/10")
    output_lines.append(f"Hit@1: {hit_at_1_count}/{query_count} = {hit_at_1:.2%}")
    output_lines.append(f"Hit@3: {hit_at_3_count}/{query_count} = {hit_at_3:.2%}")
    output_lines.append(f"MRR: {mrr:.4f}")

    report_content = "\n".join(output_lines)
    Path("ket_qua_benchmark.txt").write_text(report_content, encoding="utf-8")
    print(f"\n[✓] Đã xuất kết quả đo lường ra file ket_qua_benchmark.txt thành công!")


if __name__ == "__main__":
    main()
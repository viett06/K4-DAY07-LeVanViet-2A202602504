from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy ngữ cảnh trong cơ sở tri thức."

        if metadata_filter:
            results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy ngữ cảnh liên quan cho câu hỏi này."

        context_blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = metadata.get("source") or metadata.get("doc_id") or result.get("id", "unknown")
            context_blocks.append(f"[{index}] (nguồn: {source})\n{result['content']}")
        context = "\n\n".join(context_blocks)

        prompt = (
            "Bạn là trợ lý chỉ được trả lời dựa trên ngữ cảnh được cung cấp. "
            "Không suy đoán thông tin ngoài ngữ cảnh. "
            "Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ là không tìm thấy. "
            "Khi trả lời, trích dẫn số nguồn [1], [2], ... tương ứng với chunk đã dùng.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )
        return self.llm_fn(prompt)

    def answer_with_filter(self, question: str, metadata_filter: dict | None = None, top_k: int = 3) -> str:
        return self.answer(question, top_k=top_k, metadata_filter=metadata_filter)

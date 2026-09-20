from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Split AFTER the delimiter so ". ", "! ", "? ", ".\n" stay on the
        # preceding sentence. A character-class split like r"[.!?]\s+" would
        # swallow the punctuation and produce truncated sentences.
        sentences = re.split(r"(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        size = self.max_sentences_per_chunk
        for start in range(0, len(sentences), size):
            group = sentences[start : start + size]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Hard-cut when no separators remain (covers separators=[]).
        if not remaining_separators:
            size = max(1, self.chunk_size)
            return [current_text[i : i + size] for i in range(0, len(current_text), size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        raw_parts = list(current_text) if separator == "" else current_text.split(separator)

        chunks: list[str] = []
        small_parts: list[str] = []
        for part in raw_parts:
            if not part:
                continue
            if len(part) <= self.chunk_size:
                small_parts.append(part)
            else:
                if small_parts:
                    chunks.extend(self._merge(small_parts, separator))
                    small_parts = []
                chunks.extend(self._split(part, next_separators))

        if small_parts:
            chunks.extend(self._merge(small_parts, separator))
        return chunks

    def _merge(self, parts: list[str], separator: str) -> list[str]:
        """Join adjacent small pieces until they would exceed chunk_size."""
        if not parts:
            return []

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else f"{current}{separator}{part}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = part
            if len(current) > self.chunk_size:
                chunks.extend(self._split(current, []))
                current = ""
        if current:
            chunks.append(current)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = min(50, max(0, chunk_size // 2))
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result: dict = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = (sum(len(chunk) for chunk in chunks) / count) if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result

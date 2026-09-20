# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Văn Việt
**MSSV:** 2A202602504
**Nhóm:** [điền tên nhóm khi nộp chung REPORT_NHOM]
**Ngày:** 2026-09-20

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector gần cùng hướng trong không gian embedding, tức hai đoạn văn bản được mô hình coi là gần nghĩa với nhau — không nhất thiết phải dùng chung từ vựng.

**Ví dụ có độ tương tự CAO:**
- Câu A: Người mua có thể đổi trả hàng trong 7 ngày kể từ khi nhận.
- Câu B: Khách hàng được hoàn tiền nếu gửi lại sản phẩm trong một tuần.
- Tại sao tương đồng: Khác từ vựng (đổi trả / hoàn tiền, 7 ngày / một tuần, người mua / khách hàng) nhưng cùng nói về quyền hoàn hàng trong thời hạn ngắn. Cặp này chứng minh embedding cần hiểu nghĩa, không so khớp từ.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Người mua có thể đổi trả hàng trong 7 ngày kể từ khi nhận.
- Câu B: Python thường chậm hơn ngôn ngữ biên dịch trên tác vụ CPU nặng.
- Tại sao khác: Một câu nói chính sách TMĐT, một câu nói hiệu năng ngôn ngữ lập trình — không cùng chủ đề, không cùng thực thể.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine đo góc giữa hai vector nên không bị chi phối bởi độ dài (số chiều / magnitude). Embedding hay được L2-normalize (`||v|| = 1`), lúc đó cosine trùng với dot product — đúng cách `EmbeddingStore.search` đang xếp hạng. Euclid vẫn đổi khi hai vector cùng hướng nhưng khác độ lớn, nên kém ổn định cho so khớp nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `ceil((độ_dài − overlap) / (chunk_size − overlap))`
> `ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.111…) = 23`
> Kiểm lại bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` → **23 chunks**.
> *Đáp án:* 23

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> `ceil((10000 − 100) / (500 − 100)) = ceil(9900 / 400) = ceil(24.75) = 25` (đúng với `FixedSizeChunker`). Overlap tăng làm bước trượt `chunk_size − overlap` nhỏ hơn, nên cần nhiều cửa sổ hơn để phủ hết tài liệu. Muốn overlap lớn hơn khi thông tin quan trọng (số liệu, điều kiện, tiêu đề mục) hay nằm sát ranh giới chunk: phần chồng giúp chunk sau vẫn còn ngữ cảnh của chunk trước, dù tốn thêm chỗ lưu và thời gian embed.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tách câu bằng lookbehind `r"(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)"` để cắt *sau* dấu câu, giữ nguyên `.` `!` `?`. Không dùng `[.!?]\s+` vì regex đó nuốt mất dấu câu. Sau đó `strip`, gom `max_sentences_per_chunk` câu, nối bằng một khoảng trắng. Text rỗng / chỉ whitespace trả `[]`. Edge case chưa xử lý: viết tắt (`TS.`, `v.v.`) và số thập phân (`3.14`) vẫn bị cắt nhầm vì nhìn `. ` / `.` như hết câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Hai chiều: (1) đệ quy xuống separator còn lại khi mảnh vẫn dài hơn `chunk_size`; (2) gom các mảnh nhỏ liền kề lên sát `chunk_size` bằng `_merge` (nối lại đúng separator đang dùng). Ba base case: text rỗng → `[]`; `len(text) <= chunk_size` → `[text]`; `remaining_separators == []` → cắt cứng theo `chunk_size` (đúng test `test_empty_separators_falls_back_gracefully`). Khi một mảnh quá dài, flush nhóm nhỏ hiện tại rồi mới đệ quy — không trộn recursive-result với separator của tầng trên.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Chỉ dùng in-memory: `_use_chroma = False` dù máy có thể có `chromadb`. `_make_record` copy metadata (tránh caller sửa làm bẩn store) và `setdefault("doc_id", doc.id)`. Vector đã chuẩn hoá nên search dùng `_dot` (cosine). `search` và `search_with_filter` cùng gọi `_search_records` trên tập ứng viên khác nhau; kết quả bỏ `embedding` để in ra terminal không bị vector 64/1536 chiều.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc metadata **trước**, rồi mới similarity. Nếu lấy top-k rồi mới lọc, k slot có thể bị tài liệu sai audience chiếm hết và còn 0 kết quả dù store vẫn còn chunk hợp lệ. `delete_document` xoá mọi record `metadata["doc_id"] == doc_id` (mọi chunk của cùng file), trả `True` khi có xoá, `False` khi không thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Store rỗng hoặc không có kết quả → trả thông báo, không gọi LLM. Ngược lại retrieve top-k (tuỳ chọn `metadata_filter`), đánh số `[1] [2] [3]` kèm nguồn (`source` / `doc_id`), rồi nhúng vào prompt: chỉ dùng ngữ cảnh, không bịa, phải trích dẫn số chunk. `llm_fn` nhận prompt hoàn chỉnh — test inject lambda, `main.py` dùng demo LLM, `bench.py` dùng hàm trích ngữ cảnh.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/levanviet/Documents/Workspace/th_lab_vin/K4-DAY07-LeVanViet-2A202602504
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.04s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

`python main.py "Chunking là gì?"` chạy hết pipeline (load file → store → search → agent). Dòng `Skipping missing file: data/customer_support_playbook.txt` là bình thường. Backend: `mock embeddings fallback`.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Embed từng câu bằng `MockEmbedder`, rồi gọi `compute_similarity()` trên hai vector. Dự đoán theo **nghĩa**; điểm thực tế là cosine của vector giả lập (MD5 → số giả ngẫu nhiên, đã L2-normalize).

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người mua có thể đổi trả hàng trong 7 ngày kể từ khi nhận. | Khách hàng được hoàn tiền nếu gửi lại sản phẩm trong một tuần. | cao | 0.1763 | Không |
| 2 | Vector store lưu embedding và tìm các mục gần nhất với câu hỏi. | Kho vector giữ các vector số và truy xuất tài liệu tương tự nhất. | cao | -0.1128 | Không |
| 3 | Người mua có thể đổi trả hàng trong 7 ngày kể từ khi nhận. | Python thường chậm hơn ngôn ngữ biên dịch trên tác vụ CPU nặng. | thấp | -0.0193 | Có (thấp) |
| 4 | Chia nhỏ theo câu giữ ranh giới ngôn ngữ tự nhiên. | Brown bears live in forests across the northern hemisphere. | thấp | 0.1413 | Không (điểm cao hơn cặp 1) |
| 5 | Chunk quá nhỏ làm mất ngữ cảnh. | Nếu các đoạn quá ngắn, kết quả truy xuất thiếu thông tin. | cao | -0.1984 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 5 cùng nói “chunk ngắn → mất thông tin” nhưng cosine **âm** (−0.198), trong khi cặp 4 gần như không liên quan lại được 0.141 — cao hơn cả cặp 1 (cùng nghĩa chính sách). `MockEmbedder` băm MD5 ký tự rồi sinh vector, **không mã hoá ngữ nghĩa**. Cosine lúc này phản ánh nhiễu hash, không phản ánh nghĩa. Muốn số liệu benchmark có ý nghĩa phải bật embedder thật (`local` / `openai` / `gemini`). Hàm `compute_similarity` vẫn đúng: vector giống nhau → 1.0, vuông góc → 0.0, ngược chiều → −1.0, vector 0 → 0.0 (đã kiểm bằng unit test).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chiến lược cá nhân: **`FixedSizeChunker(chunk_size=500, overlap=50)`**. Công cụ: `bench.py`. Corpus: `data/ecommerce/` (5 chính sách Shopee), **290 chunks**. Embedding: **`gemini-embedding-001`**. Output đầy đủ: `ket_qua_benchmark.txt`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sau khi giao thành công, người mua còn bao lâu để gửi yêu cầu trả hàng hoàn tiền? | `shopee-return-refund-buyer#7` — 24 giờ với thực phẩm tươi/đông lạnh; overlap cắt mất cụm “15 ngày” | 0.812 | Đúng file, có 24 giờ; chuỗi “15 (mười lăm) ngày” bị cắt sang chunk `#6` | Nêu 24 giờ cho thực phẩm; 15 ngày nằm chunk kề |
| 2 | Sản phẩm được bảo hành miễn phí khi nào? | `shopee-warranty-buyer#5` — trường hợp **không** được BH (tự sửa, lỗi người dùng) | 0.797 | Đúng file, **sai section** (điều kiện miễn phí ở `#4`) | Nêu trường hợp loại trừ, chưa đủ 4 điều kiện miễn phí |
| 3 | Ảnh thật sản phẩm phải chiếm tối thiểu bao nhiêu diện tích tấm ảnh? | `shopee-seller-listing-rules#18` — ngành thời trang 40–50%; gold 40% ở `#11` (top-2) | 0.792 | Có liên quan; đáp án 40% ở top-2 | Nêu 40–50% (thời trang); top-2 có đúng 40% ảnh thật |
| 4 | Hàng hóa khi giao đi phải còn tối thiểu bao nhiêu thời hạn sử dụng? | `shopee-seller-listing-rules#41` — còn ít nhất 30% HSD và 30 ngày | 0.812 | Có, top-1 chứa đáp án | Đúng: ≥ 30% thời hạn sử dụng và ≥ 30 ngày |
| 5 | Mức bồi thường cho mỗi đơn hàng vi phạm? (`metadata_filter={"audience":"seller"}`) | `shopee-antifraud-seller-penalty#10` — bồi thường đến 10.000.000 VND / đơn | 0.771 | Có, top-1 chứa đáp án | Đúng: lên đến 10.000.000 VND mỗi đơn vi phạm |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 đúng `doc_id`; 4 / 5 ngữ cảnh chứa đáp án (Q2 thiếu chuỗi “lỗi kỹ thuật do nhà sản xuất”). Điểm theo `docs/SCORING.md`: **7 / 10**.

**A/B metadata filter (Q5):**
- Có filter `audience=seller`: top-3 toàn `shopee-antifraud-seller-penalty`, top-1 có 10.000.000 VND.
- Không filter: cùng top-3 antifraud (score 0.771/0.743/0.732) — lần này Gemini đã tách được file đúng cả khi không lọc.
- Filter vẫn đúng yêu cầu lab (câu hỏi không gắn sẵn “người bán bồi thường gian lận”), và thu hẹp tập ứng viên về `audience=seller`.

**Failure case:** Q2 lấy đúng `shopee-warranty-buyer` nhưng top-1 là mục loại trừ (tự sửa chữa / lỗi người dùng), còn “lỗi kỹ thuật do nhà sản xuất” nằm chunk `#4` ngay trước đó. FixedSize cắt giữa mục 1 nên overlap 50 ký tự chưa kéo đủ điều kiện miễn phí vào cùng cửa sổ. Hướng sửa: tăng overlap, hoặc chunk theo heading của điều khoản.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Chưa tới lượt demo nhóm. So với lần chạy mock (3/10), Gemini nâng lên 7/10: ranking theo nghĩa, Q4/Q5 top-1 đúng số liệu. Vẫn còn lỗi cắt giữa mục (Q1, Q2). Sẽ bổ sung bài học từ thành viên khác sau demo.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **58 / 60** |

Ghi chú tự đánh giá mục 5: đã chạy FixedSize + Gemini trên corpus Shopee; retrieval thô **7/10**, có A/B filter và failure case.

# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** K4-L3B — The LIEM'S

**Thành viên đã xác định trong mã:** Ngô Anh Khoa (HeadingChunker), Mai Quang Dũng (RecursiveChunker), Lê Văn Việt(FixedSizeChunker), Đặng Đỉnh Đoàn(SentenceChunker)

**Ngày:** 20/09/2026

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

**Chủ đề:** Chính sách và quy trình thương mại điện tử Shopee cho Người Mua/Người Bán.

Nhóm chọn chủ đề này vì tài liệu công khai có cấu trúc Markdown rõ, chứa nhiều quy tắc, thời hạn và con số có thể kiểm chứng. Corpus cũng có các nội dung gần nhau giữa buyer/seller, phù hợp để đánh giá metadata filtering và lỗi retrieval do nhiễu ngữ nghĩa.

### Danh sách tài liệu

| # | Tài liệu | Nguồn | Phiên bản | Ký tự | Metadata chính |
|---:|---|---|---|---:|---|
| 1 | Chống gian lận người bán | `help.shopee.vn/portal/4/article/140097` | Hiệu lực 28/12/2023 | 6.586 | `seller`, `antifraud` |
| 2 | Quy trình mua hàng | `help.shopee.vn/portal/4/article/77245` | Cập nhật 03/01/2025 | 2.434 | `buyer`, `marketplace` |
| 3 | Quy chế sàn Shopee | `help.shopee.vn/portal/4/article/77245` | Cập nhật 03/01/2025 | 76.117 | `both`, `marketplace` |
| 4 | Quy trình đăng bán | `help.shopee.vn/portal/4/article/77245` | Cập nhật 03/01/2025 | 1.840 | `seller`, `marketplace` |
| 5 | Trả hàng/hoàn tiền — buyer | `help.shopee.vn/portal/4/article/77251` | Hiệu lực 11/03/2026 | 17.542 | `buyer`, `return-refund` |
| 6 | Trả hàng/hoàn tiền — seller | `help.shopee.vn/portal/4/article/77251` | Hiệu lực 11/03/2026 | 3.340 | `seller`, `return-refund` |
| 7 | Quy định đăng bán | `help.shopee.vn/portal/4/article/77246` | Công bố 14/08/2024 | 21.785 | `seller`, `listing` |
| 8 | Chính sách bảo hành | `help.shopee.vn/portal/4/article/79046` | Không nêu | 3.742 | `buyer`, `warranty` |

Nguồn URL đầy đủ và thông tin license/permission nằm trong `data/ecommerce/sources.csv`.

### Data governance

- [x] Corpus chỉ chứa nguồn trợ giúp công khai của Shopee, không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Cả 8 tài liệu có `source_url`, `retrieved_at`, `document_version` và `license_or_permission` trong metadata/manifest.
- [x] File `.env` và embedding cache không được Git theo dõi.

### Metadata schema

| Trường | Kiểu | Ví dụ | Công dụng |
|---|---|---|---|
| `doc_id` | string | `shopee-return-refund-seller` | Định danh, đánh giá gold doc và xóa tài liệu |
| `title` | string | `Chính sách trả hàng...` | Hiển thị nguồn dễ hiểu |
| `audience` | enum | `buyer`, `seller`, `both` | Lọc quy định đúng đối tượng trước retrieval |
| `category` | string | `return-refund` | Thu hẹp chủ đề |
| `source_url` | URL | URL Shopee Help | Truy nguyên/citation |
| `retrieved_at` | date | `2026-09-20` | Quản trị độ mới |
| `document_version` | string | `hiệu lực 11/03/2026` | Kiểm soát phiên bản chính sách |

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích baseline trên ba tài liệu

`ChunkingStrategyComparator.compare(..., chunk_size=400)`:

| Tài liệu | Strategy | Chunk | Trung bình | Nhận xét ngữ cảnh |
|---|---|---:|---:|---|
| Return/refund buyer | Fixed | 50 | 393,8 | Có thể cắt giữa câu |
| Return/refund buyer | Sentence | 44 | 389,6 | Giữ câu tốt |
| Return/refund buyer | Recursive | 57 | 299,8 | Cân bằng biên và kích thước |
| Antifraud | Fixed | 18 | 393,7 | Có thể mất heading |
| Antifraud | Sentence | 9 | 690,7 | Mệnh đề nguyên vẹn nhưng quá dài |
| Antifraud | Recursive | 21 | 294,1 | Kích thước ổn định |
| Warranty | Fixed | 10 | 379,1 | Có thể cắt danh sách |
| Warranty | Sentence | 5 | 666,6 | Giữ cụm điều kiện nhưng chunk dài |
| Warranty | Recursive | 10 | 330,1 | Giữ tương đối tốt cấu trúc |

### Chiến lược của thành viên

**Ngô Anh Khoa — HeadingChunker:** Tách theo section Markdown, lưu hierarchy heading và lặp context heading trên các child chunk. Cách này phù hợp tài liệu chính sách có cấu trúc mục, giúp Top-1 dễ hiểu và có nguồn ngữ cảnh ngay trong chunk.

```python
heading_context = "\n".join(item[1] for item in heading_stack)
body_budget = max_size - len(heading_context) - 1
for child in RecursiveChunker(chunk_size=body_budget).chunk(body):
    chunks.append(f"{heading_context}\n{child}")
```

**Mai Quang Dũng — RecursiveChunker:** Chia đệ quy theo đoạn, dòng, câu, từ rồi cắt cứng; sau đó ghép các mảnh nhỏ tới giới hạn 400 ký tự. Chiến lược giữ kích thước ổn định mà vẫn ưu tiên biên tự nhiên.

**Lê Văn Việt — FixedSizeChunker:** Cắt cửa sổ cố định `chunk_size=400`, `overlap=50`. Overlap giữ lại phần cuối chunk trước để số liệu/điều kiện sát ranh giới không bị mất hết. Cách này đơn giản, kích thước đều, phù hợp để đối chiếu công bằng với các chiến lược còn lại trên cùng corpus.

```python
CHUNKER = FixedSizeChunker(chunk_size=400, overlap=50)
```

**Đặng Đỉnh Đoàn — SentenceChunker:** Tách theo ranh giới câu rồi gom `max_sentences_per_chunk=3` câu thành một chunk. Giữ nguyên câu chứa key fact (thời hạn, mức phạt, điều kiện bảo hành), tránh cắt giữa mệnh đề như fixed-size.

```python
CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
```

Hai baseline Fixed-size và Sentence cũng được chạy trên cùng corpus để có đối chứng công bằng.

### So sánh toàn corpus

| Strategy | Chunks | TB / max | Điểm /10 | Hit@1 | Hit@3 | MRR | Điểm mạnh | Điểm yếu |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Fixed-size | 374 | 396,9 / 400 | 6 | 40% | 80% | 0,5667 | Đơn giản, kích thước đều | Cắt giữa mệnh đề/danh sách |
| Sentence | 337 | 383,8 / 1.748 | **9** | 80% | **100%** | **0,9000** | Giữ nguyên câu chứa key fact | Không bảo đảm trần ký tự |
| Recursive | 416 | 308,7 / 400 | 7 | 60% | 80% | 0,6667 | Biên tự nhiên, size ổn định | Có thể tách điều kiện liên quan |
| Heading | 645 | 327,4 / 400 | 7 | **80%** | 80% | 0,8000 | Giữ hierarchy, Top-1 tốt | Heading lặp làm chunk cùng section cạnh tranh |

SentenceChunker tốt nhất trên bộ 5 câu hiện tại vì các key fact chủ yếu nằm trong một câu/mệnh đề hoàn chỉnh. Tuy nhiên, nó tạo chunk tới 1.748 ký tự; nếu corpus lớn hoặc giới hạn context chặt, phương án lai Heading + Sentence/Recursive và điều chỉnh kích thước sẽ bền vững hơn.

## 3. Câu hỏi đánh giá & Chất lượng truy xuất — Nhóm (10 điểm)

| # | Câu hỏi | Gold answer | Gold document / key fact |
|---:|---|---|---|
| 1 | Người mua được yêu cầu trả hàng trong bao lâu? | 15 ngày; hàng tươi sống/đông lạnh 24 giờ | `return-refund-buyer` / `15 (mười lăm) ngày` |
| 2 | Người bán phải phản hồi khiếu nại trong bao lâu? | 02 ngày lịch từ thông báo Shopee | `return-refund-seller` / `02 ngày lịch` |
| 3 | Bồi thường đơn hàng ảo bao nhiêu? | Tối đa 10.000.000 VND/đơn vi phạm | `antifraud-seller-penalty` / `10.000.000` |
| 4 | Hạn sử dụng khi giao hàng phải còn bao nhiêu? | Ít nhất 30% và ít nhất 30 ngày | `seller-listing-rules` / `30%` |
| 5 | Điều kiện bảo hành miễn phí? | Lỗi kỹ thuật; còn hạn; hóa đơn/mã đơn; tem bảo hành nguyên vẹn | `warranty-buyer` / `lỗi kỹ thuật` |

### Tổng hợp theo câu

| # | Strategy tốt nhất | Gold trong Top-3? | Ghi chú |
|---:|---|---|---|
| 1 | Sentence/Recursive/Heading | Có, Top-1 | Cả ba giữ trọn thời hạn 15 ngày và ngoại lệ 24 giờ |
| 2 | Sentence/Recursive/Heading | Có, Top-1 | Dùng filter `audience=seller` |
| 3 | Fixed-size | Có, Top-1 | Heading trả về đúng tài liệu nhưng sai đoạn; Sentence/Recursive ở hạng thấp hơn |
| 4 | Sentence/Heading | Có, Top-1 | Heading đạt cosine cao nhất 0,8179 |
| 5 | Sentence | Có, Top-1 | Agent Sentence nêu đủ cả tem bảo hành; Heading thiếu một điều kiện |

Metadata filter giúp rõ nhất ở câu 2: không lọc, Top-3 lẫn chunk `buyer`/`both` và quy định 15 ngày; lọc `audience=seller` loại hoàn toàn tài liệu buyer khỏi candidate set và đưa chunk 02 ngày lên Top-1. Câu 3 và 5 cũng lọc theo `seller`/`buyer` để giảm nhiễu.

## 4. Demo & Bài học nhóm — Nhóm (5 điểm)

Các insight dùng cho demo:

- Điểm cosine cao với đúng `doc_id` chưa đủ: câu 3 của Heading lấy đúng tài liệu nhưng không chứa key fact 10.000.000 VND, nên phải chấm 0.
- Metadata phải được áp dụng cho cả retrieval và context Agent; nếu chỉ lọc phần hiển thị thì câu trả lời vẫn có thể dùng tài liệu sai audience.
- Sentence đạt retrieval tốt nhất nhưng vi phạm mục tiêu 400 ký tự; số điểm không phải tiêu chí duy nhất khi chọn chiến lược vận hành.

Cùng dữ liệu và embedding, chỉ thay biên chunk đã làm điểm dao động từ 6 đến 9, Hit@1 từ 40% đến 80%. Nếu làm lại, nhóm sẽ thử hybrid Heading + sentence-aware packing, tăng `top_k` có kiểm soát cho section dài và mở rộng bộ câu hỏi để tránh tối ưu quá mức trên 5 query.

## Tự Đánh Giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Lựa chọn tài liệu | 10/10 |
| Thiết kế chiến lược | 15/15 |
| Chất lượng truy xuất | 9/10 |
| Chuẩn bị demo | 5/5 |
| **Tổng phần nhóm** | **39/40** |
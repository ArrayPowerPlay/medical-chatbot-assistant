# Kế Hoạch Triển Khai Metric Retrieval Theo Hướng Paper

## 1. Mục tiêu

Nâng `scripts/evaluate_retrieval.py` từ công cụ debug nội bộ thành evaluator
phù hợp hơn cho báo cáo/paper, nhưng **không làm xáo trộn retrieval pipeline hiện tại**.

Quyết định chốt:

- **Document retrieval**: bổ sung bộ metric `official-like` gần với BioASQ hơn.
- **Snippet retrieval**: **giữ cách chấm theo parent như hiện tại**, không chuyển sang
  child-level snippet retrieval, không sinh sentence windows, không thêm một pipeline
  snippet riêng.
- **Custom metrics hiện tại vẫn giữ lại** để debug và so sánh ablation.

Nói ngắn gọn:

- `documents`: báo `custom + official-like`
- `snippets`: báo `custom/proxy`, không claim là official snippet metric

---

## 2. Lý do chốt hướng này

### 2.1. Vì sao không chuyển snippet từ parent sang child

Hiện tại child chunk trong hệ thống là **đơn vị retrieval**, không phải **đơn vị snippet**.

Các vấn đề của child chunk hiện tại:

- child chunk dài khoảng `500` ký tự, vẫn khá lớn so với gold snippet
- child chunk có **title injection** kiểu `Title: ...` / `Content: ...`, nên text
  không còn là span sạch của abstract
- Weaviate hiện chỉ lưu:
  - `parent_id`
  - `pmid`
  - `text`
  và **không lưu offset/span metadata**
- nếu ép child chunk thành predicted snippet thì metric snippet sẽ phản ánh
  `retrieval unit size` nhiều hơn là chất lượng evidence span thực sự

Vì vậy, nếu đổi thẳng sang child-level snippet scoring:

- precision dễ giảm mạnh
- overlap/offset khó giải thích
- phải thiết kế lại schema chunk/index khá nhiều
- paper dễ bị reviewer hỏi tại sao “snippet” lại là một đoạn retrieval 500 ký tự

### 2.2. Vì sao không chuyển sang sentence-window snippet generation

Hướng `top-doc -> tách abstract thành câu/window -> rerank snippet candidates`
về mặt ý tưởng là hợp lý hơn child chunk, nhưng trong repo này vẫn có nhược điểm:

- phải chọn cơ chế tách câu/sentence window
- số candidate tăng nhanh
- thêm một tầng ranking mới, làm evaluator phức tạp hơn nhiều
- dễ tạo thêm một nguồn nhiễu mới trước khi document metrics được chốt ổn định

Ở giai đoạn hiện tại, hướng đó là quá nặng so với nhu cầu paper baseline.

### 2.3. Vì sao vẫn giữ snippet metric

Snippet metric vẫn có ích vì:

- cho biết retrieved parent có chứa evidence hay không
- giúp phân tích chất lượng grounding tốt hơn document PMID matching thuần túy
- phù hợp cho theo dõi nội bộ và so sánh cấu hình

Nhưng cần mô tả trung thực:

- đây là **proxy snippet evaluation**
- không phải official BioASQ snippet overlap scoring

---

## 3. Kết luận thiết kế cần triển khai

### 3.1. Bộ metric cuối cùng

Evaluator sau khi hoàn thiện sẽ báo **2 nhóm metric**:

#### A. Custom diagnostic metrics

Giữ nguyên các metric hiện tại:

- `Precision@5, @10, @20`
- `Recall@5, @10, @20`
- `F1@5, @10, @20`
- `MAP@5, @10, @20`
- `MRR`
- `Snippet_Recall@5, @10, @20`
- `Snippet_Precision@5, @10, @20`
- `Snippet_F1@5, @10, @20`

#### B. Official-like document metrics

Tính trên **top-10 documents**:

- `Mean Precision`
- `Recall`
- `F-Measure`
- `MAP`
- `GMAP`

Lưu ý:

- phần `official-like` ở đây áp dụng cho **document retrieval**
- **không áp dụng** cho snippet theo nghĩa official BioASQ overlap metric

### 3.2. Snippet metric được giữ ở mức proxy

Giữ logic hiện tại:

- đơn vị chấm là **retrieved parent chunks**
- một gold snippet được tính là match nếu:
  - `snippet["pmid"] == retrieved_item["pmid"]`
  - và `snippet["text"] in retrieved_item["text"]`

Nghĩa là snippet metric tiếp tục đo:

- retrieved parent có chứa evidence text hay không

Đây là một **parent-based containment proxy**.

Trong paper cần ghi rõ:

- snippet metric không phải official BioASQ snippet metric
- đây là metric proxy để đánh giá evidence coverage trong parent retrieval

---

## 4. Các thay đổi cần làm

### 4.1. Giữ nguyên retrieval pipeline

Không thay đổi:

- QueryAnalyzer
- Vector search
- BM25
- RRF
- Cross-Encoder
- Parent-level ranking cho text retrieval

Không thêm:

- child-level snippet fusion path
- snippet-specific reranker
- sentence-window candidate generation
- re-index / re-embed toàn corpus chỉ để phục vụ snippet metric

### 4.2. Mở rộng `evaluate_retrieval.py`

Evaluator luôn tính bộ metric mở rộng khi chạy bình thường.

CLI giữ tối giản:

- `--limit`

Không có:

- `--metric-mode`
- `--official-top-k`
- `--question-id`

### 4.3. Tính document metrics theo schema paper hiện tại

Với mỗi question tiếp tục tính:

- `precision_at_k`
- `recall_at_k`
- `f1_at_k`
- `average_precision_at_k`
- `reciprocal_rank`

Rồi aggregate trong `summary.json` thành:

- `Precision@K`
- `Recall@K`
- `F1@K`
- `MAP@K`
- `GMAP@K`
- `MRR`

Trong đó:

- `K = [5, 10, 20]`
- `@10` là lát cắt gần official-like nhất

- `Mean Precision = mean(precision_i)`
- `Recall = mean(recall_i)`
- `F-Measure = mean(f_i)` hoặc nêu rõ nếu chọn cách macro-average theo câu hỏi
- `MAP = mean(ap_i)`
- `GMAP = exp(mean(log(ap_i + eps)))`

Khuyến nghị:

- dùng `eps = 1e-6`
- ghi rõ `eps` trong `summary.json`

### 4.4. Giữ snippet metrics như một bảng riêng

Snippet metrics tiếp tục được tính như hiện tại trên:

- `top 5`
- `top 10`
- `top 20`

Không đổi logic matching.

Chỉ cần đổi cách mô tả trong code/output/paper:

- tên hoặc chú thích phải nói rõ đây là `containment-based snippet proxy`

---

## 5. Dữ liệu đánh giá

### 5.1. Không bắt buộc enrich snippet offsets ở pha này

Vì đã chốt **không triển khai official snippet overlap metric**, nên ở pha này
không cần bắt buộc sửa dataset để thêm:

- `beginSection`
- `endSection`
- `offsetInBeginSection`
- `offsetInEndSection`

Nếu sau này muốn mở rộng sang official snippet evaluation thật, mới cần làm bước đó.

### 5.2. Điều cần giữ trong mô tả dataset

`preprocess_bioasq_taskB.py` hiện:

- lấy PMID từ `documents` và `snippets`
- fetch hoặc reuse article theo đúng PMID đó
- lưu `abstractText` từ PubMed XML
- chỉ giữ gold snippet ở dạng:
  - `text`
  - `pmid`

Điểm quan trọng cần mô tả trung thực:

- corpus article và gold snippet cùng trỏ về đúng PMID
- nhưng snippet metadata provenance/offset hiện đã bị lược bỏ trong file eval

---

## 6. Thiết kế output

### 6.1. `summary.json`

Đề xuất cấu trúc:

```json
{
  "config": {
    "retrieval_top_k": 80,
    "top_k_rrf": 80,
    "k_rrf": 60,
    "child_fetch_limit": 120,
    "rerank_text_top_m": 20,
    "k_values": [5, 10, 20],
    "official_top_k": 10
  },
  "custom_metrics": {
    "document_metrics": {},
    "snippet_metrics": {}
  },
  "official_like": {
    "documents": {
      "mean_precision": 0.0,
      "recall": 0.0,
      "f_measure": 0.0,
      "map": 0.0,
      "gmap": 0.0,
      "epsilon": 1e-6
    }
  }
}
```

### 6.2. `detail.jsonl`

Mỗi question nên có thêm:

- `official_doc_metrics`
  - `precision`
  - `recall`
  - `f_measure`
  - `ap`

Nhưng không cần thêm một nhánh `official_snippet_metrics`.

---

## 7. Cách viết trong paper

Nên chia thành 2 bảng:

### Bảng chính

`Official-like document retrieval metrics`

- Mean Precision
- Recall
- F-Measure
- MAP
- GMAP

Trên `top-10 documents`.

### Bảng phụ / bảng chẩn đoán

`Custom retrieval diagnostics`

- `P@5, P@10, P@20`
- `R@5, R@10, R@20`
- `MAP@5, MAP@10, MAP@20`
- `MRR`
- `Snippet_Recall/Precision/F1@K` theo parent containment

Cách diễn đạt nên là:

- document metrics dùng để so sánh gần hơn với chuẩn BioASQ
- snippet metrics dùng để phân tích evidence coverage trong parent retrieval

Không nên viết:

- “we reproduce official BioASQ snippet evaluation”

nếu chưa triển khai overlap metric thật với offsets.

---

## 8. Kế hoạch triển khai cụ thể

### Bước 1

Refactor `evaluate_retrieval.py` để tách:

- `custom document metrics`
- `custom snippet metrics`
- `official-like document metrics`

### Bước 2

Thêm hàm tính:

- per-question document precision/recall/f/ap trên top-10
- aggregate `MAP`
- aggregate `GMAP`

### Bước 3

Mở rộng `summary.json` và `detail.jsonl` để ghi official-like document metrics.

### Bước 4

Giữ nguyên snippet evaluator hiện tại, nhưng đổi comment/tên mô tả để nhấn mạnh:

- đây là `parent-text containment proxy`

### Bước 5

Chạy lại benchmark trên cùng tập câu hỏi cố định và báo song song:

- `official-like documents`
- `custom diagnostics`

---

## 9. Test plan

### Unit tests cần có

- document precision/recall/f/ap đúng với fixture nhỏ
- `GMAP` đúng với `eps` đã chọn
- question không hit relevant doc nào vẫn cho:
  - `ap = 0`
  - `gmap` ổn định số học
- `--metric-mode custom` giữ nguyên output cũ
- `--metric-mode official` chỉ sinh bảng official-like documents
- `--metric-mode both` sinh đủ cả hai

### Regression checks

- metric custom hiện tại không bị đổi ngầm
- snippet containment logic không thay đổi
- output JSON backward-compatible ở mức hợp lý

---

## 10. Giả định đã chốt

- `RERANK_TEXT_TOP_M` tiếp tục giữ `20`
- `K_VALUES` cho custom metrics tiếp tục là `[5, 10, 20]`
- `official_top_k` cho document metrics là `10`
- không re-embed corpus chỉ để đổi snippet metric
- không chuyển snippet metric sang child-level
- không triển khai official BioASQ snippet overlap metric trong pha này

Đây là phương án cân bằng nhất giữa:

- chất lượng học thuật của bảng document metrics
- độ ổn định của evaluator
- chi phí triển khai thực tế trong repo hiện tại

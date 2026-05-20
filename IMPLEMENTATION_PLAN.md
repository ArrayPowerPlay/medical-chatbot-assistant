# Evaluation Protocol

## 1. Mục tiêu

Chốt protocol đánh giá ngắn gọn cho project để:

- tách rõ `retrieval eval` và `generation eval`
- tránh trộn retrieval tuning với generation tuning
- giữ chi phí API ở mức chấp nhận được
- đủ rõ để implement evaluator mà không phải quyết định lại từ đầu

---

## 2. Benchmark chốt

### 2.1. Benchmark chính

- `BioASQ Task B`
  - `val_bioasq.jsonl` dùng cho tuning
  - `test_bioasq.jsonl` dùng cho final report

### 2.2. Benchmark phụ cho generation

- `MedAESQA`
  - dùng cho **generation evaluation**
  - **không** dùng làm benchmark retrieval chính
  - lý do: chỉ có `40` câu, quá nhỏ để tune hoặc kết luận retrieval riêng
  - vẫn phù hợp để kiểm tra answer generation với gold expert answer và gold PMID citations

- `PubMedQA`
  - chỉ là **oracle-context benchmark**
  - nếu dùng thì feed trực tiếp dataset-provided context
  - không chạy retrieval để tránh source-article leakage

### 2.3. Split cho `MedAESQA`

Quyết định chốt:

- `12` câu `validation`
- `28` câu `test`
- tỷ lệ tương đương khoảng `30/70`

Lý do:

- `val` đủ để tune prompt và generation settings
- `test` vẫn còn đủ lớn để report
- `20/20` sẽ làm test quá nhỏ một cách không cần thiết
- `8/32` thì val hơi quá ít cho prompt tuning

Khuyến nghị:

- chia **một lần cố định**
- ưu tiên stratify theo `question_frame.Type`, `Task`, `Body system` nếu làm được
- không đổi split giữa các lần chạy

---

## 3. Metric chốt

### 3.1. Retrieval trên BioASQ

Giữ bộ metric hiện tại:

- `Precision@K`
- `Recall@K`
- `F1@K`
- `MAP@K`
- `GMAP@K`
- `MRR`
- snippet proxy metrics giữ nguyên cho nội bộ

### 3.2. Generation trên BioASQ

Bộ metric rút gọn, đủ dùng và tiết kiệm API:

- `ROUGE-SU4`
  - metric lexical chính
  - giữ lại vì đây là metric mang tính BioASQ-specific nhất

- `RAGAS Context Precision`
  - đại diện cho chất lượng retriever theo góc nhìn usefulness / noise

- `RAGAS Context Recall`
  - đại diện cho mức độ đầy đủ evidence của retriever

- `RAGAS Faithfulness`
  - đại diện cho safety / no hallucination

- `RAGAS Answer Correctness`
  - đại diện cho độ đúng đắn của answer so với expert answer

- `RAGAS Answer Relevancy`
  - chỉ dùng cho `validation` / debug prompt
  - không phải headline metric chính trong bảng final

### 3.3. Nhiều `ideal_answer`

Quyết định chốt:

- với metric reference-based, tính điểm với từng reference
- lấy `average across references` ở cấp question
- sau đó macro-average toàn bộ dataset

Không dùng `max` làm số report chính.

Có thể lưu `max` trong file detail nếu cần debug.

---

## 4. Evaluator Model Chốt

### 4.1. RAGAS evaluator

Chốt dùng:

- `evaluator_llm = gpt-4o-mini`
- `evaluator_embeddings = text-embedding-3-small`
- `temperature = 0`

Lý do:

- đủ rẻ để chạy nhiều baseline / nhiều vòng validation
- vẫn bám theo family model phổ biến trong docs `RAGAS`
- phù hợp hơn với budget thực tế của project

Lưu ý:

- literature thường dùng judge mạnh hơn như `GPT-4` / `GPT-4o`
- vì vậy phải ghi rõ đây là cost-performance trade-off
- không thay evaluator model giữa các thí nghiệm final

### 4.2. Generator model

- generation model vẫn là model của system đang benchmark
- không trộn lẫn giữa `generator` và `evaluator`

---

## 5. Generation Tuning

Sau khi retrieval đã được khóa, `generation val` dùng để tune:

- prompt cuối cho `text-only`, `kg-only`, `full`
- số lượng text passages đưa vào prompt
- số lượng KG paths đưa vào prompt
- `text / KG ratio`
- có giữ `kg_merger` hay không
- có giữ `head-tail placement` hay không
- cách interleave `text <-> KG`
- answer style ngắn / dài
- citation style
- độ mạnh của policy `if insufficient context then abstain`
- `temperature` trong vùng thấp
- `max_tokens`

### Không tune ở pha này

- `VECTOR_TOP_K`
- `KEYWORD_TOP_K`
- `TOP_K_RRF`
- `RERANK_TEXT_TOP_M`
- `RERANK_KG_TOP_N`

Các biến trên là retrieval hyperparameters, không phải generation hyperparameters.

---

## 6. MedAESQA Protocol

### 6.1. Vai trò

`MedAESQA` được dùng như benchmark phụ cho generation, không phải benchmark retrieval chính.

### 6.2. Cách áp vào system

Pipeline:

1. lấy `question`
2. chạy retrieval pipeline bình thường của system
   - retrieve ở mức `child -> parent chunk`
   - dùng `parent chunk` làm context cho generator
3. sinh answer
4. so generated answer với:
   - `expert_curated_answer`
   - retrieved context
   - gold PMID citations nếu cần phân tích bổ sung

### 6.3. Metric dùng cho MedAESQA

Nên dùng cùng bộ generation metric chính:

- `ROUGE-SU4`
- `RAGAS Context Precision`
- `RAGAS Context Recall`
- `RAGAS Faithfulness`
- `RAGAS Answer Correctness`

Có thể thêm phân tích phụ:

- citation coverage
- citation precision theo PMID

### 6.4. Có dùng `medaesqa_eval.py` không?

Không dùng làm evaluator chính cho project.

Lý do:

- script đó cần annotation schema riêng
- cần `answer_sentence_relevance`
- cần `citation_assessment`
- cần thêm cluster files cho answer recall
- phù hợp hơn như **reference / supplementary protocol**

Quyết định chốt:

- evaluator chính cho project: **custom evaluator**
- `medaesqa_eval.py`: chỉ dùng làm tài liệu tham chiếu nếu cần reproduce protocol gốc

---

## 7. Output Structure

### 7.1. Folder

- `data/`
  - chỉ giữ raw / processed datasets

- `src/evaluation/`
  - code evaluator thật sự
  - chia theo:
    - `datasets`
    - `retrieval`
    - `generation`
    - `adapters`

- `scripts/evaluation/`
  - CLI entrypoints cho evaluation

- `results/eval_results/`
  - validation / dev outputs

- `results/test_results/`
  - final frozen test outputs

### 7.2. Output files

Mỗi evaluator nên sinh:

- `detail.jsonl`
- `summary.json`
- `predictions.jsonl`

---

## 8. Final Workflow

1. tune retrieval trên `val_bioasq`
2. freeze retrieval
3. tune generation trên `val_bioasq`
4. chạy final trên `test_bioasq`
5. chạy benchmark phụ trên `MedAESQA`
6. nếu cần, chạy `PubMedQA` như oracle-context benchmark
7. làm error analysis trên output cuối

# 🏥 KNOWLEDGE BASE CHUYÊN SÂU: MED-ASSISTANT (MEDICAL KG-RAG CHATBOT)

Tài liệu này đi sâu vào **chi tiết kỹ thuật, thuật toán và cách thức triển khai (implementation details)** của dự án **Med Assistant**. Mục tiêu là cung cấp các giải thích rành mạch nhất về "cách hệ thống vận hành bên dưới" (under the hood) để sử dụng cho các báo cáo kỹ thuật chuyên sâu, bảo vệ luận văn hoặc bàn giao mã nguồn.

---

## 1. KIẾN TRÚC LUỒNG XỬ LÝ (PIPELINE ARCHITECTURE)

Dự án áp dụng thiết kế **Clean Architecture (Dependency Inversion)**. Tầng nghiệp vụ không phụ thuộc vào tầng dữ liệu mà giao tiếp qua Interface.

- **Interfaces (Contracts)**: Hệ thống định nghĩa sẵn các interface trừu tượng tại `src/interfaces/`: `ISearchEngine`, `IKGSearcher`, `ILLMGenerator`, `IQueryAnalyzer`.
- **Thực thi Bất đồng bộ (Async Execution)**: Trong file orchestrator `rag_pipeline.py`, truy vấn tìm kiếm Vector, Keyword và KG được thực thi **hoàn toàn song song** thông qua `asyncio.gather()` hoặc các Thread Pool (`ThreadPoolExecutor`). Việc này giúp triệt tiêu độ trễ thắt cổ chai, thời gian lấy dữ liệu chỉ bằng thời gian của luồng chậm nhất.

---

## 2. KỸ THUẬT TIỀN XỬ LÝ & TÌM KIẾM VĂN BẢN (TEXT RETRIEVAL)

### 2.1. Phân mảnh văn bản thích ứng (Adaptive 3-Tier Chunking)
Để giải quyết bài toán tài liệu y khoa có nhiều từ viết tắt và khái niệm dính liền, hệ thống **không** dùng Text Splitter thông thường của LangChain. Thay vào đó, áp dụng **SciSpaCy** để ngắt câu y khoa mà không làm vỡ cụm từ học thuật.
Hệ thống sử dụng cơ chế **Parent-Child Chunking**:
- **Tiêu chí**: Tìm kiếm ngữ nghĩa cần đoạn ngắn (Child) để đạt độ chính xác cao. LLM sinh câu trả lời cần đoạn dài (Parent) để hiểu bối cảnh.
- **Tier 1 (Dưới 500 ký tự)**: Parent và Child bằng nhau, lấy nguyên bài Abstract.
- **Tier 2 (Từ 500 - 2000 ký tự)**: Parent là toàn bộ Abstract. Child cắt nhỏ ~500 ký tự. Đặc biệt, **Title Injection** (ghép Tiêu đề bài báo vào từng Child) được áp dụng để giữ ngữ cảnh độc lập cho đoạn văn.
- **Tier 3 (Trên 2000 ký tự)**: Parent cắt thành 1500 ký tự (có overlap 256 ký tự). Child cắt ~500 ký tự (Title Injected).

### 2.2. Tìm kiếm Vector (Dual-Encoder) & BM25
- Sử dụng mô hình bất đối xứng (Asymmetric): Câu hỏi đầu vào đi qua `ncbi/MedCPT-Query-Encoder`. Dữ liệu tĩnh đi qua `ncbi/MedCPT-Article-Encoder`.
- **Weaviate + SQLite**: Child chunks cùng Vector của chúng được nạp vào Weaviate để chạy Cosine Similarity (Vector Search) và BM25 (Keyword Search). Mỗi Child chứa một trường `parent_id`.
- **Tra cứu ngược O(log n)**: Khi Weaviate trả về Child, hệ thống lấy `parent_id` query vào SQLite (file `parent_chunks.db`) để bốc Parent text ra. Chỉ những Parent có điểm số cao nhất từ các Child của nó mới được giữ lại.

---

## 3. TRIỂN KHAI ĐỒ THỊ TRI THỨC Y KHOA (KNOWLEDGE GRAPH IMPLEMENTATION)

Sự kết hợp KG là thành tựu lớn nhất trong kiến trúc này để xử lý Multi-hop Reasoning (Tư duy đa bước).

### 3.1. Kỹ thuật nhúng Offline (Node Embedding)
Hệ thống dùng PrimeKG nhưng phải làm giàu ngữ nghĩa trước khi nhúng.
- **Chuẩn bị Text**: Tên Node đơn độc ("Metformin") sẽ thiếu vắng thông tin. Hệ thống tự ghép chuỗi theo template `"{NodeType}: {name}"` (VD: `"Drug: Metformin"`).
- **Nhúng**: Chạy qua `MedCPT-Article-Encoder` thu được vector 768 chiều. Lưu trực tiếp vào thuộc tính `embedding_medcpt` trên Neo4j. Tạo Vector Index `medcpt_node_embeddings` trong Neo4j (chuẩn Cosine).

### 3.2. Thuật toán tìm kiếm KG Online (2-Stage KG Search)
Việc tìm kiếm KG chia làm 2 giai đoạn (Inference):
- **Giai đoạn 1 (Anchor Search - Tìm Node Neo)**:
  - Llama 70B trích xuất các Entities từ câu hỏi (VD: "Bệnh tiểu đường").
  - Đem Entities mã hóa bằng `Article-Encoder` (So khớp A-E với A-E, cùng không gian không bị lệch).
  - Truy xuất top-k=3 Node có Vector gần nhất trong Neo4j làm "Node Neo".
- **Giai đoạn 2 (Neighbour Ranking - Khám phá đồ thị)**:
  - Câu hỏi đã được LLM viết lại (Rewritten Query) được nhúng qua `Query-Encoder`.
  - Mở rộng 1-hop và 2-hop từ các Node Neo. Khi đi qua các cạnh, bộ lọc Intent của LLM sẽ quyết định có được đi tiếp không (VD: câu hỏi tìm thuốc sẽ chỉ duyệt các cạnh `TREATS`, `TARGETS`).
  - Điểm số của mỗi Neighbour node được tính bằng **Cosine Similarity(Query Vector, Neighbour Vector)** (So khớp Q-E với A-E).
  - Cắt tỉa: Lấy Top-M=10 cho 1-hop, Top-N=5 cho 2-hop. Max 50 paths để tránh bùng nổ đồ thị.

### 3.3. Tuyến tính hóa Đồ thị (KG Linearization)
Đồ thị 2-hop được xuất ra dưới dạng Triples (A -> B -> C). Hệ thống dùng mã Python (Rule-based templates) chuyển cụm này thành ngôn ngữ tự nhiên. 
- *VD: `"[Drug] Metformin TARGETS [Gene] AMPK which is ASSOCIATED_WITH [Disease] Diabetes"`*.
- **Tối ưu nhánh cụt (Dead-end Optimization)**: Nếu đường 1-hop trùng với đoạn đầu của đường 2-hop, nhánh 1-hop sẽ bị loại bỏ để nhường chỗ trống cho context khác.

---

## 4. CÔNG NGHỆ XẾP HẠNG KÉP (TWO-STAGE RERANKING & FUSION)

Hệ thống sở hữu 3 nguồn tài liệu thô (Vector, BM25, KG). Phải kết hợp chúng cực kỳ cẩn thận.

### 4.1. Fusion bằng RRF (Reciprocal Rank Fusion)
- Gộp Vector Search và BM25 lại với nhau.
- **Công thức Toán học**: `Score_RRF = Σ (1 / (k + rank))` với hệ số `k=60` tiêu chuẩn. 
- **Lưu ý kỹ thuật**: KG tuyệt đối KHÔNG được gộp ở khâu này vì đường đi KG (Path) có tính cấu trúc, đưa vào RRF sẽ gây sai lệch phân phối toán học. Kết quả bước này gọi là **Text Retrieval**.

### 4.2. Rerank bằng Cross-Encoder
- Triển khai `ncbi/MedCPT-Cross-Encoder` trên **Modal GPU Cloud**.
- Đưa cặp (Query, Text Retrieval) và (Query, KG Paths) vào cùng một Batch để inference nhằm tối ưu tốc độ.
- **Quy tắc Vàng (OOD Bias Prevention)**: Dù tính chung batch, đầu ra phải được **tách riêng rẽ** và sắp xếp thành 2 list độc lập: Top-M Text và Top-N KG. Nếu trộn chung để sort, Text tự do thường có logic mượt mà hơn nên Cross-Encoder sẽ chấm điểm text cao hơn, dập tắt các bằng chứng KG.
- Giữ nguyên các Logits âm (`score <= 0`) thay vì cắt bỏ, vì bản chất MedCPT trả về raw logits, việc cắt bỏ sẽ làm mất Recall và Snippet Coverage.

---

## 5. THỦ THUẬT PROMPT & GENERATION (LLM GENERATION DETAILS)

Trước khi đưa context vào Llama 3.3 70B, một loạt các tiểu xảo (heuristics) được áp dụng ở `prompt_builder.py` và `kg_merger.py`:

### 5.1. Nén đồ thị (Post-Rerank KG Merging)
- Nhóm các đường đi có chung Prefix. Thay vì nạp vào LLM 2 câu: "A tác động B gây ra C" và "A tác động B gây ra D", thuật toán gộp thành "A tác động B gây ra C và D".
- **Density Bonus**: Khi gộp N path, điểm số của path tổng sẽ được cộng thưởng `Agg_Score = MAX(scores) + 0.05 * (N - 1)`. Đường nào càng nhiều nhánh con sẽ càng được đẩy lên ưu tiên.

### 5.2. Trộn đan xen 1-1 (Manual Interleaving)
Vì Text và KG xếp hạng độc lập, ta trộn ngữ cảnh theo kiểu "Zipper" (Khóa kéo): `Text 1, KG 1, Text 2, KG 2, Text 3...`. Điều này đảm bảo LLM nhìn thấy mật độ bằng chứng ở hai định dạng là tương đương.

### 5.3. Trị liệu "Lost-in-the-Middle" (Head-Tail Placement)
LLM thường chỉ nhớ phần đầu và phần cuối ngữ cảnh (U-shape Attention). Do đó:
- Top 1, 3, 5... đặt ở vị trí **HEAD** (Đầu prompt).
- Top 2, 4, 6... đặt ở vị trí **TAIL** (Cuối prompt).
- Các ngữ cảnh điểm thấp nhất bị nhồi vào **GIỮA** prompt.

### 5.4. Query Analyzer (Lịch sử & Anti-Preamble)
- **Cửa sổ Lịch sử (History Window)**: Hệ thống lấy đúng 5 lượt (5 turns = 10 messages, biến `HISTORY_TURNS_FOR_LLM`) đẩy vào Llama 70B (Groq) để phân tích đại từ nhân xưng và trích xuất Intent một lần duy nhất trước khi Retrieval. Lịch sử đầy đủ lưu trong PostgreSQL.
- **Type-conditional Prompting**: Llama xác định luôn loại câu hỏi (factoid, yes/no, list, summary) và tiêm thẳng format đầu ra vào System Prompt.
- **Anti-Preamble**: Chỉ thị cứng cấm LLM sinh mào đầu ("Based on the provided context..."). Code Python thực hiện hàm `strip_preamble()` loại bỏ triệt để trước khi tính điểm Evaluation (ROUGE-SU4) để không bị loãng điểm do trùng lặp từ vựng dư thừa.

---

## 6. CHI TIẾT VỀ MÔI TRƯỜNG ĐÁNH GIÁ (EVALUATION SPECS)

Khung đánh giá được thiết kế cứng (Deterministic) với `temperature=0` cho QueryAnalyzer.
- **Grid Search Framework**: Có hẳn module tự động quét qua ma trận Hyperparameters (`VECTOR_TOP_K`, `KEYWORD_TOP_K`, `RERANK_TEXT_TOP_M`...) trên 20 câu mẫu để chốt cấu hình Retrieval mạnh nhất.
- **Proxy Snippet Coverage**: Do không dùng chunk chữ làm dự đoán Snippet, hệ thống đánh giá xem văn bản của "Gold Snippet" có tồn tại (substring match) bên trong Parent Chunk được lấy về hay không.
- **MedAESQA Citation Pipeline**: Riêng MedAESQA được dùng riêng để đo đếm khả năng Trích dẫn (Citation-Precision, Citation-Recall, Citation-F1) của thuật toán. Luôn bật `use_citations=True` để kiểm tra Hallucination. Thêm toàn bộ PMID của MedAESQA vào corpus để tính coverage trọn vẹn, không coi đó là Data Leakage.

---
**Lời kết (Conclusion):** 
Bằng cách phân tách tinh tế các giai đoạn chunking, ứng dụng Dual-Encoder với không gian vector bất đối xứng, tìm kiếm lai Text + KG, và xử lý Prompt thông minh (Head-Tail, RRF, Interleaving), Med Assistant đạt hiệu năng Truy xuất (MRR ~0.86) và Sinh văn bản chuyên ngành vượt xa các giải pháp RAG mã nguồn mở hiện hành. Toàn bộ tính toán tốn kém nhất được san sẻ song song và dùng Cloud API để đạt độ trễ real-time thấp nhất.

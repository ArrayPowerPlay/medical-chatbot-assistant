# 🏥 KNOWLEDGE BASE TỔNG HỢP: MED-ASSISTANT (MEDICAL KG-RAG CHATBOT)

Tài liệu này là "bách khoa toàn thư" về dự án **Med Assistant**. Nó chứa toàn bộ mọi thông tin từ tổng quan đến chi tiết kỹ thuật sâu nhất, thiết kế kiến trúc, công nghệ, cách thu thập dữ liệu và toàn bộ các bộ chỉ số đánh giá (metrics). Bất cứ ai đọc tài liệu này đều có thể hiểu tường tận 100% về dự án và trả lời được mọi câu hỏi liên quan.

---

## 1. TỔNG QUAN DỰ ÁN (OVERVIEW & DOMAIN)

- **Tên dự án**: Med Assistant (MedKG-RAG Chatbot).
- **Mục tiêu**: Xây dựng một hệ thống AI Chatbot chuyên ngành y tế tiên tiến, có khả năng trả lời chính xác, tin cậy dựa trên các bằng chứng y khoa xác thực (Evidence-based) nhằm chống lại hiện tượng ảo giác (hallucination) của LLM.
- **Lĩnh vực (Domain)**: Dự án tập trung ranh giới chuyên môn hẹp và sâu: **Bệnh tật - Thuốc - Đích tác dụng (Disease-Drug-Target)**.
- **Công nghệ cốt lõi**: Kết hợp song song Kỹ thuật tìm kiếm lai (Hybrid Retrieval: Vector + BM25) với Hệ thống Đồ thị tri thức (Knowledge Graph - PrimeKG) và mô hình LLM Llama 3.3 70B.

---

## 2. PHÂN HỆ NGƯỜI DÙNG & TÍNH NĂNG WEB (USER ROLES & WEB FEATURES)

Hệ thống web được thiết kế như một Single Page Application (SPA) hiện đại, hỗ trợ Dark/Light mode và trả lời theo thời gian thực (Streaming SSE). 

**Hệ thống phân thành 3 vai trò:**

1. **Guest User (Khách vãng lai)**
   - Chỉ được hỏi tối đa **10 câu hỏi miễn phí**.
   - Cảnh báo giới hạn lượt hỏi hiển thị linh hoạt dưới cùng của tin nhắn AI mới nhất.
   - Không lưu lại lịch sử hội thoại vĩnh viễn (xóa sau khi tải lại trang).
2. **Registered User (Thành viên đăng ký)**
   - Không giới hạn số lượng câu hỏi.
   - **Lịch sử chat**: Lưu trữ vĩnh viễn, cho phép người dùng đổi tên, ghim (pin) hoặc xóa luồng chat. Cung cấp tính năng **Tìm kiếm hội thoại** (theo tiêu đề hoặc nội dung tin nhắn thông qua `/api/conversations/search`). Giao diện dùng pagination cuộn ngược (Reverse-scroll infinite chat history).
   - **Tính minh bạch (Verifiability)**: Nút bật/tắt để người dùng xem được các "Bằng chứng y khoa" (Sources) mà AI đã trích xuất, bao gồm các đoạn text từ PubMed, số PMID, đường dẫn Knowledge Graph và điểm số tương đồng (Similarity score).
   - **Hệ thống Feedback**: Người dùng có thể Like/Dislike cho câu trả lời và để lại bình luận văn bản giúp hệ thống cải thiện.
3. **Administrator (Quản trị viên)**
   - **Dashboard Phân tích**: Xem thống kê hệ thống, lượng user, lượng khách, số đăng ký mới, số câu hỏi trong ngày.
   - **Quản lý Users**: Tìm kiếm (theo Email/Username dạng `ILIKE` realtime), xóa user, buộc reset mật khẩu.
   - **Giám sát Chat**: Đọc được toàn bộ (Read-only) lịch sử chat của user nào đó để đánh giá tỷ lệ ảo giác của AI.
   - **Giám sát Feedback**: Quản lý Feedback Xấu/Tốt bên cạnh câu trả lời để đánh giá lại luồng RAG.

---

## 3. CÔNG NGHỆ ÁP DỤNG (TECH STACK & ARCHITECTURE)

Dự án áp dụng **Kiến trúc phân tầng (Layered Architecture) kết hợp Clean Architecture (Dependency Inversion)**, giúp hệ thống module hóa, dễ bảo trì và bóc tách hoàn toàn logic nghiệp vụ khỏi tầng dữ liệu. Hệ thống có 5 phân tầng chính:
1. **Tầng Trình diễn (Presentation Layer - Frontend)**: Ứng dụng SPA React (Vite, Tailwind v4) xử lý UI/UX phức tạp, phân quyền, và nhận stream chat.
2. **Tầng API & Routing (Backend)**: FastAPI cung cấp RESTful Endpoints, JWT Authentication, phân quyền (User/Admin/Guest) và đẩy dữ liệu SSE.
3. **Tầng Điều phối (Orchestration Layer - Use Case)**: Lớp `RAGPipeline` điều phối toàn bộ thuật toán: Query Analyzer, các bộ Retriever (Vector/BM25/KG), RRF, Reranker, và Generator. Lớp này độc lập với Database.
4. **Tầng Giao tiếp Dữ liệu (Data Access / Infrastructure)**: Các lớp Repositories triển khai theo Hợp đồng (`src/interfaces`) giao tiếp trực tiếp với PostgreSQL, Neo4j, Weaviate và SQLite.
5. **Tầng Dữ liệu ngoại tuyến (Data Pipeline - Offline)**: Tập hợp các kịch bản cào dữ liệu, chia nhỏ (chunking), nhúng vector và hydrate Database.

**Bộ Công nghệ (Tech Stack) đầy đủ:**
- **Frontend**: React, TypeScript, Vite, Tailwind CSS v4. Quản lý state bằng Zustand, call API bằng Axios (JWT Interceptor).
- **Backend**: Python 3.11+, FastAPI (RESTful & SSE Streaming), Pydantic (Validate dữ liệu).
- **LLM & API**:
  - LLM Sinh đáp án & Phân tích câu hỏi: `meta-llama/Llama-3.3-70B-Versatile` (Chạy qua nền tảng **Groq API** cho tốc độ siêu nhanh).
  - Cross-Encoder: Chạy trên Cloud GPU của nền tảng **Modal**.
- **Cơ sở dữ liệu (Storage & Infrastructure)**:
  - **PostgreSQL**: Lưu User, Lịch sử Chat, Feedback (Chạy Docker, mount volume).
  - **Weaviate**: Vector DB dùng để chứa Text Embeddings và chạy BM25 Search (Chạy Docker).
  - **Neo4j**: Graph DB lưu trữ Đồ thị PrimeKG (Chạy Docker).
  - **SQLite**: Lưu Parent Chunks để tra cứu với tốc độ siêu tốc (O(log n)).
- **Mô hình Nhúng (Embedding & Reranker)**: 
  - `ncbi/MedCPT-Article-Encoder`, `ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Cross-Encoder`.
- **Môi trường Đánh giá (Evaluation)**: Thư viện `ragas`. Dùng `gpt-4o-mini` và `text-embedding-3-small` làm giám khảo.

---

## 4. DỮ LIỆU & CÁCH THỨC THU THẬP (DATA SOURCES & COLLECTION)

Hệ thống kết hợp tài liệu văn bản phi cấu trúc (Unstructured Text) và Đồ thị có cấu trúc (Structured Graph).

### 4.1. Text Corpus (BioASQ PubMed)
- **Nguồn**: Lấy từ BioASQ PubMed Annual Baseline Corpus. Kích thước khoảng 300,000 bài báo y khoa (Title + Abstract).
- **Cách thu thập & Chọn lọc**: Dữ liệu thô được filter dựa trên **MeSH Tree Numbers**, chỉ giữ lại các bài báo thuộc nhánh C (Diseases - Bệnh tật) và D (Chemicals and Drugs - Thuốc & Hóa chất).
- **Bổ sung PMIDs**: Để phục vụ cho cả việc đánh giá tập MedAESQA, toàn bộ các mã PMID tham chiếu của BioASQ Test Set và MedAESQA Test Set đều được kịch bản gọi trực tiếp vào **NCBI E-utilities API** tải về chuẩn xác (Title, Abstract) và nạp bổ sung vào Corpus.
- **Tiền xử lý & Chunking (Adaptive 3-Tier Chunking)**: 
  Không dùng Text Splitter thông thường, hệ thống dùng **SciSpaCy** để ngắt câu y khoa mà không làm vỡ các từ viết tắt chuyên ngành.
  - **Parent-Child Logic**: LLM cần ngữ cảnh dài (Parent), nhưng Weaviate Vector/BM25 cần mảnh nhỏ để tìm cho chuẩn (Child).
  - **Tier 1 (<= 500 ký tự)**: Parent = Child = Full Article.
  - **Tier 2 (<= 2000 ký tự)**: Parent = Full Article. Child = cắt nhỏ ~500 ký tự. Có nhồi thêm Tiêu đề bài báo vào đầu Child (**Title Injection**).
  - **Tier 3 (> 2000 ký tự)**: Parent = 1500 chars (overlap 256). Child = cắt ~500 chars (+ Title Injection).
  - Sau khi chunk, Child chunks đẩy lên Weaviate (mang theo `parent_id`), Parent chunks lưu vào SQLite (`parent_chunks.db`).

### 4.2. Knowledge Graph (PrimeKG)
- **Nguồn**: PrimeKG của Đại học Harvard.
- **Cách thu thập & Tối ưu**: Đồ thị thô chứa mọi thứ. Hệ thống chạy script `build_kg.py` để quét và giữ lại ĐÚNG các loại Node: `Disease`, `Drug`, `GeneProtein`, `EffectPhenotype`.
- Giữ lại các Edge cốt lõi như: `TREATS`, `CONTRAINDICATES`, `TARGETS`, `HAS_SIDE_EFFECT`, `ASSOCIATED_WITH`, v.v...
- **Tạo Node Embeddings (Offline)**: Để tìm kiếm được trên KG bằng Vector, hệ thống dán nhãn Node bằng template `"{NodeType}: {name}"` (VD: "Drug: Metformin"). Sau đó nhúng qua mô hình `MedCPT-Article-Encoder` thu được vector 768 chiều. Lưu thẳng vector này vào thuộc tính `embedding_medcpt` trên Neo4j.

---

## 5. CHI TIẾT LUỒNG THỰC THI (END-TO-END RAG PIPELINE)

Quy trình RAG (Retrieval-Augmented Generation) được thực thi hoàn toàn bất đồng bộ (Async) ở các bước truy vấn.

### Giai đoạn 1: Query Analyzer (Phân tích câu hỏi bằng LLM)
- Lấy 5 lượt chat gần nhất từ PostgreSQL (config: `HISTORY_TURNS_FOR_LLM = 5`).
- Gửi Prompt cho Llama 70B để thực hiện 4 tác vụ:
  1. Gộp ngữ cảnh/sửa lỗi chính tả (Rewriting).
  2. Xác định Loại câu hỏi (Question Type): factoid, summary, list, yesno.
  3. Phân loại Ý định (Intent): `treatment_lookup`, `symptom_lookup`, `no_rag_needed` (nếu chat ngoài lề y khoa, hệ thống báo bypass RAG để trả lời luôn).
  4. Trích xuất Thực thể (NER): Bệnh, Thuốc, Gen...

### Giai đoạn 2: Tìm kiếm Song song (Parallel Retrieval)
Thực thi bằng `asyncio.gather` qua 3 luồng:
1. **Vector Search (Weaviate)**: Câu hỏi được nhúng qua `MedCPT-Query-Encoder`. Query vào Weaviate bằng Cosine Similarity để lấy ra Top Child Chunks. Lấy `parent_id` quét vào SQLite để lôi Parent text ra.
2. **Keyword Search (Weaviate - BM25)**: Bắt đúng từ khóa chính xác trên Weaviate. Cắt nghĩa bù cho Vector. Cũng lấy Parent text tương tự.
3. **Knowledge Graph Search (Neo4j)**:
   - *Bước Neo (Anchor)*: Các Entities trích xuất từ Bước 1 được nhúng bằng `MedCPT-Article-Encoder`. So sánh Vector (A-E vs A-E) trên Neo4j để tìm top 3 Node Neo (Anchor nodes).
   - *Bước Mở rộng (Ranking)*: Từ Node Neo, lan truyền 1-hop và 2-hop (chỉ đi qua các cạnh mà LLM Intent cho phép). Điểm số các node lân cận được tính bằng Cosine Similarity giữa Vector câu hỏi (`Query-Encoder`) và Vector Node (`Article-Encoder`).
   - *Bước Tuyến tính hóa (Linearization)*: Biến cấu trúc A->B->C của đồ thị thành câu văn tiếng Anh (Template rule-based). Bỏ các đường nhánh 1-hop cụt nếu nó là chặng đầu của một nhánh 2-hop (Dead-end optimization).

### Giai đoạn 3: Hệ thống Xếp hạng Kép (Two-Stage Reranking)
1. **RRF (Reciprocal Rank Fusion)**: Dùng công thức toán `Σ (1 / (k + rank))` (k=60) để gộp Text Vector + Text BM25 lại thành một list duy nhất gọi là **Text Search**.
2. **Cross-Encoder (Reranking)**: Bắn cả Text Search và KG Paths lên API Modal Cloud chạy `MedCPT-Cross-Encoder`. 
   - *Quy tắc Chống Bias*: Dù tính điểm chung batch, đầu ra được SẮP XẾP RIÊNG RẼ thành Top-M Text và Top-N KG. Nếu sort chung, Text mạch lạc sẽ nuốt chửng điểm của KG Paths (OOD Bias). Hệ thống giữ lại cả các điểm logits âm (`score <= 0`) để giữ tính tuần tự.

### Giai đoạn 4: Hậu xử lý & Sinh đáp án (Prompt Builder & Generator)
- **KG Merging**: Nén ngữ cảnh KG. Nếu có 2 dòng "A targets B" và "A targets C", sẽ gộp thành "A targets B and C". Thưởng điểm Density Bonus (`+0.05` điểm cho mỗi nhánh ghép).
- **Interleaving**: Trộn đan xen ngữ cảnh dạng Khóa kéo: 1 Text, 1 KG, 1 Text, 1 KG.
- **Head-Tail Placement (Chống "Lost-in-the-Middle")**: Các đoạn Text điểm cao nhất được đặt ở Đầu và Cuối Prompt, đoạn điểm thấp nhét vào Giữa.
- **Anti-Preamble**: Ép Llama 70B không được sinh mào đầu ("Based on context...").
- Cuối cùng, Llama 70B sinh câu trả lời và stream (SSE) kết quả về Web.

---

## 6. MÔI TRƯỜNG ĐÁNH GIÁ & CÁC METRICS (EVALUATION SPECS)

Đánh giá học thuật khắt khe, chạy qua scripts tại `scripts/evaluation/`. Nhiệt độ của QueryAnalyzer luôn khóa ở `0` (Deterministic).

### 6.1. Tập Dữ Liệu Đánh Giá (Datasets)
- **BioASQ Task B**: 500 câu Validation (để tinh chỉnh Hyperparameters), 500 câu Test (để chốt kết quả). Tập trung đánh giá Text Retrieval và Answer Generation.
- **MedAESQA**: 42 câu Test chuyên biệt có tính đa bước cao. Chỉ dùng để test khâu Generation (đặc biệt là khả năng sinh Trích dẫn).

### 6.2. Các Chỉ số Tìm kiếm (Retrieval Metrics) - Chạy trên BioASQ
Tính toán tại K=5, 10, 20. Khung Grid Search chạy quét ma trận tham số cấu hình.
- **Document-Level Metrics (So khớp PMID)**:
  - `Precision@K`: Tỷ lệ tài liệu lấy về là đúng.
  - `Recall@K`: Tỷ lệ tài liệu chuẩn (Gold) đã tìm thấy trong top K.
  - `F1@K`: Trung bình điều hòa của Precision và Recall.
  - `MAP@K` (Mean Average Precision): Điểm AP trung bình cắt tại K.
  - `GMAP@K` (Geometric MAP): Trung bình nhân, phạt nặng các câu hỏi điểm quá thấp.
  - `MRR` (Mean Reciprocal Rank): Nghịch đảo vị trí của tài liệu chuẩn đầu tiên xuất hiện.
- **Snippet-Level Proxy Metrics**: Do không dùng chunk làm dự báo Snippet, dự án dùng Proxy: Một "Gold Snippet" được coi là tìm thấy nếu chuỗi của nó nằm lọt bên trong Parent Chunk đã tải về. Tính ra các chỉ số: `Snippet Precision@K`, `Snippet Recall@K`, `Snippet F1@K`.

### 6.3. Các Chỉ số Sinh văn bản (Generation Metrics)
- **Trên tập BioASQ**:
  - `ROUGE-SU4-F1`: So khớp từ vựng chuẩn BioASQ (Chạy qua hàm `strip_preamble` cắt mào đầu trước khi tính).
  - Khung **RAGAS** (Sử dụng LLM GPT-4o-mini làm giám khảo): 
    - `Context Precision` & `Context Recall` (Chất lượng ngữ cảnh).
    - `Faithfulness` (Đáp án có trung thành với ngữ cảnh không).
    - `Answer Correctness` (Đáp án có đúng nghĩa không).
    - `Answer Relevancy` (Đáp án có sát câu hỏi không).
- **Trên tập MedAESQA (Đánh giá khả năng trích dẫn - Citation Pipeline)**:
  - Bật mode `use_citations=True`. Không dùng RAGAS vì không cần thiết.
  - Tính `ROUGE-SU4-F1`.
  - Tính Citation Metrics: `Citation-Precision` (Trích dẫn có khớp tài liệu không), `Citation-Recall` (Có dẫn đủ tài liệu chuẩn không), `Citation-F1`.

*(Kết quả mạnh nhất cấu hình hiện tại đạt MRR Retrieval ~0.86, bỏ xa các Baseline thuần Text).*

---

## 7. HƯỚNG DẪN TRIỂN KHAI (QUICK START)

1. Yêu cầu: Python 3.11+, Node.js >= 18, Docker & Docker Compose.
2. Chuẩn bị file `.env` chứa API Keys: Groq, OpenAI, Modal Token, URL DB.
3. Chạy cơ sở hạ tầng (PostgreSQL, Neo4j, Weaviate):
   ```bash
   docker-compose up -d
   ```
4. Khởi chạy Backend API (FastAPI):
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
5. Khởi chạy Frontend (Vite/React):
   ```bash
   cd frontend
   npm run dev
   ```
Truy cập `http://localhost:5173`.
Hệ thống cũng đi kèm một loạt notebooks trong `notebooks/` và scripts build dữ liệu offline trong `src/dataset_builder/` dành cho Dev thao tác.

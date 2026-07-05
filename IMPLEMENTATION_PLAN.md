# MedKG-RAG: Kế Hoạch Triển Khai (Implementation Plan)

Tài liệu này định nghĩa chi tiết các Phase (Giai đoạn) thực hiện việc tái cấu trúc dự án theo mô hình Clean Architecture (Phân tầng, Dependency Inversion), chuyển đổi sang Async, và thiết lập môi trường cho Ablation Study. 

---

## 🚀 Phase 1: Xây dựng Tầng Cốt lõi (Domain / Interfaces Layer)
**Mục tiêu**: Định nghĩa các "hợp đồng" (interfaces) để tách biệt logic nghiệp vụ khỏi chi tiết hạ tầng. Không có code truy cập dữ liệu ở đây.

**Các công việc cụ thể**:
1. Tạo thư mục `src/interfaces/`.
2. Tạo file `src/interfaces/storage.py`: Định nghĩa abstract class `ISearchEngine`, `IParentStore`.
3. Tạo file `src/interfaces/kg.py`: Định nghĩa abstract class `IKGSearcher`.
4. Tạo file `src/interfaces/llm.py`: Định nghĩa abstract class `ILLMGenerator`, `IQueryAnalyzer`.
5. Tạo file `src/interfaces/embeddings.py`: Định nghĩa abstract class `IEmbedder`.

---

## 🚀 Phase 2: Nâng cấp Tầng Hạ tầng (Infrastructure Layer) & Async
**Mục tiêu**: Chuyển các client hiện tại sang bất đồng bộ (async) và cho chúng kế thừa từ các interfaces ở Phase 1.

**Các công việc cụ thể**:
1. **Weaviate**: Cập nhật `src/storage/weaviate_client.py` thành `AsyncWeaviateChildStore`, dùng async client, kế thừa `ISearchEngine`.
2. **Neo4j**: Cập nhật `src/kg/neo4j_client.py` và `src/kg/kg_search.py` để sử dụng `AsyncGraphDatabase` và kế thừa `IKGSearcher`.
3. **Groq LLM**: Cập nhật `src/generation/llm_generator.py` và `src/query/query_analyzer.py` sử dụng `AsyncGroq`, kế thừa `ILLMGenerator` và `IQueryAnalyzer`.
4. **Embeddings**: Cập nhật `src/embeddings/medcpt_embedder.py` (chạy song song/async wrapping nếu cần).
5. Đảm bảo mọi method truy xuất dữ liệu đều được đổi thành `async def`.

---

## 🚀 Phase 3: Cập nhật Tầng Nghiệp vụ (Use Case Layer)
**Mục tiêu**: Loại bỏ code hardcode khởi tạo hạ tầng, áp dụng Dependency Injection (DIP) và chạy luồng song song bất đồng bộ.

**Các công việc cụ thể**:
1. **Parallel Retriever**: 
   - Cập nhật `src/retrieval/parallel_retrieval.py` 
   - Xóa bỏ `concurrent.futures.ThreadPoolExecutor`.
   - Thay thế bằng `await asyncio.gather(...)` để chạy 3 luồng (Vector, BM25, KG) đồng thời mà không block thread.
2. **RAG Pipeline**: 
   - Cập nhật `src/pipeline/rag_pipeline.py`.
   - Sửa hàm `__init__` để nhận dependency từ bên ngoài (nhận vào `query_analyzer: IQueryAnalyzer`, `search_engine: ISearchEngine`, v.v.).
   - Tạo class `RunConfig` để nhận cờ cho Ablation Study (ví dụ: `use_kg`, `use_bm25`, `use_cross_encoder`).
   - Sửa hàm `run()` thành `async def run(...)`.

---

## 🚀 Phase 4: Cấu hình Tầng Trình bày (Presentation Layer) & DI Container
**Mục tiêu**: Cập nhật API FastAPI để nó tự động tiêm (inject) các phụ thuộc vào Pipeline khi khởi động server.

**Các công việc cụ thể**:
1. **API Router**: Cập nhật `api/routes/chat.py` để dùng `await pipeline.run(...)`.
2. **Main / Lifespan**: Trong `api/main.py`, thiết lập Dependency Injection:
   - Khởi tạo các hạ tầng (`AsyncWeaviateChildStore`, `AsyncGroq`, `Neo4jClient`...).
   - Bơm chúng vào `RAGPipeline(...)`.
   - Gán pipeline đã setup vào `app.state.pipeline`.

---

## 🚀 Phase 5: Xây dựng Cấu trúc Ablation Study (Code & Results)
**Mục tiêu**: Tạo các file script test riêng biệt cho từng version để đảm bảo code chạy và kết quả không bị trộn lẫn, dễ dàng truy vết.

**Các công việc cụ thể**:
1. Tạo 3 thư mục trong `scripts/evaluation/bioasq/`:
   - `baseline_vector/`
   - `no_kg_hybrid/`
   - `full_system/`
2. Trong mỗi thư mục, tạo 2 file `test_generation.py` và `test_retrieval.py` (Không tạo file `val_`).
3. Trong các file này, set cứng `RunConfig` phù hợp với tên của version. 
4. Chỉnh sửa logic ghi file trong `scripts/evaluation/shared/generation_bioasq_common.py` và `retrieval_common.py` để chúng lưu kết quả tương ứng vào:
   - `results/test_results/bioasq/baseline_vector/`
   - `results/test_results/bioasq/no_kg_hybrid/`
   - `results/test_results/bioasq/full_system/`
5. Đảm bảo các script evaluation cũ (ví dụ `scripts/evaluation/bioasq/test_generation.py` cũ) vẫn lưu đúng vào thư mục mặc định của nó như ban đầu.

---
*Vui lòng ra lệnh cho tôi nếu bạn muốn tôi bắt đầu lập trình Phase 1!*

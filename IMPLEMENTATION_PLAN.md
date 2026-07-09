# Bỏ qua Query Rewriting cho các câu hỏi không cần RAG (no_rag_needed)

Trong yêu cầu, bạn muốn thảo luận về việc không dùng `rewritten_query` khi intent là `no_rag_needed` và suy nghĩ xem việc Rewrite Query có thực sự quan trọng hay không. Dưới đây là phân tích và kế hoạch thực hiện.

## 1. Tư duy: Việc Rewrite Query có thực sự quan trọng không?

Việc **Rewrite Query (Viết lại câu hỏi)** cực kỳ quan trọng trong RAG, nhưng **chỉ quan trọng đối với bước Retrieval (Truy xuất tài liệu)**.

**Tại sao nó quan trọng cho Retrieval?**
- **Giải quyết ngữ cảnh (Coreference Resolution):** Nếu người dùng hỏi câu 1 là "Aspirin là gì?" và câu 2 là "Tác dụng phụ của nó là gì?", từ "nó" sẽ khiến máy tìm kiếm thất bại. Rewrite Query sẽ biến câu 2 thành "Tác dụng phụ của Aspirin là gì?".
- **Tối ưu hóa từ khóa (Keyword/Search Optimization):** Người dùng thường nhập rườm rà ("Bác sĩ ơi cho tôi hỏi dạo này tôi hay bị đau đầu và chóng mặt thì là bệnh gì"). Rewrite Query sẽ cô đọng lại thành "nguyên nhân đau đầu chóng mặt", giúp ElasticSearch/Vector Search tìm kiếm chính xác hơn.

**Tại sao nó KHÔNG cần thiết cho LLM Generation (khi no_rag_needed)?**
- Khi người dùng nói chuyện phiếm ("Xin chào", "Bạn là ai", "Cảm ơn"), LLM nên nhận được **chính xác câu nói** của người dùng để trả lời tự nhiên nhất.
- Nếu dùng `rewritten_query` (ví dụ "greeting" hoặc ""), LLM sẽ thiếu ngữ cảnh cảm xúc và sắc thái của câu gốc, hoặc thậm chí lỗi nếu chuỗi rỗng.
- Việc bạn đề xuất dùng trực tiếp câu hỏi gốc (original query) cho các trường hợp không cần RAG là **hoàn toàn chính xác và hợp lý**.

> [!TIP]
> **Kết luận:** Giữ nguyên cơ chế Rewrite Query để phục vụ tìm kiếm RAG, nhưng **bỏ qua kết quả rewrite** (dùng câu hỏi gốc) khi giao tiếp trực tiếp với LLM (intent = `no_rag_needed`).

## 2. Proposed Changes (Kế hoạch thay đổi)

Dưới đây là kế hoạch cập nhật mã nguồn để ưu tiên sử dụng `original_query` thay vì `rewritten_query` cho các intent không cần RAG.

### `src/pipeline/rag_pipeline.py`

Cập nhật hàm `run` và `run_stream` để chọn câu hỏi đầu vào cho LLM Generator.

#### [MODIFY] [rag_pipeline.py](file:///d:/workspace/Repo/medical-chatbot-assistant/src/pipeline/rag_pipeline.py)
- Thay vì luôn truyền `rewritten_query` vào `build_prompts`, ta sẽ kiểm tra `intents`.
- Thêm logic: 
  ```python
  # Sử dụng original query nếu intent là no_rag_needed, ngược lại dùng rewritten_query
  final_query_for_llm = query if "no_rag_needed" in intents else rewritten_query
  
  system_prompt, user_prompt = build_prompts(
      query=final_query_for_llm,
      ...
  )
  ```
- Nếu `no_rag_needed`, `vector_results`, `bm25_results`, `kg_results` đều được bỏ qua (như bạn đã đề cập ở phiên trước). `interleaved_items` sẽ là mảng rỗng.

### `src/generation/prompt_builder.py`

#### [MODIFY] [prompt_builder.py](file:///d:/workspace/Repo/medical-chatbot-assistant/src/generation/prompt_builder.py)
- Đảm bảo rằng prompt template hoạt động tốt khi `retrieved_items` rỗng (hiện tại có lẽ đã hỗ trợ nhưng ta cần đảm bảo system prompt cho phép LLM tự trả lời mà không bị ép buộc phải "chỉ dùng context").

## User Review Required

> [!IMPORTANT]
> - Bạn có đồng ý với phương án chỉ sử dụng câu hỏi gốc (`query`) để generate câu trả lời khi `intent == "no_rag_needed"`, trong khi vẫn giữ `rewritten_query` cho nhánh RAG không?
> - Lỗi nhập lung tung khiến `rewritten_query = ""` cũng sẽ tự động được xử lý bằng cách này, vì lúc đó analyzer có thể xếp nó vào `no_rag_needed` (hoặc `general`) và truyền thẳng chuỗi gốc cho LLM xử lý. Bạn thấy hợp lý chứ?

Bạn hãy xem qua và phản hồi nhé. Nếu đồng ý, tôi sẽ tiến hành thực thi!

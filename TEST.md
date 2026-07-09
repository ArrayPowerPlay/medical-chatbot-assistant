# Hướng dẫn Test Backend (FastAPI) qua Swagger UI

Vì dự án hiện tại chưa có giao diện Frontend (React) hoàn thiện, FastAPI cung cấp sẵn một công cụ rất mạnh mẽ để kiểm thử API có tên là **Swagger UI**. Bạn không cần tải Postman mà có thể test trực tiếp trên trình duyệt.

## 1. Mở Swagger UI
1. Chạy server backend của dự án (thường bằng lệnh `uvicorn api.main:app --reload` hoặc chạy script khởi động tương ứng).
2. Mở trình duyệt và truy cập vào: **`http://localhost:8000/docs`** (nếu cấu hình cổng là 8000).

Tại đây, bạn sẽ thấy danh sách tất cả các API của dự án như `/api/auth/register`, `/api/chat`,...

## 2. Các bước Test luồng User & Chat

### Bước 1: Tạo tài khoản hoặc Đăng nhập Guest
Để gọi các API của Phase 2, bạn cần có quyền truy cập (Token).
- Tìm đến mục **POST `/api/auth/guest`** (hoặc `/api/auth/register` nếu bạn muốn tạo tài khoản).
- Nhấn nút **"Try it out"** ở góc phải của API đó.
- Nhấn **"Execute"**.
- Kéo xuống phần **Server response** > **Response body**, bạn sẽ thấy một chuỗi `access_token` được trả về. Hãy bôi đen và **copy (Ctrl+C)** chuỗi ký tự dài ngoằng đó (đừng copy dấu ngoặc kép).

### Bước 2: Khai báo Token để xác thực (Authorize)
- Kéo lên trên cùng của trang Swagger UI, bạn sẽ thấy một nút màu xanh lá có chữ **"Authorize"** kèm icon ổ khóa.
- Nhấn vào nút đó. Một hộp thoại sẽ hiện ra yêu cầu nhập giá trị vào trường `HTTPBearer`.
- **Dán (Ctrl+V)** cái token bạn vừa copy ở Bước 1 vào ô trống đó.
- Nhấn **"Authorize"**, sau đó nhấn **"Close"**.
- *Lúc này, trên tất cả các API sẽ có icon ổ khóa đang khóa lại, báo hiệu rằng bạn đã "đăng nhập" thành công vào Swagger.*

### Bước 3: Tạo thử một cuộc hội thoại mới
- Tìm đến mục **POST `/api/chat`**.
- Nhấn **"Try it out"**.
- Ở ô **Request body**, bạn nhập thông tin dưới dạng JSON, ví dụ:
  ```json
  {
    "question": "What are the symptoms of Type 2 Diabetes?",
    "conversation_id": null
  }
  ```
- Nhấn **"Execute"**.
- API sẽ trả về câu trả lời, nguồn (sources) và đặc biệt là `conversation_id`. Hãy **copy cái `conversation_id`** này để dùng cho bước sau.

### Bước 4: Test các API của Phase 2
Bạn có thể tiếp tục test các API như:
- **PUT `/api/conversations/{conv_id}`**: Dán cái id vừa copy vào mục `conv_id`, sau đó đổi title hoặc bật `is_pinned = true`.
- **GET `/api/conversations`**: Chạy thử để xem danh sách hội thoại của bạn.
- **GET `/api/conversations/search?q=Diabetes`**: Chạy thử để tìm kiếm.

---

## 3. Nếu dùng Postman hoặc công cụ tương tự
Nếu bạn dùng Postman, Insomnia hay bất kỳ HTTP Client nào khác, quá trình tương tự:
1. Gọi API `POST http://localhost:8000/api/auth/guest`.
2. Lấy `access_token` từ response.
3. Ở các API khác (như `/api/chat`), chuyển sang tab **Headers** hoặc **Auth** (chọn loại Bearer Token), và dán token vào. Header thực tế sẽ trông giống thế này:
   `Authorization: Bearer eyJhbGciOiJIUzI1...`
4. Gửi request.

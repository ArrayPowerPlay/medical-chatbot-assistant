# Implementation Plan: New Features for MedKG-RAG Chatbot

Mục tiêu: Bổ sung các tính năng tương tác người dùng, quản lý tài khoản, cải thiện UX cho chatbot và chuyển đổi kiến trúc Frontend sang React.js. Kế hoạch được chia thành các Phase nhỏ để dễ dàng triển khai và kiểm thử.

## Cấu trúc Tổng quan

- **Frontend**: React.js (Vite) + Tailwind CSS + Zustand (State Management) + Axios (Data Fetching).
- **Backend**: FastAPI + PostgreSQL.

---

## Các Giai đoạn Triển khai (Phases)

### Phase 1: Database & Backend Auth (Authentication)
*Thiết lập hạ tầng tài khoản và bảo mật.*

1. **Database Schema**:
   - Thêm bảng `Users` (PostgreSQL): `id`, `email`, `password_hash`, `role` (user/guest), `question_count` (int, default=0), `created_at`.
   - Cập nhật bảng `Conversation`: thêm khóa ngoại `user_id` (liên kết với `Users`), thêm cột `is_pinned` (boolean, default=false).
   - Cập nhật bảng `Message`: thêm cột `feedback_type` (like/dislike/none), `feedback_comment` (text).
2. **Xác thực (Auth)**:
   - Tích hợp thư viện `passlib[bcrypt]` và sinh JWT token.
   - Viết các API: `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/guest`.
   - Viết middleware/dependency FastAPI để xác thực header `Authorization: Bearer <token>`.

### Phase 2: Backend APIs (Conversations & Streaming Chat)
*Nâng cấp các API chat và quản lý lịch sử.*

1. **Quản lý Cuộc hội thoại (Conversation APIs)**:
   - **Ghim & Đổi tên**: `PUT /api/conversations/{id}` nhận dữ liệu update `title` và `is_pinned`.
   - **Xóa**: `DELETE /api/conversations/{id}`.
   - **Tìm kiếm**: `GET /api/conversations/search?q=...` (Exact match trên `title` hoặc `content` của message).
   - Cập nhật API GET danh sách: Sắp xếp theo `is_pinned` trước (theo thời gian giảm dần), sau đó đến hội thoại không ghim (theo thời gian giảm dần). Tất cả hiển thị chung một danh sách, hội thoại được ghim sẽ có icon đánh dấu.
2. **Feedback API**:
   - `POST /api/conversations/{id}/messages/{msg_id}/feedback`: Nhận và cập nhật `feedback_type`, `feedback_comment` vào bảng `Message`.
3. **Chat & Streaming (SSE)**:
   - Cập nhật `POST /api/chat`: Thêm logic kiểm tra (Nếu role 'guest' và `question_count >= 10` -> 403 Forbidden). Cập nhật `question_count` sau mỗi câu trả lời.
   - Tích hợp **Server-Sent Events (SSE)**: Thay vì trả về 1 JSON cục lớn, gọi Groq API ở chế độ `stream=True` và dùng `StreamingResponse` của FastAPI để yield từng từ (token) về client ngay lập tức.
   - Đảm bảo bắt các sự kiện ngắt kết nối (`Request.is_disconnected`) hoặc `CancelledError` để hủy luồng gọi Groq LLM nếu user bấm Stop.

### Phase 3: Frontend Foundation & Auth UI
*Xây dựng bộ khung Frontend bằng React.*

1. **Khởi tạo Project**:
   - Setup React (Vite) + Tailwind CSS. Cấu hình hỗ trợ Dark Mode qua CSS variables (`dark:class`).
   - Setup `Axios` interceptor để tự động chèn JWT token vào header của mọi request.
   - Khởi tạo Store với `Zustand` để quản lý Global State: User, Token, Theme, và danh sách Conversations.
2. **Auth UI**:
   - Xây dựng giao diện Login, Register, và nút "Continue as Guest".
   - Xử lý luồng đăng nhập, lưu JWT vào `localStorage` và chuyển hướng (redirect) vào trang Chat chính.

### Phase 4: Frontend Chat Layout & Streaming Integration
*Giao diện chat chính.*

1. **Layout**:
   - Khung Chat chính và Sidebar.
   - Thêm tính năng **Mở/Đóng Sidebar** (Toggled thông qua icon Hamburger).
   - Nút bật/tắt Theme (Sáng/Tối).
2. **Chat & Streaming**:
   - Giao diện bong bóng chat (User vs Assistant) với Markdown render.
   - Tích hợp client xử lý SSE để đọc từng token và hiển thị chữ gõ dần dần (Typing effect).
   - Hiện vòng xoay "Thinking..." khi đợi token đầu tiên từ SSE.
   - Thêm nút **Stop Generating** và sử dụng `AbortController` để ngắt request fetch SSE.
   - Xử lý lỗi HTTP 403: Vô hiệu hóa ô nhập chat và thông báo vượt giới hạn cho Guest.

### Phase 5: Frontend Conversation Management & Tương tác
*Các tính năng cá nhân hóa lịch sử.*

1. **Sidebar Lịch sử**:
   - Render chung một danh sách hội thoại (Pinned nằm ở trên cùng, hiện icon chiếc ghim kế bên).
   - Nút "Thùng rác" để xóa hội thoại.
   - Nút "Cây bút" để đổi tên hội thoại.
   - Thanh Search Bar ở đầu Sidebar, khi gõ sẽ gọi API tìm kiếm và hiển thị kết quả.
2. **Feedback UI**:
   - Thêm icon 👍 / 👎 dưới mỗi câu trả lời của LLM.
   - Click chọn sẽ hiển thị textarea nhỏ để nhập comment -> nhấn Gửi để gọi API feedback.

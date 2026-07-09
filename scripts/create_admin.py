import argparse
import sys
import os

# Add the project root to the Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passlib.context import CryptContext
from src.storage.conversation_store import ConversationStore

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def main():
    parser = argparse.ArgumentParser(description="Tạo tài khoản Admin cho hệ thống Medical Chatbot.")
    
    group = parser.add_mutually_exclusive_group(required=True)   # Only use username OR email, not both
    group.add_argument("--username", type=str, help="Username của Admin")
    group.add_argument("--email", type=str, help="Email của Admin (sẽ được dùng làm username khi đăng nhập)")
    
    parser.add_argument("--password", type=str, required=True, help="Mật khẩu của Admin")
    
    args = parser.parse_args()
    
    identifier = args.username if args.username else args.email
    password = args.password
    
    try:
        store = ConversationStore()
        
        # Check if user already exists
        existing_user = store.get_user_by_username(identifier)
        if existing_user:
            print(f"[-] Lỗi: Username hoặc Email '{identifier}' đã tồn tại!")
            sys.exit(1)
            
        # Hash password
        hashed_password = pwd_context.hash(password)
        
        # We store the identifier in the username column so the Admin can login via the regular /login endpoint
        # If --email was provided, we also try to store it in the email column if possible
        with store.conn.cursor() as cur:
            if args.email:
                cur.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s) RETURNING id;",
                    (identifier, identifier, hashed_password, "admin")
                )
            else:
                cur.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id;",
                    (identifier, hashed_password, "admin")
                )
            result = cur.fetchone()
            admin_id = result[0] if result else None
            
        if admin_id:
            print(f"[+] Thành công! Đã tạo tài khoản Admin (ID: {admin_id}).")
            print(f"    Tài khoản (Username): {identifier}")
            print(f"    Mật khẩu: {password}")
            print(f"    Bây giờ bạn có thể dùng tài khoản này để đăng nhập vào trang Admin.")
        else:
            print("[-] Lỗi: Không thể tạo tài khoản Admin.")
            
    except Exception as e:
        print(f"[-] Lỗi kết nối hoặc thao tác Database: {e}")
    finally:
        if 'store' in locals():
            store.close()

if __name__ == "__main__":
    main()

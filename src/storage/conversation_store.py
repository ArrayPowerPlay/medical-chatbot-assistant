"""This module manages several multi-turn sessions. Each conversation is
identified by an UUID and contains an ordered sequence of messages."""
import uuid
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional, Any

from config.settings import settings
from config.logging_config import logger


class ConversationStore:
    """
    Manages conversations in PostgreSQL.
    """
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                host=settings.POSTGRE_HOST,
                port=settings.POSTGRE_PORT,
                user=settings.POSTGRE_USER,
                password=settings.POSTGRE_PASSWORD,
                dbname=settings.POSTGRE_DB
            )
            self.conn.autocommit = True     # SQL commands are auto-committed
            self._create_tables()
            logger.info("[Conversation Store]: Connected to PostgreSQL successfully!")
        except psycopg2.OperationalError as e:
            logger.error(f"[Conversation Store]: Failed to connect to PostgreSQL db: {e}")
            raise     # 'raise' prevents the error from being ignored and continues to be reported

    def _create_tables(self):
        """Create 'users', 'conversations' and 'messages' tables if not exists."""
        with self.conn.cursor() as cur:    # Create a cursor to give SQL commands
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    role VARCHAR(16) NOT NULL DEFAULT 'user',
                    question_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            
            # try:
            #     cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS username TEXT UNIQUE;")
            #     cur.execute("ALTER TABLE users ALTER COLUMN email DROP NOT NULL;")
            #     cur.execute("ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;")
            # except Exception as e:
            #     logger.warning(f"Failed to alter users table: {e}")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    title       TEXT NOT NULL DEFAULT 'New Chat',
                    is_pinned   BOOLEAN NOT NULL DEFAULT false,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            
            # Alter tables for existing DBs safely
            # try:
            #     cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;")
            #     cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN NOT NULL DEFAULT false;")
            # except Exception as e:
            #     logger.warning(f"Failed to alter conversations table: {e}")
            # SERIAL = auto incremented number
            # ON DELETE CASCADE = messages deleted if corresponding conversation is being deleted
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id               SERIAL PRIMARY KEY,
                    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role             VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
                    content          TEXT NOT NULL,
                    feedback_type    VARCHAR(16) CHECK (feedback_type IN ('like', 'dislike', 'none')),
                    feedback_comment TEXT,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            
            # try:
            #     cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback_type VARCHAR(16) CHECK (feedback_type IN ('like', 'dislike', 'none'));")
            #     cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback_comment TEXT;")
            # except Exception as e:
            #     logger.warning(f"Failed to alter messages table: {e}")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                ON messages(conversation_id, created_at);
            """)

    def create_conversation(self, title: str = "New Chat", user_id: Optional[int] = None) -> str:
        """
        Create a new conversation session.
        Returns:
            str: UUID of the new conversation.
        """
        conv_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id, title, user_id) VALUES (%s, %s, %s);",
                (conv_id, title, user_id),
            ) 
        logger.info(f"[Conversation Store] Created conversation {conv_id} for user {user_id}")
        return conv_id
    
    def conversation_exists(self, conversation_id: str, user_id: Optional[int] = None) -> bool:
        """Check whether a conversation exists."""
        # SELECT 1: return 1 if db can find a suitable row
        with self.conn.cursor() as cur:
            if user_id is not None:
                cur.execute("SELECT 1 from conversations WHERE id = %s AND user_id = %s;", (conversation_id, user_id))
            else:
                cur.execute("SELECT 1 from conversations WHERE id = %s;", (conversation_id,))
            return cur.fetchone() is not None
        
    def update_conversation(self, conversation_id: str, title: Optional[str] = None, is_pinned: Optional[bool] = None) -> None:
        """Update the display title and/or pinned status of a conversation."""
        updates = []
        params = []
        if title is not None:
            updates.append("title = %s")
            params.append(title)
        if is_pinned is not None:
            updates.append("is_pinned = %s")
            params.append(is_pinned)
            
        if not updates:
            return

        updates.append("updated_at = NOW()")
        params.append(conversation_id)
        
        query = f"UPDATE conversations SET {', '.join(updates)} WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to an existing conversation."""
        if role not in ('assistant', 'user'):
            raise ValueError(f"Invalid role: '{role}'. Must be 'user' or 'assistant'!")
        
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s);",
                (conversation_id, role, content),
            )
            cur.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = %s;",
                (conversation_id,),
            )
            
    def add_feedback(self, message_id: int, feedback_type: str, feedback_comment: Optional[str] = None) -> bool:
        """Add feedback to a specific message. Returns True if successful."""
        if feedback_type not in ('like', 'dislike', 'none'):
            raise ValueError(f"Invalid feedback_type: '{feedback_type}'")
            
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE messages SET feedback_type = %s, feedback_comment = %s WHERE id = %s;",
                (feedback_type, feedback_comment, message_id)
            )
            return cur.rowcount > 0

    def get_history(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve the most recent messages in chronological order.
        
        Args: 
            conversation_id: UUID string of the conversation.
            limit: Maximum number of messages to be returned.
        
        Returns:
            List[Dict[str, str]]: Each dict has 'role' and 'content' fields."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if limit is not None:
                query = """
                    SELECT role, content FROM messages
                    WHERE conversation_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
                params = (conversation_id, limit)
                cur.execute(query, params)
                rows = cur.fetchall()
                return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
            else:
                query = """
                    SELECT role, content FROM messages
                    WHERE conversation_id = %s
                    ORDER BY created_at ASC;
                """
                params = (conversation_id,)
                cur.execute(query, params)
                rows = cur.fetchall()
                return [{"role": r["role"], "content": r["content"]} for r in rows]
            
    def list_conversations(self, user_id: Optional[int] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all conversations ordered by pinned status and recently updated."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            query = "SELECT id, title, is_pinned, created_at, updated_at FROM conversations"
            params = []
            
            if user_id is not None:
                query += " WHERE user_id = %s"
                params.append(user_id)
                
            query += " ORDER BY is_pinned DESC, updated_at DESC"
            
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
                
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "is_pinned": r["is_pinned"],
                    "created_at": r["created_at"].isoformat(),
                    "updated_at": r["updated_at"].isoformat()       # convert TIMESTAMPZ to string
                }
                for r in rows
            ]
            
    def search_conversations(self, search_query: str, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search conversations by title or message content."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # We use ILIKE for case-insensitive search
            pattern = f"%{search_query}%"
            query = """
                SELECT DISTINCT c.id, c.title, c.is_pinned, c.created_at, c.updated_at 
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE (c.title ILIKE %s OR m.content ILIKE %s)
            """
            params = [pattern, pattern]
            
            if user_id is not None:
                query += " AND c.user_id = %s"
                params.append(user_id)
                
            query += " ORDER BY c.is_pinned DESC, c.updated_at DESC"
            
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "is_pinned": r["is_pinned"],
                    "created_at": r["created_at"].isoformat(),
                    "updated_at": r["updated_at"].isoformat()
                }
                for r in rows
            ]
        
    def delete_conversation(self, conversation_id: str, user_id: Optional[int] = None) -> bool:
        """Return True if deleted, False if not found."""
        with self.conn.cursor() as cur:
            if user_id is not None:
                cur.execute("DELETE FROM conversations WHERE id = %s AND user_id = %s;", (conversation_id, user_id))
            else:
                cur.execute("DELETE FROM conversations WHERE id = %s;", (conversation_id,))
            return cur.rowcount > 0    # Number of rows affected by the SQL query
        
    def get_message_page(
        self,
        conversation_id: str,
        limit: int = settings.MESSAGE_PAGE_SIZE,
        before_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve a page of messages for display for user, using cursor-based pagination.
        
        Args:
            conversation_id: UUID of the conversation.
            limit: max messages to return.
            before_id: If provided, return messages with id < before_id.

        Returns:
            Dict with:
                - 'messages': List[Dict] each having id, role, content, created_at.
                - 'has_more': bool - True if there are still older messages beyond this page.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Fetch limit + 1 to check if older messages exist
            fetch_count = limit + 1

            if before_id is not None:
                # Scroll-up: load messages older than the cursor
                query = """
                    SELECT id, role, content, created_at FROM messages
                    WHERE conversation_id = %s AND id < %s
                    ORDER BY id DESC
                    LIMIT %s
                """
                cur.execute(query, (conversation_id, before_id, fetch_count))
            else: 
                # First load: get the newest messages
                query = """
                    SELECT id, role, content, created_at FROM messages
                    WHERE conversation_id = %s 
                    ORDER BY id DESC
                    LIMIT %s
                """
                cur.execute(query, (conversation_id, fetch_count))

            rows = cur.fetchall()

            # Older messages exist if 'rows' has more rows than the configure limit constant
            has_more = len(rows) > limit
            rows = rows[:limit]

            # Reverse to chronological order for user experience
            messages = [
                {
                    "id": r["id"],
                    "role": r["role"],
                    "content": r["content"],
                    "created_at": r["created_at"]
                }
                for r in reversed(rows)
            ]

            return {"messages": messages, "has_more": has_more}

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("[Conversation Store]: Connection closed.")
            
    # --- USER MANAGEMENT METHODS ---
    def create_user_with_username(self, username: str, password_hash: str, role: str = "user") -> Optional[int]:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id;",
                (username, password_hash, role)
            )
            result = cur.fetchone()
            return result[0] if result else None

    def create_user_with_email(self, email: str, role: str = "user") -> Optional[int]:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, role) VALUES (%s, %s) RETURNING id;",
                (email, role)
            )
            result = cur.fetchone()
            return result[0] if result else None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s;", (username,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s;", (email,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s;", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None
            
    def increment_question_count(self, user_id: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("UPDATE users SET question_count = question_count + 1 WHERE id = %s;", (user_id,))
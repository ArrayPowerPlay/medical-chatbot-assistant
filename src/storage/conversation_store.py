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
        """Create 'conversations' and 'messages' tables if not exists."""
        with self.conn.cursor() as cur:    # Create a cursor to give SQL commands
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT 'New Chat',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            # SERIAL = auto incremented number
            # ON DELETE CASCADE = messages deleted if corresponding conversation is being deleted
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id               SERIAL PRIMARY KEY,
                    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role             VARCHARS(16) NOT NULL CHECK (role IN ('user', 'assistant')),
                    content          TEXT NOT NULL,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                ON messages(conversation_id, created_at);
            """)

    def create_conversation(self, title: str = "New Chat") -> str:
        """
        Create a new conversation session.
        Returns:
            str: UUID of the new conversation.
        """
        conv_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id, title) VALUES (%s, %s);",
                (conv_id, title),
            ) 
        logger.info(f"[Conversation Store] Created conversation {conv_id}")
        return conv_id
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """Check whether a conversation exists."""
        # SELECT 1: return 1 if db can find a suitable row
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 from conversations WHERE id = %s;", (conversation_id,))
            return cur.fetchone() is not None
        
    def update_title(self, conversation_id: str, title: str) -> None:
        """Update the display title of a conversation."""
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = %s, updated_at = NOW() WHERE id = %s;",
                (title, conversation_id),
            )

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
            
    def list_conversations(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """List all conversations ordered by most recently updated."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if limit is not None:
                query = """
                    SELECT id, title, created_at, updated_at FROM conversations
                    ORDER BY updated_at DESC LIMIT %s;
                """
            else:
                query = """
                    SELECT id, title, create_at, updated_at FROM conversations
                    ORDER BY updated_at DESC;
                """
            cur.execute(query)
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "created_at": r["created_at"].isoformat(),  # Convert datetime object to string
                    "updated_at": r["updated_at"].isoformat()
                }
                for r in rows
            ]
        
    def delete_conversation(self, conversation_id: int) -> bool:
        """Return True if deleted, False if not found."""
        with self.conn.cursor() as cur:
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
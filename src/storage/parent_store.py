import sqlite3
from typing import List, Dict, Optional
from pathlib import Path


class ParentStore:
    """SQLite-based storage for parent chunks. Parents are only looked up by ID"""
    def __init__(self, db_path : str):
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # Return sqlite3.Row object instead of tuple
        # sqlite3.Row object: can act as tuple or dict
        self.conn.row_factory = sqlite3.Row    
        self._create_table()                   # Make sure database has table

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                pmid TEXT NOT NULL,
                title TEXT,
                text TEXT NOT NULL
            )
        """)
        # Create index for 'pmid' for fast retrieval in some cases, for instance
        # retrieve chunks of a article (with the same pmid)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent_pmid
            ON parent_chunks(pmid)
        """)
        self.conn.commit()

    def insert_parents(self, parents: List[Dict]):
        """Batch insert parent chunks. Skips duplicates by using IGNORE"""
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO parent_chunks
                (parent_id, pmid, title, text)
            VALUES
                (:parent_id, :pmid, :title, :text)
            """,
            parents
        )
        self.conn.commit()

    def get_parent(self, parent_id: str) -> Optional[Dict]:
        """Retrieve a single parent chunk by ID"""
        cursor = self.conn.execute(
            "SELECT * FROM parent_chunks WHERE parent_id = ?",
            (parent_id,),                     # Accept list or tuple
        )
        row = cursor.fetchone()               # Retrieve one row
        return dict(row) if row else None
    
    def get_parent_batch(self, parent_ids: List[str]) -> Dict[str, Dict]:
        """Retrieve multiple parents by IDs. Returns {parent_id: data}"""
        if not parent_ids:
            return {}
        
        place_holders = ",".join(["?"] * len(parent_ids))
        cursor = self.conn.execute(
            f"SELECT * FROM parent_chunks WHERE parent_id in ({place_holders})",
            parent_ids
        )

        return {row["parent_id"]: dict(row) for row in cursor.fetchall()}
    
    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM parent_chunks")
        return cursor.fetchone()[0]       # cursor.fetchone() return tuple (count,)
    
    def close(self):
        """Close the database file"""
        self.conn.close()
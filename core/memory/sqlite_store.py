import asyncio
import aiosqlite
import sqlite3 # Import for sqlite3.InterfaceError
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("Sebastian.SQLiteLongTermStore")

DEFAULT_DB_PATH = "core/memory/data/sebastian_memory.db"

class SQLiteLongTermStore:
    """
    SQLite-based long-term memory store.
    This is not a full implementation of MemoryInterface but provides the logic
    that a MemoryManager could use for its long-term operations.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)
        self.conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock() # To protect connection state and schema init
        logger.info(f"SQLiteLongTermStore configured with db_path: {self.db_path}")

    async def _ensure_db_path_exists(self):
        """Ensures the directory for the SQLite DB file exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initializes the database connection and schema."""
        async with self._lock:
            if self.conn and await self._is_connection_valid():
                logger.debug("Database connection already initialized and valid.")
                return

            await self._ensure_db_path_exists()
            try:
                self.conn = await aiosqlite.connect(self.db_path)
                self.conn.row_factory = aiosqlite.Row # Access columns by name
                await self._create_schema()
                logger.info(f"Successfully connected to SQLite database at {self.db_path} and schema initialized.")
            except aiosqlite.Error as e:
                logger.error(f"Failed to initialize SQLite database at {self.db_path}: {e}", exc_info=True)
                self.conn = None # Ensure conn is None if init fails
                raise # Re-raise the exception to signal failure

    async def _is_connection_valid(self) -> bool:
        """Checks if the current connection is valid."""
        if not self.conn:
            return False
        try:
            # Perform a simple query to check the connection
            await self.conn.execute("SELECT 1")
            return True
        except (aiosqlite.OperationalError, sqlite3.InterfaceError, AttributeError): # AttributeError if conn was closed abruptly
            logger.warning("SQLite connection is no longer valid.")
            return False
        
    async def _get_connection(self) -> aiosqlite.Connection:
        """Gets a valid connection, re-initializing if necessary."""
        async with self._lock: # Protects self.conn
            if not self.conn or not await self._is_connection_valid():
                logger.info("Connection invalid or not established. Re-initializing...")
                await self.initialize() # This will re-attempt connection and schema
            if not self.conn: # If initialize failed
                raise aiosqlite.OperationalError("Failed to establish a valid database connection.")
            return self.conn

    async def _create_schema(self) -> None:
        """Creates the necessary tables if they don't exist."""
        if not self.conn:
            logger.error("Cannot create schema, database connection is not established.")
            return

        # Enable WAL mode for better concurrency
        await self.conn.executescript("PRAGMA journal_mode=WAL;")

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                tags_json TEXT, -- Store tags as a JSON array string
                embedding_json TEXT, -- For future semantic search
                created_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW')),
                updated_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
            );
        """)
        # Indexing common query fields
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_user_id ON long_term_memories(user_id);")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_memory_type ON long_term_memories(memory_type);")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_created_at ON long_term_memories(created_at DESC);")
        
        # Trigger to update 'updated_at' timestamp
        await self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS update_ltm_updated_at
            AFTER UPDATE ON long_term_memories
            FOR EACH ROW
            BEGIN
                UPDATE long_term_memories
                SET updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW')
                WHERE id = OLD.id;
            END;
        """)
        await self.conn.commit()
        logger.debug("Long-term memory schema checked/created.")

    async def store(self, user_id: str, memory_type: str, data: Dict[str, Any], tags: Optional[List[str]] = None, embedding: Optional[List[float]] = None) -> str:
        conn = await self._get_connection()
        memory_id = str(uuid.uuid4())
        data_json = json.dumps(data)
        tags_json = json.dumps(tags) if tags else None
        embedding_json = json.dumps(embedding) if embedding else None
        
        try:
            await conn.execute(
                """
                INSERT INTO long_term_memories (id, user_id, memory_type, data_json, tags_json, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, user_id, memory_type, data_json, tags_json, embedding_json)
            )
            await conn.commit()
            logger.info(f"Stored long-term memory '{memory_id}' for user '{user_id}'. Type: {memory_type}")
            return memory_id
        except aiosqlite.Error as e:
            logger.error(f"Failed to store long-term memory for user '{user_id}': {e}", exc_info=True)
            raise

    async def retrieve(
        self, 
        user_id: str, 
        query_text: Optional[str] = None, # For basic keyword search on data_json
        memory_type: Optional[str] = None, 
        tags: Optional[List[str]] = None, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        conn = await self._get_connection()
        sql_query = "SELECT id, user_id, memory_type, data_json, tags_json, embedding_json, created_at, updated_at FROM long_term_memories WHERE user_id = ?"
        params: List[Any] = [user_id]

        if memory_type:
            sql_query += " AND memory_type = ?"
            params.append(memory_type)
        
        if query_text: # Basic keyword search
            sql_query += " AND data_json LIKE ?"
            params.append(f"%{query_text}%") # This is inefficient for large datasets; semantic search is preferred.

        if tags: # This requires more advanced JSON querying or a normalized tags table for efficiency
            # For basic SQLite JSON array search (available in recent versions):
            # This is a simplified example; robust tag searching might need json_each or a separate tags table.
            for tag in tags:
                sql_query += " AND EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)"
                params.append(tag)
                # Or, if tags_json is just a string like '["tag1", "tag2"]':
                # sql_query += " AND INSTR(tags_json, ?) > 0"
                # params.append(f'"{tag}"') # Be careful with quoting and JSON structure

        sql_query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results = []
        try:
            async with conn.execute(sql_query, tuple(params)) as cursor:
                async for row in cursor:
                    data = json.loads(row["data_json"])
                    row_tags = json.loads(row["tags_json"]) if row["tags_json"] else []
                    embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else []
                    results.append({
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "memory_type": row["memory_type"],
                        "data": data,
                        "tags": row_tags,
                        "embedding": embedding,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    })
            logger.debug(f"Retrieved {len(results)} memories for user '{user_id}' with criteria.")
            return results
        except aiosqlite.Error as e:
            logger.error(f"Failed to retrieve long-term memories for user '{user_id}': {e}", exc_info=True)
            return [] # Return empty list on error

    async def get_by_id(self, memory_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = await self._get_connection()
        sql_query = "SELECT id, user_id, memory_type, data_json, tags_json, embedding_json, created_at, updated_at FROM long_term_memories WHERE id = ?"
        params: List[Any] = [memory_id]
        if user_id: # Optional: ensure memory belongs to the user if provided
            sql_query += " AND user_id = ?"
            params.append(user_id)
            
        try:
            async with conn.execute(sql_query, tuple(params)) as cursor:
                row = await cursor.fetchone()
                if row:
                    data = json.loads(row["data_json"])
                    tags = json.loads(row["tags_json"]) if row["tags_json"] else []
                    embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else []
                    return {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "memory_type": row["memory_type"],
                        "data": data,
                        "tags": tags,
                        "embedding": embedding,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    }
            return None
        except aiosqlite.Error as e:
            logger.error(f"Failed to retrieve long-term memory by ID '{memory_id}': {e}", exc_info=True)
            return None

    async def update(self, memory_id: str, data_update: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        conn = await self._get_connection()
        # Fetch existing data to merge, or update specific fields if schema allows partial JSON updates
        current_memory = await self.get_by_id(memory_id, user_id)
        if not current_memory:
            logger.warning(f"Cannot update memory: ID '{memory_id}' not found or user mismatch.")
            return False

        # Merge new data with existing data
        updated_data = current_memory['data']
        updated_data.update(data_update.get('data', {})) # Assuming data_update might contain a 'data' key
        
        updated_tags = data_update.get('tags', current_memory.get('tags'))
        updated_embedding = data_update.get('embedding', current_memory.get('embedding'))

        data_json = json.dumps(updated_data)
        tags_json = json.dumps(updated_tags) if updated_tags is not None else None
        embedding_json = json.dumps(updated_embedding) if updated_embedding is not None else None
        
        try:
            # The trigger will handle updated_at
            cursor = await conn.execute(
                """
                UPDATE long_term_memories 
                SET data_json = ?, tags_json = ?, embedding_json = ?
                WHERE id = ? 
                """,
                (data_json, tags_json, embedding_json, memory_id)
            )
            await conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Updated long-term memory '{memory_id}'.")
                return True
            logger.warning(f"Update for memory ID '{memory_id}' affected 0 rows (possibly already deleted or ID mismatch).")
            return False
        except aiosqlite.Error as e:
            logger.error(f"Failed to update long-term memory '{memory_id}': {e}", exc_info=True)
            return False

    async def delete(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        conn = await self._get_connection()
        sql_query = "DELETE FROM long_term_memories WHERE id = ?"
        params: List[Any] = [memory_id]
        if user_id: # Optional: ensure memory belongs to the user if provided for deletion
            sql_query += " AND user_id = ?"
            params.append(user_id)
            
        try:
            cursor = await conn.execute(sql_query, tuple(params))
            await conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Deleted long-term memory '{memory_id}'.")
                return True
            logger.warning(f"Delete for memory ID '{memory_id}' affected 0 rows (not found or user mismatch).")
            return False
        except aiosqlite.Error as e:
            logger.error(f"Failed to delete long-term memory '{memory_id}': {e}", exc_info=True)
            return False

    async def close(self) -> None:
        """Closes the database connection."""
        async with self._lock:
            if self.conn:
                try:
                    await self.conn.close()
                    logger.info(f"SQLite database connection to {self.db_path} closed.")
                except aiosqlite.Error as e:
                    logger.error(f"Error closing SQLite connection: {e}", exc_info=True)
                finally:
                    self.conn = None # Ensure it's marked as closed
            else:
                logger.debug("No active SQLite connection to close.")

from pathlib import Path # Add Path import if not already at the top of the file
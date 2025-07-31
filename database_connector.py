import json
import os
import asyncpg
from utils import async_retry
from data_classes import ProcessedChunk
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class DatabaseConnector:
    def __init__(self, host, port, user, password, database, min_size, max_size, command_timeout):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.db_pool = None

    async def init_database(self) -> bool:
        """Initialize the PostgreSQL connection pool."""
        # Validate required parameters
        if not all([self.host, self.port, self.user, self.password, self.database]):
            logger.error("Missing required database parameters. Check your .env file.")
            logger.error(f"Host: {self.host}, Port: {self.port}, User: {self.user}, Database: {self.database}")
            return False
            
        try:
            self.db_pool = await asyncpg.create_pool(
                host=self.host,
                port=int(self.port),
                user=self.user,
                password=self.password,
                database=self.database,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout
            )
            
            # Test the connection
            async with self.db_pool.acquire() as connection:
                await connection.fetchval("SELECT 1")
            
            logger.info(f"PostgreSQL pool initialized successfully (host: {self.host}:{self.port})")
            return True
            
        except ValueError as e:
            logger.error(f"Invalid port number: {self.port}")
            return False
        except asyncpg.InvalidCatalogNameError:
            logger.error(f"Database '{self.database}' does not exist")
            return False
        except asyncpg.InvalidPasswordError:
            logger.error("Invalid username or password")
            return False
        except asyncpg.ConnectionDoesNotExistError:
            logger.error(f"Cannot connect to PostgreSQL server at {self.host}:{self.port}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            logger.error(f"Connection details: {self.host}:{self.port}/{self.database} as {self.user}")
            self.db_pool = None
            return False
    
    async def close_database(self):
        """Close the PostgreSQL connection pool."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Pool de conexões PostgreSQL finalizado")
    
    @async_retry()
    async def remove_page_chunks(self, database_id: str, page_id: str, model: str):
        """Delete all existing chunks for a page before re-processing."""
        if self.db_pool is None:
            logger.info(f"Cannot delete chunks for page {page_id} due to database connection issue.")
            return
        
        try:
            async with self.db_pool.acquire() as connection:
                query = "DELETE FROM notion.ntn_page_content_embeddings WHERE database_id = $1 AND page_id = $2 AND model = $3"
                result = await connection.execute(query, database_id, page_id, model)
                
                # Extract deleted count from result string like "DELETE 5"
                deleted_count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                logger.info(f"Deleted {deleted_count} existing chunks for page {page_id}")
                return deleted_count
        except Exception as e:
            logger.info(f"Error deleting chunks for page {page_id}: {e}")
            return None
    
    @async_retry()
    async def insert_chunk(self, chunk: ProcessedChunk):
        """Insert a processed chunk into PostgreSQL."""
        if self.db_pool is None:
            logger.info(f"Skipping insertion for chunk {chunk.chunk_number} for page {chunk.page_id} due to database connection issue.")
            return None
        
        vector_dimension = os.getenv("VECTOR_DIMENSION",1536)
        if len(chunk.embedding) != vector_dimension:
            logger.info(f"Original embedding length: {len(chunk.embedding)}, converting to {vector_dimension} dimensions")
            padded_embedding_vector = chunk.embedding + [0] * (vector_dimension - len(chunk.embedding))
        else:
            padded_embedding_vector = chunk.embedding

        try:
            async with self.db_pool.acquire() as connection:
                # Convert embedding list to PostgreSQL vector format
                # PostgreSQL vector extension expects the vector as a string in format: '[1.0, 2.0, 3.0]'
                embedding_str = '[' + ','.join(map(str, padded_embedding_vector)) + ']'
                
                # Parse last_edited_time if it's a string
                last_edited_time = chunk.last_edited_time
                if isinstance(last_edited_time, str):
                    if last_edited_time:  # Only parse if not empty
                        last_edited_time = datetime.fromisoformat(last_edited_time.replace('Z', '+00:00'))
                    else:
                        # If empty string, use current time
                        last_edited_time = datetime.now(timezone.utc)
                
                query = """
                    INSERT INTO notion.ntn_page_content_embeddings
                        (
                        page_id,
                        page_title,
                        database_id,
                        database_title,
                        url,
                        title, 
                        summary,
                        chunk_number,
                        content,
                        model,
                        embedding,
                        metadata,
                        last_edited_time
                        ) 
                    VALUES 
                        (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10,
                        $11,
                        $12,
                        $13
                        )
                    RETURNING id
                """
                
                result = await connection.fetchrow(
                    query,
                    chunk.page_id,
                    chunk.page_title,
                    chunk.database_id,
                    chunk.database_title,
                    chunk.url,
                    chunk.title,
                    chunk.summary,
                    chunk.chunk_number,
                    chunk.content,
                    chunk.model,
                    embedding_str,
                    json.dumps(chunk.metadata),
                    last_edited_time
                )
                
                logger.info(f"Inserted chunk {chunk.chunk_number} for page {chunk.page_id}")
                return result
                
        except Exception as e:
            logger.info(f"Error inserting chunk: {e}")
            logger.info(f"Error type: {type(e).__name__}")
            if "connection" in str(e).lower():
                logger.info("Database connection failed - check your PostgreSQL credentials")
            return None


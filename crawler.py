import os
import asyncio
import pandas as pd
from data_classes import NotionPage, ProcessedChunk, CrawlerDeps, GlobalDeps
from database_connector import DatabaseConnector
from notion_connector import NotionConnector
from datetime import datetime, timezone
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class Crawler:
    def __init__(self, provider: str, model: str, deps: CrawlerDeps, global_deps: GlobalDeps):
        from ai_providers import get_embedding_model
        # Parse the embedding model string to get provider and model
        # Assuming format like "huggingface:all-MiniLM-L6-v2"
        self.embedding_model = get_embedding_model(provider, model)
        self.notion_connector = NotionConnector(notion_api_key=os.getenv("NOTION_API_KEY"))
        self.database_connector = DatabaseConnector(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DATABASE"),
            min_size=1,
            max_size=10,
            command_timeout=60
        )
        self.deps = deps
        self.global_deps = global_deps
    
    async def run_crawler(self, pages: [NotionPage]) -> list:
        try:
            # Initialize database connection
            logger.info("Initializing database connection...")
            success = await self.database_connector.init_database()
            if not success:
                logger.error("Failed to connect to PostgreSQL database")
                logger.error(f"DB Config: host={os.getenv('POSTGRES_HOST')}, port={os.getenv('POSTGRES_PORT')}, user={os.getenv('POSTGRES_USER')}, database={os.getenv('POSTGRES_DATABASE')}")
                return []
            else:
                logger.info("Database connection successful")        
            # Delegate to the coordinator for actual processing
            
            if not pages:
                logger.info("No pages to process")
                return []
            
            # Determine which pages actually need updating (only if DB connection succeeded)
            if success:
                pages_to_process = await self.filter_pages_to_process(pages)
            else:
                logger.info("Database connection failed. Processing all pages without update check.")
                pages_to_process = pages
            if not pages_to_process:
                logger.info("All pages are up to date!")
                return []
            
            # Create concurrency limiter
            concurrency_limiter = asyncio.Semaphore(self.deps.max_concurrent_chunks)
            
            # Create processing tasks for each page
            processing_tasks = [
                self.process_page(page) 
                for page in pages_to_process
            ]
            
            # Execute all page processing in parallel and collect results
            logger.info(f"Starting processing of {len(processing_tasks)} pages")
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Log any exceptions that occurred
            exceptions = [r for r in results if isinstance(r, Exception)]
            if exceptions:
                logger.warning(f"Encountered {len(exceptions)} exceptions during processing")
                for i, exc in enumerate(exceptions):
                    logger.warning(f"Task {i} failed: {exc}")
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            logger.info(f"Successfully processed {len(successful_results)} pages")
            return results
        
        except Exception as e:
            logger.info(f"Erro durante processamento Zaub RAG: {e}")
            return []
        
        finally:
            # Finaliza conexão com banco de dados
            await self.database_connector.close_database()
    
    async def filter_pages_to_process(self, pages: List[NotionPage]) -> List[NotionPage]:
        """
        DATABASE QUERY FILTER: Determines which pages need processing based on freshness.
        
        Responsibilities:
        - Query database for existing page timestamps
        - Compare with current page last_edited_time
        - Filter out pages that are already up-to-date
        - Return only pages that need processing
        
        Args:
            pages: All pages to check
            
        Returns:
            List of pages that need processing (new or outdated)
        """
        if self.database_connector.db_pool is None:
            logger.info("Cannot check for updates due to database connection issue. Processing all pages.")
            return pages
        
        try:
            async with self.database_connector.db_pool.acquire() as connection:
                pages_to_process = []
                
                for page in pages:
                    # Parse last_edited_time if it's a string
                    last_edited_time = page.last_edited_time
                    if isinstance(last_edited_time, str):
                        if last_edited_time:  # Only parse if not empty
                            last_edited_time = datetime.fromisoformat(last_edited_time.replace('Z', '+00:00'))
                        else:
                            # If empty string, use current time
                            last_edited_time = datetime.now(timezone.utc)
                    
                    # Check if page exists and if it needs updating
                    query = """
                        SELECT 
                            CASE 
                                WHEN NOT EXISTS (
                                    SELECT 1 FROM notion.ntn_page_content_embeddings 
                                    WHERE database_id = $1 AND page_id = $2 and model = $3
                                ) THEN true  -- Page doesn't exist, needs processing
                                WHEN EXISTS (
                                    SELECT 1 FROM notion.ntn_page_content_embeddings 
                                    WHERE database_id = $1 AND page_id = $2 and model = $3
                                    AND (last_edited_time IS NULL OR last_edited_time < $4)
                                ) THEN true  -- Page exists but is outdated
                                ELSE false  -- Page is up to date
                            END as needs_update
                    """
                    
                    result = await connection.fetchval(
                        query, 
                        page.database_id, 
                        page.page_id, 
                        self.embedding_model.model_identifier,
                        last_edited_time
                    )
                    
                    if result:
                        pages_to_process.append(page)
                    
                    notion_delay = self.deps.delay_seconds
                    if notion_delay > 0:
                        await asyncio.sleep(notion_delay)
                
                logger.info(f"Update check: {len(pages_to_process)} of {len(pages)} pages need processing")
                return pages_to_process
            
        except Exception as e:
            logger.info(f"Error checking for updates: {e}")
            logger.info("Processing all pages as fallback.")
            return pages
    
    async def process_page(self, page: NotionPage):
        """
        SINGLE PAGE PROCESSOR: Handles the complete workflow for one Notion page.
        
        Responsibilities:
        - Acquire concurrency slot (respects global limits)
        - Delete old chunks from database
        - Fetch fresh content from Notion API
        - Delegate to chunk processor for content analysis
        - Handle page-level errors and rate limiting
        - Release concurrency slot when done
        
        Args:
            page_info: Structured information about the page to process
            concurrency_limiter: Semaphore to control concurrent page processing
        """

        logger.info(f"Processing page: {page.page_title}")
        
        # Step 1: Clean up old data
        await self.database_connector.remove_page_chunks(page.database_id, page.page_id, self.embedding_model.model_identifier)
        
        # Step 2: Get fresh content from Notion
        page_content = await self.notion_connector.retrieve_notion_page_content(page)
        logger.debug(page_content.page_title)
        
        # Step 3: Process content into chunks and store
        result = await self.process_page_content(page_content)

        if len(result.chunks) > 0:
            return result
        else:
            return false
    
    async def process_page_content(self, page: NotionPage):
    
        if self.database_connector.db_pool is None:
            logger.error(f"Skipping page processing for {page.page_id} due to database connection issue.")
            return
        
        # Step 1: Split content into manageable chunks
        text_chunks = self.split_content_into_chunks(page.content)

        for chunk_number, chunk_text in enumerate(text_chunks, 1):
            logger.debug(f"Processing chunk {chunk_number}")
            # Get embedding asynchronously
            embedding = await self.embedding_model.get_embedding(chunk_text)

            logger.debug(embedding)

            # Generate basic title and summary for the chunk
            # You can replace this with AI-generated content if needed
            chunk_title = f"Chunk {chunk_number} - {page.page_title}"
            chunk_summary = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text

            metadata = {
                    "source": "notion",
                    "chunk_size": len(chunk_text),
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "page_title": page.page_title,
                    "database_title": page.database_title,
                    "database_id": page.database_id,
                    "page_id": page.page_id,
                    "url": page.url
                }
            
            chunk = ProcessedChunk(
                page_id=page.page_id,
                page_title=page.page_title,
                database_id=page.database_id,
                database_title=page.database_title,
                url=page.url,
                title=chunk_title,
                summary=chunk_summary,
                chunk_number=chunk_number,
                model=self.embedding_model.model_identifier,
                content=chunk_text,
                embedding=embedding,
                metadata=metadata,
                last_edited_time=page.last_edited_time
            )
            logger.debug(chunk.title)
            page.chunks.append(chunk)

        persistence_tasks = []
        for chunk in page.chunks:
            if isinstance(chunk, ProcessedChunk):
                persistence_tasks.append(self.database_connector.insert_chunk(chunk))
        
        if persistence_tasks:
            await asyncio.gather(*persistence_tasks, return_exceptions=True)
            logger.info(f"Processed and stored {len(persistence_tasks)} chunks for page {page.page_title}")
            return page
        else:
            logger.info(f"No chunks were successfully processed for page {page.page_title}")

            return False
    
    def split_content_into_chunks(self, text: str) -> List[str]:
        """
        TEXT SPLITTER: Intelligently splits text into chunks while preserving structure.
        
        Responsibilities:
        - Split text into chunks of approximately chunk_size characters
        - Respect code block boundaries (```)
        - Respect paragraph boundaries (\n\n) 
        - Respect sentence boundaries (. )
        - Ensure no chunk is too small to be meaningful
        - Create overlapping chunks for better context preservation
        
        Args:
            text: Raw text content to split
            overlap: Number of characters to overlap between chunks (default: 100)
            
        Returns:
            List of text chunks ready for AI processing
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + self.deps.chunk_size

            # If we're at the end of the text, just take what's left
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to find a code block boundary first (```)
            chunk = text[start:end]
            code_block = chunk.rfind('```')
            if code_block != -1 and code_block > self.deps.chunk_size * 0.3:
                end = start + code_block

            # If no code block, try to break at a paragraph
            elif '\n\n' in chunk:
                # Find the last paragraph break
                last_break = chunk.rfind('\n\n')
                if last_break > self.deps.chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                    end = start + last_break

            # If no paragraph break, try to break at a sentence
            elif '. ' in chunk:
                # Find the last sentence break
                last_period = chunk.rfind('. ')
                if last_period > self.deps.chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                    end = start + last_period + 1

            # Extract chunk and clean it up
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position for next chunk with overlap
            start = max(start + 1, end - self.deps.chunk_overlap)

        return chunks
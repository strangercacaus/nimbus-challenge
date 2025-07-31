from dataclasses import dataclass
from dotenv import load_dotenv
import os
import logging

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from typing import List, Union
import asyncpg
from ai_providers import get_embedding_model, get_pydantic_model
from database_connector import DatabaseConnector
from data_classes import AgentDeps, GlobalDeps

logger = logging.getLogger(__name__)

load_dotenv()

class ZaubAgent:
    def __init__(
            self,
            embedding_model: str,
            pydantic_model: str,
            deps: AgentDeps,
            global_deps: GlobalDeps
        ):
        self.deps = deps
        self.global_deps = global_deps
        # Parse the embedding model string to get provider and model
        # Assuming format like "huggingface:all-MiniLM-L6-v2"
        provider, model = embedding_model.split(':', 1)
        self.embedding_model = get_embedding_model(provider, model)
        
        # Parse the pydantic model string to get provider and model
        # Assuming format like "openai:gpt-4o-mini"
        pydantic_provider, pydantic_model_name = pydantic_model.split(':', 1)
        pydantic_model_obj = get_pydantic_model(pydantic_provider, pydantic_model_name)
        
        self.agent = Agent(
            pydantic_model_obj,
            system_prompt=deps.system_prompt,
            deps_type=AgentDeps,
            retries=deps.max_retries
        )
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
        # Register tools after agent creation
        self._register_tools()

    async def initialize(self):
        """Initialize the database connection"""
        success = await self.database_connector.init_database()
        if not success:
            raise RuntimeError("Failed to initialize database connection")
        return success

    async def run(self, user_query: str):
        """Run the agent with a user query"""
        return await self.agent.run(user_query, deps=self.deps)
    
    def run_stream(self, user_query: str, **kwargs):
        """Run the agent with streaming (for supported models)"""
        return self.agent.run_stream(user_query, deps=self.deps, **kwargs)
    
    async def close(self):
        """Close database connections and cleanup resources"""
        await self.database_connector.close_database()

    async def get_embedding(self,text: str) -> List[float]:
        """Get embedding vector using the configured AI provider.
        args:
            text: The text to get the embedding for
            ai_provider: The AI provider to use for the embedding
        returns:
            List[float]: The embedding vector
        """
        try:
            # For anthropic_model, we need to use OpenAI for embeddings since Anthropic doesn't provide embeddings
            return await self.embedding_model.get_embedding(text)
        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            return [0] * self.vector_dimension  # Return zero vector on error

    def _register_tools(self):
        """Register all tool methods with the agent"""
        self.agent.tool(self.retrieve_relevant_documentation)
        self.agent.tool(self.list_documentation_pages)
        self.agent.tool(self.get_page_content)
        self.agent.tool(self.search_by_url)

    async def retrieve_relevant_documentation(self, ctx: RunContext[AgentDeps], user_query: str) -> str:
        """
        Retrieve relevant documentation chunks based on the query with RAG using PostgreSQL.
        
        Args:
            user_query: The user's question or query
            
        Returns:
            A formatted string containing the most relevant documentation chunks
        """
        try:
            # Get the embedding for the query
            query_embedding = await self.get_embedding(user_query)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Query PostgreSQL for relevant documents using cosine similarity
            async with self.database_connector.db_pool.acquire() as connection:
                query = """
                    SELECT 
                        title,
                        page_title,
                        summary,
                        content,
                        database_title,
                        page_id,
                        url,
                        (embedding <=> $1::vector) as distance
                    FROM notion.ntn_page_content_embeddings 
                    WHERE model = $2
                    ORDER BY embedding <=> $1::vector 
                    LIMIT $3
                """
                
                results = await connection.fetch(query, embedding_str, self.embedding_model.model_identifier, self.deps.max_chunks)
                
                if not results:
                    return "No relevant documentation found."
                    
                # Format the results
                formatted_chunks = []
                for doc in results:
                    similarity_score = 1 - doc['distance']  # Convert distance to similarity
                    chunk_text = f"""
    # {doc['title']}

    **Página**: {doc['page_title']}
    **Database**: {doc['database_title']}

    {doc['content']}

    **URL**: {doc['url'] or 'N/A'}
    **Page ID**: {doc['page_id']}
    **Relevance Score**: {similarity_score:.4f}
    """
                    formatted_chunks.append(chunk_text)
                    
                # Join all chunks with a separator
                return "\n\n---\n\n".join(formatted_chunks)
            
        except Exception as e:
            logger.info(f"Error retrieving documentation: {e}")
            return f"Error retrieving documentation: {str(e)}"

    async def list_documentation_pages(self, ctx: RunContext[AgentDeps]) -> List[str]:
        """
        Retrieve a list of all available Zaub documentation pages.
        
        Returns:
            List[str]: List of unique page titles and their database sources with URLs
        """
        try:
            async with self.database_connector.db_pool.acquire() as connection:
                query = """
                    SELECT DISTINCT 
                        page_title,
                        database_title,
                        page_id,
                        url
                    FROM notion.ntn_page_content_embeddings 
                    WHERE model = $1
                    ORDER BY database_title, page_title
                """
                
                results = await connection.fetch(query, self.embedding_model.model_identifier)
                
                if not results:
                    return []
                    
                # Format as readable list
                pages = []
                for doc in results:
                    url_info = f" - {doc['url']}" if doc['url'] else ""
                    pages.append(f"{doc['database_title']}: {doc['page_title']} (ID: {doc['page_id']}){url_info}")
                
                return pages
            
        except Exception as e:
            logger.info(f"Error retrieving documentation pages: {e}")
            return []

    async def get_page_content(self, ctx: RunContext[AgentDeps], page_id: str) -> str:
        """
        Retrieve the full content of a specific documentation page by combining all its chunks.
        
        Args:
            page_id: The page ID to retrieve
            
        Returns:
            str: The complete page content with all chunks combined in order
        """
        try:
            async with self.database_connector.db_pool.acquire() as connection:
                query = """
                    SELECT 
                        title,
                        page_title,
                        content,
                        chunk_number,
                        database_title,
                        summary,
                        url
                    FROM notion.ntn_page_content_embeddings 
                    WHERE page_id = $1
                    AND model = $2
                    ORDER BY chunk_number
                """
                
                results = await connection.fetch(query, page_id, self.embedding_model.model_identifier)
                
                if not results:
                    return f"No content found for page ID: {page_id}"
                    
                # Format the page with its title and all chunks
                first_chunk = results[0]
                page_title = first_chunk['page_title']  # Original page title
                database_title = first_chunk['database_title']
                url = first_chunk['url']
                
                formatted_content = [f"# {page_title}\n"]
                formatted_content.append(f"**Database**: {database_title}")
                if url:
                    formatted_content.append(f"**URL**: {url}")
                formatted_content.append("")  # Empty line
                
                # Add each chunk's content with AI-generated titles
                for chunk in results:
                    if chunk['chunk_number'] > 1:
                        formatted_content.append(f"## {chunk['title']}")  # AI-generated title for chunk
                    formatted_content.append(chunk['content'])
                    formatted_content.append("")  # Empty line between chunks
                    
                # Join everything together
                return "\n".join(formatted_content)
            
        except Exception as e:
            logger.info(f"Error retrieving page content: {e}")
            return f"Error retrieving page content: {str(e)}"

    async def search_by_url(self, ctx: RunContext[AgentDeps], url: str) -> str:
        """
        Search for documentation by URL.
        
        Args:
            url: The URL to search for
            
        Returns:
            str: The complete page content for the given URL
        """
        try:
            async with self.database_connector.db_pool.acquire() as connection:
                query = """
                    SELECT 
                        title,
                        page_title,
                        content,
                        chunk_number,
                        database_title,
                        summary,
                        page_id
                    FROM notion.ntn_page_content_embeddings 
                    WHERE url = $1
                    AND model = $2
                    ORDER BY chunk_number
                """
                
                results = await connection.fetch(query, url, self.embedding_model.model_identifier)
                
                if not results:
                    return f"No content found for URL: {url}"
                    
                # Format the page with its title and all chunks
                first_chunk = results[0]
                page_title = first_chunk['page_title']  # Original page title
                database_title = first_chunk['database_title']
                page_id = first_chunk['page_id']
                
                formatted_content = [f"# {page_title}\n"]
                formatted_content.append(f"**Database**: {database_title}")
                formatted_content.append(f"**URL**: {url}")
                formatted_content.append(f"**Page ID**: {page_id}")
                formatted_content.append("")  # Empty line
                
                # Add each chunk's content with AI-generated titles
                for chunk in results:
                    if chunk['chunk_number'] > 1:
                        formatted_content.append(f"## {chunk['title']}")  # AI-generated title for chunk
                    formatted_content.append(chunk['content'])
                    formatted_content.append("")  # Empty line between chunks
                    
                # Join everything together
                return "\n".join(formatted_content)
            
        except Exception as e:
            logger.info(f"Error searching by URL: {e}")
            return f"Error searching by URL: {str(e)}"
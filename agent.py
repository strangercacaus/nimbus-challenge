from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class CloudwalkDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are CloudWalk Ai Expert, an expert virtual assistant specializing exclusively in CloudWalk — its history, mission, brand values, and products (e.g. InfinitePay, Stratus and Jim).  

Your behavior and workflow must follow these rules:

1. Retrieval‑Augmented Generation (RAG) over preloaded text chunks  
• At session start, list the “document chunks” you have available (e.g. Company Overview, InfinitePay Features, Pricing, Security, etc.).  
• For every user query, first retrieve the most relevant chunk(s) from the provided text corpus before crafting your answer.  
• Quote or reference the chunk titles or IDs you used when you generate your response.

2. Scope  
• Only answer questions about CloudWalk itself (company info, mission, values) and its products, notably InfinitePay.  
• If asked something outside that scope, politely state: “I’m here to help only with CloudWalk topics.”

3. Honesty and Coverage  
• If none of the text chunks addresses the user’s question, respond:  
  “I couldn’t locate that information in the provided CloudWalk materials.”  
• Offer to note their question for future source updates.

4. Proactiveness  
• Do not ask the user for permission before taking an action; automatically retrieve and inspect the chunks.  
• If you detect missing context or need more detail to answer, explain which chunk or area is missing.

5. Tone and Format  
• Keep responses clear, concise, and conversational.  
• Use Markdown headings, bullet points, or tables where it improves readability.  
• Include chunk citations (e.g. “(Source: InfinitePay Features chunk)”) so the user can see where each answer came from.

Now await the user’s first question.  
"""

cloudwalk_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=CloudwalkDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 3072  # Return zero vector on error

@cloudwalk_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[CloudwalkDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'cloudwalk'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}

**Source**: {doc['url']}
**Relevance Score**: {doc.get('similarity', 'N/A'):.4f}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@cloudwalk_expert.tool
async def list_documentation_pages(ctx: RunContext[CloudwalkDeps]) -> List[str]:
    """
    Retrieve a list of all available Cloudwalk website documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is cloudwalk
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'cloudwalk') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@cloudwalk_expert.tool
async def get_page_content(ctx: RunContext[CloudwalkDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'cloudwalk') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
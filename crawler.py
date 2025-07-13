import os
import sys
import json
import asyncio
import requests
import time
import random
from xml.etree import ElementTree
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse, quote, unquote
import re
from dotenv import load_dotenv
from functools import wraps

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Configuration for retry logic and rate limiting
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
OPENAI_BASE_DELAY = float(os.getenv("OPENAI_BASE_DELAY", "1.0"))
SUPABASE_MAX_RETRIES = int(os.getenv("SUPABASE_MAX_RETRIES", "3"))
SUPABASE_BASE_DELAY = float(os.getenv("SUPABASE_BASE_DELAY", "2.0"))
MAX_CONCURRENT_CHUNKS = int(os.getenv("MAX_CONCURRENT_CHUNKS", "3"))
CRAWL_DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "0.5"))

def async_retry(max_retries: int = 5, base_delay: float = 5.0, backoff_factor: float = 2.0):
    """
    Async retry decorator with exponential backoff for handling rate limits and timeouts.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        backoff_factor: Multiplier for delay on each retry
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this is a retryable error
                    error_str = str(e).lower()
                    is_rate_limit = any(keyword in error_str for keyword in [
                        'rate limit', 'rate_limit', 'too many requests', 
                        'quota exceeded', 'rate exceeded', '429'
                    ])
                    is_timeout = any(keyword in error_str for keyword in [
                        'timeout', 'timed out', 'connection timeout',
                        'read timeout', 'request timeout'
                    ])
                    is_server_error = any(keyword in error_str for keyword in [
                        'internal server error', '500', '502', '503', '504',
                        'bad gateway', 'service unavailable', 'gateway timeout'
                    ])
                    is_connection_error = any(keyword in error_str for keyword in [
                        'connection error', 'connection refused', 'connection reset',
                        'connection timeout', 'network error'
                    ])
                    
                    # Only retry on specific error types
                    if not (is_rate_limit or is_timeout or is_server_error or is_connection_error):
                        print(f"❌ Non-retryable error in {func.__name__}: {e}")
                        raise e
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff and jitter
                        delay = base_delay * (backoff_factor ** attempt)
                        jitter = random.uniform(0.1, 0.5) * delay
                        total_delay = delay + jitter
                        
                        error_type = "rate limit" if is_rate_limit else "timeout/connection" if (is_timeout or is_connection_error) else "server error"
                        print(f"⏳ {error_type.capitalize()} in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). Retrying in {total_delay:.2f}s...")
                        
                        await asyncio.sleep(total_delay)
                    else:
                        print(f"❌ Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase client with error handling
supabase = None
try:
    supabase: Client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    print("✅ Supabase connection successful")
except Exception as e:
    print(f"⚠️  Supabase connection failed: {e}")
    print("   Continuing without database storage...")
    print("   Content will be processed but not stored.")
    print("   Please check your SUPABASE_URL and SUPABASE_SERVICE_KEY in .env file")
    supabase = None

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def clean_url(url: str) -> str:
    """Clean and normalize URL to fix encoding issues."""
    if not url:
        return ""
    
    # Remove all whitespace and control characters
    url = re.sub(r'\s+', '', url)  # Remove all whitespace
    url = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', url)  # Remove control characters
    
    # Handle Unicode normalization
    url = url.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Ensure proper URL encoding for special characters
    try:
        # Parse the URL to handle encoding properly
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            # Reconstruct the URL with proper encoding
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                url += f"?{parsed.query}"
            if parsed.fragment:
                url += f"#{parsed.fragment}"
    except Exception:
        pass  # If parsing fails, use the cleaned URL as-is
    
    return url.strip()

def validate_url(url: str) -> bool:
    """Validate URL format."""
    if not url:
        return False
    
    # Check if URL starts with valid protocols
    valid_prefixes = ['http://', 'https://', 'file://', 'raw:']
    if not any(url.startswith(prefix) for prefix in valid_prefixes):
        return False
    
    # Additional validation
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

@async_retry(max_retries=OPENAI_MAX_RETRIES, base_delay=OPENAI_BASE_DELAY, backoff_factor=2.0)
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

@async_retry(max_retries=OPENAI_MAX_RETRIES, base_delay=OPENAI_BASE_DELAY, backoff_factor=2.0)
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model = os.getenv("EMBEDDING_MODEL"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 3072  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": os.getenv("SITE_URL").split(".")[1],
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

@async_retry(max_retries=SUPABASE_MAX_RETRIES, base_delay=SUPABASE_BASE_DELAY, backoff_factor=2.0)
async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    if supabase is None:
        print(f"⚠️  Skipping insertion for chunk {chunk.chunk_number} for {chunk.url} due to Supabase connection issue.")
        return None

    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Supabase URL: {os.getenv('SUPABASE_URL')}")
        if "nodename nor servname provided" in str(e):
            print("❌ DNS resolution failed - check your SUPABASE_URL in .env file")
            print("   Visit https://supabase.com/dashboard to verify your project URL")
        return None

async def process_and_store_document(url: str, markdown: str, metadata: Dict[str, Any]):
    """Process a document and store its chunks in parallel."""
    if supabase is None:
        print(f"⚠️  Skipping document processing for {url} due to Supabase connection issue.")
        return
    
    # Split into chunks
    chunks = chunk_text(markdown, chunk_size=3000)
    
    # Process chunks with controlled concurrency to avoid rate limits
    # Each chunk makes 2 OpenAI API calls (title/summary + embedding)
    max_concurrent_chunks = MAX_CONCURRENT_CHUNKS  # Limit to avoid rate limits
    semaphore = asyncio.Semaphore(max_concurrent_chunks)
    
    async def process_chunk_with_limit(chunk: str, chunk_number: int):
        async with semaphore:
            return await process_chunk(chunk, chunk_number, url)
    
    # Process chunks with rate limiting
    tasks = []
    for i, chunk in enumerate(chunks):
        task = process_chunk_with_limit(chunk, i+1)
        tasks.append(task)
    
    # Wait for all chunks to be processed
    processed_chunks = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Insert all chunks in parallel
    insertion_tasks = []
    for chunk in processed_chunks:
        if isinstance(chunk, ProcessedChunk):
            insertion_tasks.append(insert_chunk(chunk))
    
    if insertion_tasks:
        await asyncio.gather(*insertion_tasks, return_exceptions=True)
        print(f"📝 Processed and stored {len(insertion_tasks)} chunks for {url}")
    else:
        print(f"⚠️  No chunks were successfully processed for {url}")

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with semaphore control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Browser configuration for crawl4ai
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False
    )
    
    # Crawl configuration
    crawl_config = CrawlerRunConfig(
        word_count_threshold=200,
        css_selector=None,
        screenshot=False,
        user_agent="Mozilla/5.0 (compatible; RAG-AI-Agent/1.0)",
        verbose=False,
        delay_before_return_html=2.0,
        remove_overlay_elements=True,
        simulate_user=True,
        override_navigator=True,
        magic=True
    )
    
    # Create the crawler (opens the browser)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        async def process_url(url: str):
            async with semaphore:
                # Clean and validate URL before crawling
                original_url = url
                url = clean_url(url)
                
                if not validate_url(url):
                    print(f"❌ Skipping invalid URL:")
                    print(f"   Original: '{original_url}' (length: {len(original_url)})")
                    print(f"   Cleaned:  '{url}' (length: {len(url)})")
                    print(f"   Bytes: {[hex(ord(c)) for c in original_url[:50]]}")
                    return
                
                print(f"🔍 Crawling: {url}")
                
                try:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    if result.success:
                        print(f"✅ Successfully crawled: {url}")
                        await process_and_store_document(url, result.markdown.raw_markdown, result.metadata)
                        
                        # Add delay after processing to be respectful of rate limits
                        if CRAWL_DELAY_SECONDS > 0:
                            await asyncio.sleep(CRAWL_DELAY_SECONDS)
                    else:
                        print(f"❌ Failed to crawl: {url}")
                        print(f"   Error: {result.error_message}")
                except Exception as e:
                    print(f"❌ Exception crawling {url}: {e}")
        
        # Process all URLs
        tasks = [process_url(url) for url in urls]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    finally:
        # After all URLs are done, close the crawler (and the browser)
        await crawler.close()

def get_ai_docs_urls() -> List[str]:
    """Get URLs from docs sitemap."""
    sitemap_url = os.getenv("SITEMAP_URL")
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap and clean them
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        raw_urls = [loc.text for loc in root.findall('.//ns:loc', namespace) if loc.text]
        
        # Clean and validate URLs
        cleaned_urls = []
        for raw_url in raw_urls:
            cleaned_url = clean_url(raw_url)
            if validate_url(cleaned_url):
                cleaned_urls.append(cleaned_url)
            else:
                print(f"⚠️  Skipped invalid URL from sitemap:")
                print(f"   Raw: '{raw_url}'")
                print(f"   Cleaned: '{cleaned_url}'")
        
        print(f"📊 Sitemap processing: {len(raw_urls)} raw URLs → {len(cleaned_urls)} valid URLs")
        return cleaned_urls
    except Exception as e:
        print(f"❌ Error fetching sitemap: {e}")
        return []

async def main():
    # Get URLs from docs
    print("🔄 Fetching URLs from sitemap...")
    urls = get_ai_docs_urls()
    if not urls:
        print("❌ No URLs found to crawl")
        return
    
    print(f"🎯 Found {len(urls)} valid URLs to crawl")
    
    # Show first few URLs for debugging
    if urls:
        print("📋 Sample URLs:")
        for i, url in enumerate(urls[:5]):  # Show 5 instead of 3
            print(f"   {i+1}. {url}")
            print(f"      Length: {len(url)} characters")
        if len(urls) > 5:
            print(f"   ... and {len(urls) - 5} more URLs")
    
    print(f"\n🚀 Starting parallel crawl...")
    await crawl_parallel(urls)
    print(f"✅ Crawling complete!")

if __name__ == "__main__":
    asyncio.run(main())

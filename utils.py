import os
import re
import sys
import json
import random
import backoff
import asyncio
import nest_asyncio
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
from functools import wraps
import logging

logger = logging.getLogger(__name__)

df_people = pd.read_csv("df_pages.csv")
df_pages = pd.read_csv("df_people.csv")

def async_retry(max_retries: int = 5, base_delay: float = 5.0, backoff_factor: float = 2.0):
    """
    Async retry decorator with exponential backoff for handling rate limits and timeouts.
    """
    def decorator(func):
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
                        logger.info(f"Non-retryable error in {func.__name__}: {e}")
                        raise e
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff and jitter
                        delay = base_delay * (backoff_factor ** attempt)
                        jitter = random.uniform(0.1, 0.5) * delay
                        total_delay = delay + jitter
                        
                        error_type = "rate limit" if is_rate_limit else "timeout/connection" if (is_timeout or is_connection_error) else "server error"
                        logger.info(f"⏳ {error_type.capitalize()} in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). Retrying in {total_delay:.2f}s...")
                        
                        await asyncio.sleep(total_delay)
                    else:
                        logger.info(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator

def get_notion_username_from_dw(person_id):
    """Get Notion username from DW."""
    return df_people[df_people["person_id"] == person_id]["person_name"].values[0]

def get_notion_page_name_from_dw(page_id):
    """Get Notion username from DW."""
    return df_pages[df_pages["page_id"] == page_id]["page_title"].values[0]
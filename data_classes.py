from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import logging
logger = logging.getLogger(__name__)

@dataclass
class ProcessedChunk:
    page_id: str
    page_title: str
    database_id: str
    database_title: str
    url: str
    title: str
    summary: str
    chunk_number: int
    model: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    last_edited_time: Optional[str] = None

@dataclass
class NotionPage:
    page_id: str
    page_title: str
    database_id: str
    database_title: str
    url: str
    content: List[str]
    chunks: List[ProcessedChunk]
    last_edited_time: Optional[str] = None

@dataclass
class AgentDeps:
    system_prompt: str
    max_retries: int = 2
    max_chunks: int = 10
    max_tokens: int = 4096
    temperature: float = 0.5
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    vector_dimension: int = 1536
    similarity_threshold: float = 0.7

@dataclass
class CrawlerDeps:
    system_prompt: str
    chunk_size: int
    chunk_overlap: int
    max_concurrent_chunks: int
    delay_seconds: float

@dataclass
class GlobalDeps:
    df_people: pd.DataFrame
    df_pages: pd.DataFrame
    df_processing_pages: pd.DataFrame
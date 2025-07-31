# Imports necessários
import os
import re
import sys
import json
import random
import backoff
import asyncio
import logging
import nest_asyncio
import pandas as pd
from pathlib import Path
from crawler import Crawler
from dotenv import load_dotenv
from data_classes import NotionPage, CrawlerDeps, GlobalDeps
from prompts import get_crawler_prompt
from ai_providers import get_model_list, get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

sys.path.append('/work/')

nest_asyncio.apply()

load_dotenv()

logger = logging.getLogger(__name__)
df_people = pd.read_csv("data/df_people.csv")
df_pages = pd.read_csv("data/df_pages.csv")
df_processing_pages = pd.read_csv("data/df_processing_pages.csv")# Configuration

provider = 'huggingface'
model = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
prompt = get_crawler_prompt()


async def main( df_people, df_pages, df_processing_pages):
    # Create crawler dependencies
    crawler_deps = CrawlerDeps(
        system_prompt=prompt,
        chunk_size=500,
        chunk_overlap=100,
        max_concurrent_chunks=10,
        delay_seconds=0.6
    )
    global_deps = GlobalDeps(
        df_people=df_people,
        df_pages=df_pages,
        df_processing_pages=df_processing_pages
    )

    # Create crawler with embedding model string and dependencies
    crawler = Crawler(
        provider=provider,
        model=model,
        deps=crawler_deps,
        global_deps=global_deps
    )
    
    pages = [
        NotionPage(
            page_id= row.page_id,
            page_title= row.page_title,
            database_id= row.database_id,
            database_title= row.database_title,
            url= row.url,
            last_edited_time= row.last_edited_time,
            content = [],
            chunks = []
        )
        for row in df_processing_pages.itertuples()
    ]
    resultado = await crawler.run_crawler(pages)
    return resultado

if __name__ == "__main__":
    resultado = asyncio.run(main(df_people, df_pages, df_processing_pages))
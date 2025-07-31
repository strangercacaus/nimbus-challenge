import os
import json
import re
import asyncio
import logging
from typing import List, Dict, Union
from utils import async_retry

logger = logging.getLogger(__name__)

def get_model_list():
    return {
    "embedding":{
        "huggingface":{
            "all-MiniLM-L6-v2":372,
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct": 1536,
            # "pucpr/e5-base-portuguese-v1": 768,
            # "pucpr/e5-large-portuguese-v1": 1024,
            # "paraphrase-multilingual-MiniLM-L12-v2":384
        },
        # "openai": {
            # "text-embedding-3-small":1536,
            # "text-embedding-ada-002":1536,
        # }
    },
    "language":{
        "openai": [
            "gpt-4o-mini"
        ],
        "anthropic": [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-7-sonnet-latest",
            # "claude-sonnet-4-20250514",
            # "claude-opus-4-20250514"
        ],
        "huggingface": [
            # "rhaymison/flan-t5-portuguese-small-summarization",
            # "recogna-nlp/ptt5-base-summ:",
            # "google/flan-t5-large"
        ]
    }
}

def get_model_dimension(provider: str, embedding_model: str) -> int:
    embedding_model_list = get_model_list()["embedding"]
    return embedding_model_list[provider].get(embedding_model, 1536)

class openai_model:
    def __init__(
        self, 
        model_identifier:str = 'text-embedding-3-small', 
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI models. Install it with: pip install openai")
            
        self.model_identifier = model_identifier  
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @async_retry()
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model = self.model_identifier,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            return [0] * get_model_dimension('openai', self.model_identifier)

class huggingface_model:
    def __init__(
        self, 
        model_identifier:str = None, 
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFace models. Install it with: pip install sentence-transformers")
        
        self.model_identifier = model_identifier
        self.client = SentenceTransformer(model_identifier)

    @async_retry()
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector using sentence-transformers."""
        try:
            # Run the embedding in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.client.encode, text)
            return embedding.tolist()
        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            return [0.0] * get_model_dimension('huggingface', self.model_identifier)

def get_embedding_model(provider: str, model: str):
    """Factory function to get the appropriate embedding model based on configuration."""
    if provider.lower() == 'openai':
        return openai_model(model_identifier=model)
    elif provider.lower() == 'huggingface':
        return huggingface_model(model_identifier=model)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

def get_pydantic_model(provider: str, model_identifier: str):
    """Factory function to get the appropriate PydanticAI model based on the AI provider."""
    
    if provider == 'openai':
        try:
            from pydantic_ai.models.openai import OpenAIModel
        except ImportError:
            raise ImportError("pydantic-ai package with OpenAI support is required. Install it with: pip install pydantic-ai[openai]")
        return OpenAIModel(model_identifier)
    elif provider == 'anthropic':
        try:
            from pydantic_ai.models.anthropic import AnthropicModel
        except ImportError:
            raise ImportError("pydantic-ai package with Anthropic support is required. Install it with: pip install pydantic-ai[anthropic]")
        return AnthropicModel(model_identifier)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")
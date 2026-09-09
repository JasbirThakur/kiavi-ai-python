from typing import AsyncGenerator, List, Dict, Any, Optional
from openai import OpenAI
from groq import Groq
from app.config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_LLM_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL
)
from app.utils.logger import logger

def get_nvidia_client() -> Optional[OpenAI]:
    if NVIDIA_API_KEY:
        return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    return None

def get_groq_client() -> Optional[Groq]:
    if GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)
    return None

def call_llm_sync(messages: List[Dict[str, str]], temperature: float = 0.5, max_tokens: int = 1024) -> str:
    """Synchronous LLM call with Groq -> NVIDIA fallback"""
    # 1. Try Groq first for ultra-fast generation
    groq_client = get_groq_client()
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Groq sync call failed: {e}. Falling back to NVIDIA NIM...")

    # 2. Try NVIDIA NIM
    nvidia_client = get_nvidia_client()
    if nvidia_client:
        try:
            response = nvidia_client.chat.completions.create(
                model=NVIDIA_LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"NVIDIA sync call failed: {e}")

    return ""

async def stream_llm_tokens(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 1500
) -> AsyncGenerator[str, None]:
    """Asynchronous token streaming with fallback support"""
    groq_client = get_groq_client()
    if groq_client:
        try:
            stream = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
            return
        except Exception as e:
            logger.warning(f"Groq streaming failed: {e}. Trying NVIDIA...")

    nvidia_client = get_nvidia_client()
    if nvidia_client:
        try:
            stream = nvidia_client.chat.completions.create(
                model=NVIDIA_LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
            return
        except Exception as e:
            logger.error(f"NVIDIA streaming failed: {e}")

    yield "I'm currently unable to connect to the AI model. Please check the system configuration."


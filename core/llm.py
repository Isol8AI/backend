import httpx
import json
from core.config import settings
import logging
from typing import List, Dict, AsyncGenerator

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class LLMService:
    def __init__(self):
        self.token = settings.HUGGINGFACE_TOKEN
        self.api_url = settings.HF_API_URL

    def _build_messages(self, history: List[Dict[str, str]], current_message: str) -> List[Dict[str, str]]:
        """Build OpenAI-compatible messages array from conversation history."""
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant. Note: Previous assistant messages may contain '[Response from model-name]' prefixes - these are internal metadata annotations showing which AI model generated that response. Do not include such prefixes in your own responses; just respond naturally."
        })

        # Add conversation history
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": current_message
        })

        return messages

    async def generate_response_stream(
        self,
        current_message: str,
        history: List[Dict[str, str]] = None,
        model: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using HuggingFace Inference Providers API.
        Yields chunks of text as they arrive.
        """
        if not self.token:
            yield "Configuration Error: HUGGINGFACE_TOKEN is missing."
            return

        model_id = model or DEFAULT_MODEL
        url = f"{self.api_url}/chat/completions"

        messages = self._build_messages(history or [], current_message)

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": True
        }

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=120.0
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"Stream error {response.status_code}: {error_text}")
                        yield f"Error: {response.status_code}"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue

            except httpx.ReadTimeout:
                yield "Error: The model is taking too long to respond."
            except Exception as e:
                logger.error(f"Stream LLM Error: {str(e)}")
                yield f"Error: {str(e)}"


llm_service = LLMService()
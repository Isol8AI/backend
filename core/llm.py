import httpx
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.endpoint_url = settings.LLM_ENDPOINT_URL
        self.token = settings.HUGGINGFACE_TOKEN

    async def generate_response(self, prompt: str) -> str:
        """
        Sends the prompt to the Private HF Inference Endpoint.
        Handles Headers for auth and simple error checking.
        """
        if not self.endpoint_url:
            return "Configuration Error: LLM_ENDPOINT_URL is missing."

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False
            }
        }

        async with httpx.AsyncClient() as client:
            try:
                # Timeout set to 120s to allow for "Warm up" if paused
                response = await client.post(
                    self.endpoint_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=120.0
                )
                
                if response.status_code == 503:
                    # Cold Start / Loading
                    return "Model is waking up from cold storage. Please try again in 2 minutes."
                
                response.raise_for_status()
                
                # HF Text Generation Inference usually returns list of dicts
                data = response.json()
                if isinstance(data, list) and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
                
                return str(data)

            except httpx.ReadTimeout:
                return "The model is taking too long to respond (likely waking up). Please try again."
            except Exception as e:
                logger.error(f"LLM Error: {str(e)}")
                return f"Error contacting Secure Model: {str(e)}"

llm_service = LLMService()

 
# core/intelligence/inference_manager.py

from typing import Any, Dict, Optional
import logging
import requests
import json

logger = logging.getLogger(__name__)

class InferenceManager:
    """
    Manages AI model inference requests.
    Supports local and remote inference endpoints.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with optional config.
        Config example:
        {
            "mode": "local" or "remote",
            "remote_url": "http://api.example.com/infer",
            "api_key": "secret_api_key",
            "model_name": "gpt-4o-mini"
        }
        """
        self.config = config or {}
        self.mode = self.config.get("mode", "local")
        self.remote_url = self.config.get("remote_url")
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "default-model")

    def infer_local(self, prompt: str) -> str:
        """
        Stub for local inference.
        Replace with actual model integration (e.g., PyTorch, TensorFlow).
        """
        logger.debug(f"Local inference requested with prompt: {prompt}")
        # Placeholder logic
        response = f"[Local inference result for model {self.model_name}]"
        return response

    def infer_remote(self, prompt: str) -> str:
        """
        Calls remote API for inference.
        """
        if not self.remote_url or not self.api_key:
            logger.error("Remote URL or API key not configured.")
            raise ValueError("Remote inference requires 'remote_url' and 'api_key' in config.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 500
        }
        try:
            logger.debug(f"Sending remote inference request to {self.remote_url} with payload {payload}")
            response = requests.post(self.remote_url, headers=headers, data=json.dumps(payload), timeout=10)
            response.raise_for_status()
            result = response.json()
            # Assume response contains 'text' field with inference result
            return result.get("text", "[No text returned]")
        except requests.RequestException as e:
            logger.error(f"Remote inference failed: {e}")
            return "[Remote inference error]"

    def infer(self, prompt: str) -> str:
        """
        Main entry point for inference.
        Routes to local or remote based on configuration.
        """
        if self.mode == "local":
            return self.infer_local(prompt)
        elif self.mode == "remote":
            return self.infer_remote(prompt)
        else:
            logger.error(f"Unknown inference mode: {self.mode}")
            return "[Invalid inference mode]"


if __name__ == "__main__":
    # Basic demonstration of InferenceManager usage

    config_local = {"mode": "local", "model_name": "sebastian-core-v1"}
    im_local = InferenceManager(config_local)
    print(im_local.infer("What is the weather today?"))

    config_remote = {
        "mode": "remote",
        "remote_url": "https://api.example.com/inference",
        "api_key": "test_api_key_123",
        "model_name": "gpt-4o-mini"
    }
    im_remote = InferenceManager(config_remote)
    print(im_remote.infer("Tell me a witty remark."))

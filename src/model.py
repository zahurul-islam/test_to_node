# src/model.py
from typing import List
from llama_cpp import Llama
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeSequenceModel:
    def __init__(self, 
                 model_path: str, 
                 n_gpu_layers: int = -1,
                 context_length: int = 4096):
        """We need tp initialize the CodeGemma model using llama-cpp-python."""
        try:
            self.model_path = model_path
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_batch=512,
                #n_threads=16,
                n_threads=8,
                n_gpu_layers=n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                verbose=True
            )
            logger.info("Model loaded successfully with GPU support")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, prompt: str, max_length: int = 512) -> List[str]:
        """Predict node sequence from input prompt."""
        formatted_prompt = (
            "Given a text description, output a sequence of nodes that represent the required operations. "
            f"Text: {prompt}\nNodes:"
        )
        
        try:
            response = self.model(
                formatted_prompt,
                max_tokens=max_length,
                temperature=0.1,
                stop=["Text:", "\n\n"],
                echo=False
            )
            
            if response and 'choices' in response:
                nodes_text = response['choices'][0]['text'].strip()
                nodes = [node.strip() for node in nodes_text.split(',') if node.strip()]
                return nodes
            return []
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return []

    @classmethod
    def load(cls, model_dir: str) -> 'NodeSequenceModel':
        """Load model from directory."""
        model_path = os.path.join(model_dir, "model.gguf")
        if not os.path.exists(model_path):           
            model_path = model_dir
        return cls(model_path=model_path)

import ollama
from typing import List, Dict, Any

class Llama3Client:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
    
    def generate(
        self, 
        prompt: str, 
        context: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate text using LLaMA 3 with optional context."""
        full_prompt = f"""Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"""
        
        response = ollama.generate(
            model=self.model_name,
            prompt=full_prompt,
            options={
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
        
        return response['response']

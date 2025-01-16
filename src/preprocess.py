from typing import List, Dict
import re

def clean_text(text: str) -> str:
    """Clean and normalize text data"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    return text

def format_nodes(nodes: List[str]) -> str:
    """Format nodes into a consistent string format"""
    return ", ".join(nodes)

def preprocess_data(data: List[Dict]) -> List[Dict]:
    """Preprocess training examples"""
    processed_data = []
    
    for example in data:
        # Clean prompt text
        clean_prompt = clean_text(example["prompt"])
        
        # Format nodes
        formatted_nodes = format_nodes(example["nodes"])
        
        # Create training example
        processed_example = {
            "prompt": clean_prompt,
            "nodes": formatted_nodes,
            "text": f"Prompt: {clean_prompt}\nNodes: {formatted_nodes}"
        }
        
        processed_data.append(processed_example)
    
    return processed_data

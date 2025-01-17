"""
LLM Fine-tuning Pipeline
=======================

This script implements a complete pipeline for fine-tuning Large Language Models (LLMs) 
using the llama.cpp framework with HuggingFace datasets integration. It supports GGUF 
model formats and includes data preprocessing, training, and evaluation capabilities.

Requirements:
------------
- datasets
- llama-cpp-python
- transformers
- accelerate
- Custom modules: preprocess.py, train.py, test.py

Main Components:
--------------
1. Data Loading and Processing
2. Model and Tokenizer Initialization
3. Training Pipeline
4. Evaluation and Model Saving

Usage:
-----
python script.py

The script expects training data in JSON format at 'data/training_examples.json'
"""

import json
from datasets import Dataset
from preprocess import preprocess_data
from llama_cpp import Llama
from transformers import AutoTokenizer
from accelerate import Accelerator
from train import train_model
from test import test_model
import os

def load_data(file_path: str) -> list:
    """
    Load training examples from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing training data
        
    Returns:
        list: List of training examples
        
    Expected JSON format:
    {
        "training_examples": [
            {
                "input": "example input",
                "output": "example output"
            },
            ...
        ]
    }
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["training_examples"]

def split_data(data: list) -> tuple:
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        data (list): Complete dataset to be split
        
    Returns:
        tuple: (train_data, val_data, test_data)
            - train_data: 85% of the data
            - val_data: 13% of the data
            - test_data: 2% of the data
            
    Note:
        The split is deterministic (not randomized) based on list indexing
    """
    train_size = int(0.85 * len(data))
    val_size = int(0.13 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def main():
    """
    Main execution function that orchestrates the LLM fine-tuning pipeline.
    
    Pipeline Steps:
    1. Initialize hardware acceleration
    2. Load and split dataset
    3. Initialize GGUF model and tokenizer
    4. Prepare datasets with accelerator
    5. Train model
    6. Evaluate on test set
    7. Save fine-tuned model
    
    Configuration:
    - Model path: Expects GGUF model at specified absolute path
    - Context length: Set to 2048 tokens
    - CPU Threads: Uses 16 threads for computation
    - GPU Layers: All layers offloaded to GPU (-1)
    
    Outputs:
    - Trained model saved to ./fine_tuned_model/
    - Test results printed to console
    """
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load data
    data = load_data("data/training_examples.json")
    
    # Split data
    train_data, val_data, test_data = split_data(data)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Load GGUF model and tokenizer
    model_path = "/home/zahurul/Documents/work/playground/incari_16_01/models/codegemma-7b.Q4_K_M.gguf"
    tokenizer_name = "google/codegemma-7b"
    
    # Initialize GGUF model with optimized settings
    model = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context length
        n_threads=16,  # Number of CPU threads
        n_gpu_layers=-1  # Number of layers to offload to GPU (0 for CPU-only)
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Prepare datasets for training
    train_dataset = accelerator.prepare(train_dataset)
    val_dataset = accelerator.prepare(val_dataset)
    
    # Train model
    trained_model = train_model(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        accelerator
    )
    
    # Test model
    test_results = test_model(trained_model, tokenizer, test_dataset)
    print(f"Test Results: {test_results}")
    
    # Save model
    output_dir = "./fine_tuned_model"
    os.makedirs(output_dir, exist_ok=True)
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()

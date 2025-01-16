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
    """Load training examples from JSON file"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["training_examples"]

def split_data(data: list) -> tuple:
    """Split data into train, validation, and test sets"""
    train_size = int(0.85 * len(data))
    val_size = int(0.13 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def main():
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
    model_path = "/home/zahurul/Documents/work/playground/incari_16_01/models/codegemma-7b.Q4_K_M.gguf"  # Absolute path to model
    tokenizer_name = "google/codegemma-7b"  # Using HF tokenizer for CodeGemma
    
    # Initialize GGUF model
    model = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context length
        n_threads=4,  # Number of CPU threads
        n_gpu_layers=0  # Number of layers to offload to GPU (0 for CPU-only)
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

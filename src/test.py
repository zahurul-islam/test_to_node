from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict

def test_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_dataset: Dataset
) -> Dict[str, float]:
    """Evaluate model performance on test set"""
    
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for example in test_dataset:
        # Tokenize input
        inputs = tokenizer(
            example["prompt"],
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode and evaluate
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        expected_nodes = example["nodes"]
        
        # Simple evaluation - check if all expected nodes are present
        if all(node in generated_text for node in expected_nodes.split(", ")):
            total_correct += 1
        total_samples += 1
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_predictions": total_correct
    }

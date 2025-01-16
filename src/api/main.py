from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "./fine_tuned_model"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/generate")
async def generate_nodes(prompt: str):
    """Generate sequence of nodes from text prompt"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract nodes from generated text
    if "Nodes:" in generated_text:
        nodes = generated_text.split("Nodes:")[1].strip()
    else:
        nodes = generated_text
    
    return {
        "prompt": prompt,
        "nodes": nodes
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

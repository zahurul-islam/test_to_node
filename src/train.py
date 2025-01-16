from datasets import Dataset
from accelerate import Accelerator
from tqdm import tqdm
import os
from datetime import datetime

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    accelerator: Accelerator,
    checkpoint_dir: str = "checkpoints",
    checkpoint_freq: int = 1
):
    """Fine-tune the model using prompt-based learning with checkpointing
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for text processing
        train_dataset: Training dataset
        val_dataset: Validation dataset
        accelerator: Accelerator for distributed training
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Save checkpoint every N epochs
    """
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Convert datasets to lists
    train_data = train_dataset.to_list()
    val_data = val_dataset.to_list()
    
    # Training parameters
    batch_size = 1  # Llama model processes one example at a time
    num_epochs = 3
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Create instruction prompt template
    instruction = "Continue the code snippet while maintaining the same style and functionality:"
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Process in batches
        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i + batch_size]
            
            # Process one example at a time
            for example in batch:
                # Get the code from the example
                code = example["prompt"]
                
                # Split code into prompt and target
                split_point = int(len(code) * 0.8)  # Use 80% as prompt
                prompt = f"{instruction}\n{code[:split_point]}"
                target = code[split_point:]
                
                # Generate completion
                output = model.create_completion(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=512,
                    stop=["\n\n"]
                )
                
                # Calculate loss (simple token overlap)
                output_text = output["choices"][0]["text"]
                output_tokens = tokenizer.encode(output_text)
                target_tokens = tokenizer.encode(target)
                overlap = len(set(output_tokens) & set(target_tokens))
                loss = 1 - (overlap / len(target_tokens))
                
                # Generate new completion with updated context
                output = model.create_completion(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=512,
                    stop=["\n\n"]
                )
            
            epoch_loss += loss / len(batch)
        
        # Validation
        val_loss = 0
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i + batch_size]
            
            # Process one example at a time
            for example in batch:
                # Get the code from the example
                code = example["prompt"]  # Changed from "text" to "input"
                
                # Split code into prompt and target
                split_point = int(len(code) * 0.8)
                prompt = f"{instruction}\n{code[:split_point]}"
                target = code[split_point:]
                
                # Generate completion
                output = model.create_completion(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=512,
                    stop=["\n\n"]
                )
                
                # Calculate validation loss
                output_text = output["choices"][0]["text"]
                output_tokens = tokenizer.encode(output_text)
                target_tokens = tokenizer.encode(target)
                overlap = len(set(output_tokens) & set(target_tokens))
                val_loss += 1 - (overlap / len(target_tokens))
        
        # Calculate average losses
        avg_train_loss = epoch_loss / len(train_data)
        avg_val_loss = val_loss / len(val_data)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss: {avg_val_loss}")
        
        # Save checkpoint if it's time
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"New best model saved with val loss: {best_val_loss}")
    
    return model

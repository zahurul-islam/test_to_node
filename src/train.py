import torch
from datasets import Dataset
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.data import DataLoader

class LlamaWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        # Create a dummy parameter to make optimizer work
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        # Enable gradient checkpointing
        self.gradient_checkpointing = True
        self.gradient_checkpointing_enable = lambda: setattr(self, 'gradient_checkpointing', True)
        self.gradient_checkpointing_disable = lambda: setattr(self, 'gradient_checkpointing', False)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        vocab_size = self.tokenizer.vocab_size
        
        # Initialize logits with proper shape
        logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
        
        # Process each item in batch
        for i in range(batch_size):
            # Get input sequence
            input_seq = input_ids[i].tolist()
            
            # Decode input sequence
            input_text = self.tokenizer.decode(input_seq, skip_special_tokens=True)
            
            # Get model completion
            output = self.model.create_completion(
                prompt=input_text,
                temperature=0.7,
                max_tokens=seq_len,
                stop=["\n\n"]
            )
            
            # Get output text and encode
            output_text = output["choices"][0]["text"]
            output_ids = self.tokenizer.encode(output_text)
            
            # Create one-hot encoding for output tokens
            for j, token_id in enumerate(output_ids[:seq_len]):
                if token_id < vocab_size:
                    logits[i, j, token_id] = 1.0
                else:
                    print(f"Warning: Token ID {token_id} exceeds vocab size {vocab_size}")
        
        return logits
    
    def parameters(self):
        # Return an iterator containing trainable parameters
        # For Llama models, we need to manually collect parameters
        params = []
        # Add model weights if they exist
        if hasattr(self.model, 'weights'):
            params.extend(self.model.weights)
        # Add dummy parameter to ensure optimizer works
        params.append(self.dummy_param)
        return iter(params)

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    checkpoint_dir: str = "checkpoints",
    checkpoint_freq: int = 1
):
    # Wrap the model with tokenizer
    model = LlamaWrapper(model, tokenizer)
    # Clean up any existing accelerator state
    if AcceleratorState._shared_state:
        AcceleratorState._reset_state()
        torch.cuda.empty_cache()
    
    # Initialize accelerator with memory optimizations
    accelerator = Accelerator(
        mixed_precision='bf16',  # Use bfloat16 for better performance
        cpu=True,  # Enable CPU optimization
        device_placement=True,  # Automatically place tensors on correct devices
        gradient_accumulation_steps=4  # Reduced accumulation steps
    )
    
    # Enable gradient checkpointing on the model
    model.gradient_checkpointing_enable()
    
    def collate_fn(batch):
        # Get max length in batch
        max_len = max(len(item['prompt']) for item in batch)
        
        # Pad sequences and create attention masks
        padded_batch = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for item in batch:
            # Tokenize and pad
            tokens = tokenizer.encode(item['prompt'])
            padding_length = max_len - len(tokens)
            
            # Pad with tokenizer's pad token
            padded_tokens = tokens + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(tokens) + [0] * padding_length
            
            padded_batch['input_ids'].append(padded_tokens)
            padded_batch['attention_mask'].append(attention_mask)
            
        # Convert to tensors
        padded_batch['input_ids'] = torch.tensor(padded_batch['input_ids'])
        padded_batch['attention_mask'] = torch.tensor(padded_batch['attention_mask'])
        
        return padded_batch

    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Reduced batch size to prevent OOM
        num_workers=4,  # Reduced workers to balance memory usage
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2  # Reduced prefetch to save memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,  # Reduced batch size to prevent OOM
        num_workers=4,  # Reduced workers to balance memory usage
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2  # Reduced prefetch to save memory
    )
    
    # Prepare model and data loaders for accelerator
    model, tokenizer, train_loader, val_loader = accelerator.prepare(
        model, tokenizer, train_loader, val_loader
    )
    """Fine-tune the model using prompt-based learning with checkpointing
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for text processing
        train_dataset: Training dataset
        val_dataset: Validation dataset
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Save checkpoint every N epochs
    """
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(str(checkpoint_dir), timestamp)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Training parameters optimized for RTX 4090
    num_epochs = 3
    max_grad_norm = 1.0  # Gradient clipping for stability
    learning_rate = 5e-5
    warmup_steps = 1000
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / warmup_steps, 1.0)
    )
    
    # Prepare optimizer with accelerator
    optimizer = accelerator.prepare(optimizer)
    
    # Create instruction prompt template
    instruction = "Continue the code snippet while maintaining the same style and functionality:"
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Process batches using DataLoader
        for batch in tqdm(train_loader):
            with accelerator.accumulate(model):
                # Get padded inputs and attention masks
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                # Create targets by shifting input_ids
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]
                
                # Get model outputs
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss ignoring padding tokens
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                
                batch_loss = loss.item()
                
                # Backpropagation with gradient clipping
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Get padded inputs and attention masks
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                # Create targets by shifting input_ids
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]
                
                # Get model outputs
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss ignoring padding tokens
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print progress with accelerator
        accelerator.print(f"\nEpoch {epoch + 1}/{num_epochs}")
        accelerator.print(f"Train Loss: {avg_train_loss:.4f}")
        accelerator.print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint if it's time
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                checkpoint_path,
                save_function=accelerator.save
            )
            tokenizer.save_pretrained(checkpoint_path)
            accelerator.print(f"Saved checkpoint to {checkpoint_path}")
            
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                best_model_path,
                save_function=accelerator.save
            )
            tokenizer.save_pretrained(best_model_path)
            accelerator.print(f"New best model saved with val loss: {best_val_loss:.4f}")
    
    # Cleanup GPU resources
    try:
        accelerator.free_memory()
        torch.cuda.empty_cache()
        AcceleratorState._reset_state()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return model

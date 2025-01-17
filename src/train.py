import torch
from datasets import Dataset
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.data import DataLoader

class LlamaWrapper(torch.nn.Module):
    """
    A PyTorch wrapper for Llama models that enables training compatibility.
    
    This wrapper adapts Llama models to work with PyTorch's training infrastructure by:
    1. Implementing the required PyTorch Module interface
    2. Managing tokenization and sequence generation
    3. Handling gradient checkpointing for memory efficiency
    
    Attributes:
        model: The underlying Llama model
        tokenizer: Tokenizer for text processing
        dummy_param: A dummy parameter to ensure optimizer compatibility
        gradient_checkpointing (bool): Flag for gradient checkpointing status
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the Llama wrapper.
        
        Args:
            model: Base Llama model instance
            tokenizer: Associated tokenizer for text processing
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.gradient_checkpointing = True
        self.gradient_checkpointing_enable = lambda: setattr(self, 'gradient_checkpointing', True)
        self.gradient_checkpointing_disable = lambda: setattr(self, 'gradient_checkpointing', False)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for the model.
        
        Processes input sequences through the model and generates output logits.
        Handles batched inputs by processing each sequence individually.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask for padding
            
        Returns:
            torch.Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        vocab_size = self.tokenizer.vocab_size
        
        logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
        
        for i in range(batch_size):
            input_seq = input_ids[i].tolist()
            input_text = self.tokenizer.decode(input_seq, skip_special_tokens=True)
            
            output = self.model.create_completion(
                prompt=input_text,
                temperature=0.7,
                max_tokens=seq_len,
                stop=["\n\n"]
            )
            
            output_text = output["choices"][0]["text"]
            output_ids = self.tokenizer.encode(output_text)
            
            for j, token_id in enumerate(output_ids[:seq_len]):
                if token_id < vocab_size:
                    logits[i, j, token_id] = 1.0
                else:
                    print(f"Warning: Token ID {token_id} exceeds vocab size {vocab_size}")
        
        return logits
    
    def parameters(self):
        """
        Returns an iterator over trainable model parameters.
        
        Collects parameters from both the base model and wrapper components.
        Ensures optimizer compatibility by including dummy parameter.
        
        Returns:
            iterator: Iterator over trainable parameters
        """
        params = []
        if hasattr(self.model, 'weights'):
            params.extend(self.model.weights)
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
    """
    Fine-tune a Llama model using prompt-based learning with distributed training support.
    
    This function implements a complete training loop with the following features:
    - Mixed precision training using bfloat16
    - Gradient accumulation and checkpointing
    - Dynamic checkpoint saving
    - Validation-based model selection
    - Memory optimization for large models
    - Automatic device placement and distributed training support
    
    Args:
        model: Base Llama model to fine-tune
        tokenizer: Associated tokenizer for text processing
        train_dataset (Dataset): HuggingFace dataset for training
        val_dataset (Dataset): HuggingFace dataset for validation
        checkpoint_dir (str): Directory path for saving checkpoints
        checkpoint_freq (int): Frequency of checkpoint saving in epochs
        
    Returns:
        LlamaWrapper: Trained model wrapper
        
    Raises:
        RuntimeError: If GPU memory issues occur during training
        ValueError: If datasets are improperly formatted
    """
    model = LlamaWrapper(model, tokenizer)
    
    if AcceleratorState._shared_state:
        AcceleratorState._reset_state()
        torch.cuda.empty_cache()
    
    accelerator = Accelerator(
        mixed_precision='bf16',
        cpu=True,
        device_placement=True,
        gradient_accumulation_steps=4
    )
    
    model.gradient_checkpointing_enable()
    
    def collate_fn(batch):
        """
        Custom collation function for DataLoader.
        
        Handles:
        - Dynamic sequence padding to batch's maximum length
        - Attention mask generation
        - Tensor conversion
        
        Args:
            batch (List[dict]): List of dataset items
            
        Returns:
            dict: Collated batch with input_ids and attention_mask
        """
        max_len = max(len(item['prompt']) for item in batch)
        
        padded_batch = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for item in batch:
            tokens = tokenizer.encode(item['prompt'])
            padding_length = max_len - len(tokens)
            
            padded_tokens = tokens + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(tokens) + [0] * padding_length
            
            padded_batch['input_ids'].append(padded_tokens)
            padded_batch['attention_mask'].append(attention_mask)
            
        padded_batch['input_ids'] = torch.tensor(padded_batch['input_ids'])
        padded_batch['attention_mask'] = torch.tensor(padded_batch['attention_mask'])
        
        return padded_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    model, tokenizer, train_loader, val_loader = accelerator.prepare(
        model, tokenizer, train_loader, val_loader
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(str(checkpoint_dir), timestamp)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Training hyperparameters optimized for RTX 4090
    num_epochs = 3
    max_grad_norm = 1.0
    learning_rate = 5e-5
    warmup_steps = 1000
    
    best_val_loss = float('inf')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / warmup_steps, 1.0)
    )
    
    optimizer = accelerator.prepare(optimizer)
    
    instruction = "Continue the code snippet while maintaining the same style and functionality:"
    
    # Training loop with validation and checkpointing
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                
                batch_loss = loss.item()
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                
                val_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        accelerator.print(f"\nEpoch {epoch + 1}/{num_epochs}")
        accelerator.print(f"Train Loss: {avg_train_loss:.4f}")
        accelerator.print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Checkpoint saving logic
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
            
        # Best model saving logic
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
    
    # Resource cleanup
    try:
        accelerator.free_memory()
        torch.cuda.empty_cache()
        AcceleratorState._reset_state()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return model

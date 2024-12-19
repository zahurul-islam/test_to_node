import torch
from transformers import Trainer, TrainingArguments
from typing import Dict
import wandb

def train_model(
    model,
    train_dataset,
    val_dataset,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float
) -> Dict:
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    return train_result

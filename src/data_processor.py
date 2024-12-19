import json
import random
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_split_data(data_path: str, train_split: float, val_split: float, test_split: float) -> Tuple[List, List, List]:
    """Load and split the data into train, validation, and test sets."""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        examples = data["training_examples"]
        random.shuffle(examples)
        
        # Calculate splits
        train_size = int(len(examples) * train_split)
        val_size = int(len(examples) * val_split)
        
        # Split data
        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:]
        
        logger.info(f"Data split complete - Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
        return train_examples, val_examples, test_examples
        
    except Exception as e:
        logger.error(f"Error loading or splitting data: {e}")
        raise

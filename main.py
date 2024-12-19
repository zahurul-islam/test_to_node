import logging
from pathlib import Path
import json
from tqdm import tqdm
from src.config import *
from src.data_processor import load_and_split_data
from src.model import NodeSequenceModel
from src.evaluator import NodeSequenceEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load and split data
    logger.info("Loading and splitting data...")
    train_examples, val_examples, test_examples = load_and_split_data(
        str(TRAIN_DATA_PATH),
        TRAIN_SPLIT,
        VAL_SPLIT,
        TEST_SPLIT
    )
    
    # Initialize model
    logger.info("Initializing CodeGemma model...")
    model = NodeSequenceModel(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        context_length=CONTEXT_LENGTH
    )
    
    # Evaluate on validation set
    logger.info("Evaluating model on validation set...")
    val_predictions = []
    val_actual = []
    
    for example in tqdm(val_examples, desc="Validating"):
        pred_nodes = model.predict(example["prompt"])
        val_predictions.append(pred_nodes)
        val_actual.append(example["nodes"])
    
    # Calculate metrics
    evaluator = NodeSequenceEvaluator()
    metrics = evaluator.calculate_metrics(val_predictions, val_actual)
    
    logger.info("Validation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save predictions and metrics
    results = {
        "metrics": metrics,
        "predictions": [
            {
                "prompt": ex["prompt"],
                "predicted_nodes": pred,
                "actual_nodes": ex["nodes"]
            }
            for ex, pred in zip(val_examples, val_predictions)
        ]
    }
    
    with open(OUTPUT_DIR / "validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {OUTPUT_DIR / 'validation_results.json'}")

if __name__ == "__main__":
    main()

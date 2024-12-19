from pathlib import Path

# Model configuration
MODEL_PATH = "model/codegemma-7b-Q6_K.gguf" 
MAX_LENGTH = 512
BATCH_SIZE = 32  # Reduced due to larger model
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.13
TEST_SPLIT = 0.02
CONTEXT_LENGTH = 4096 
N_GPU_LAYERS = -1 

# Data configuration
DATA_DIR = Path("data")
TRAIN_DATA_PATH = DATA_DIR / "training_data.json"
OUTPUT_DIR = Path("output")
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"

# We need to create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
